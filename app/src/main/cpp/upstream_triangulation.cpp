// SPDX-License-Identifier: BSD-3-Clause

#include <jni.h>

#include <cmath>

#include <Eigen/Dense>

namespace {

bool TriangulatePoint(const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>& cam1_from_world,
                      const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>& cam2_from_world,
                      const Eigen::Vector2d& cam_point1,
                      const Eigen::Vector2d& cam_point2,
                      Eigen::Vector3d* xyz) {
    if (xyz == nullptr) {
        return false;
    }

    Eigen::Matrix4d A;
    A.row(0) = cam_point1(0) * cam1_from_world.row(2) - cam1_from_world.row(0);
    A.row(1) = cam_point1(1) * cam1_from_world.row(2) - cam1_from_world.row(1);
    A.row(2) = cam_point2(0) * cam2_from_world.row(2) - cam2_from_world.row(0);
    A.row(3) = cam_point2(1) * cam2_from_world.row(2) - cam2_from_world.row(1);

    const Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    if (svd.info() != Eigen::Success || svd.matrixV()(3, 3) == 0) {
        return false;
    }

    *xyz = svd.matrixV().col(3).hnormalized();
    return xyz->allFinite();
}

jdoubleArray EmptyResult(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_UpstreamTriangulation_nativeTriangulate(
        JNIEnv* env,
        jobject,
        jdoubleArray first_projection_array,
        jdoubleArray second_projection_array,
        jdouble first_x,
        jdouble first_y,
        jdouble second_x,
        jdouble second_y) {
    if (first_projection_array == nullptr || second_projection_array == nullptr ||
        env->GetArrayLength(first_projection_array) != 12 ||
        env->GetArrayLength(second_projection_array) != 12) {
        return EmptyResult(env);
    }

    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> first_projection;
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> second_projection;
    env->GetDoubleArrayRegion(first_projection_array, 0, 12, first_projection.data());
    env->GetDoubleArrayRegion(second_projection_array, 0, 12, second_projection.data());
    if (env->ExceptionCheck() || !first_projection.allFinite() ||
        !second_projection.allFinite() || !std::isfinite(first_x) ||
        !std::isfinite(first_y) || !std::isfinite(second_x) ||
        !std::isfinite(second_y)) {
        return EmptyResult(env);
    }

    Eigen::Vector3d point;
    if (!TriangulatePoint(
            first_projection,
            second_projection,
            Eigen::Vector2d(first_x, first_y),
            Eigen::Vector2d(second_x, second_y),
            &point)) {
        return EmptyResult(env);
    }

    jdoubleArray result = env->NewDoubleArray(3);
    if (result != nullptr) {
        const jdouble values[3] = {point.x(), point.y(), point.z()};
        env->SetDoubleArrayRegion(result, 0, 3, values);
    }
    return result;
}
