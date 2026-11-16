#include <jni.h>

#include "epnp_pose_estimator_core.h"

#include <cmath>
#include <exception>
#include <limits>
#include <vector>

namespace {

jdoubleArray EmptyResult(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_EpnpPoseEstimator_nativeEstimate(
        JNIEnv* env,
        jobject,
        jdoubleArray object_points_array,
        jdoubleArray image_points_array,
        jdoubleArray intrinsics_array) {
    try {
        if (!object_points_array || !image_points_array || !intrinsics_array) {
            return EmptyResult(env);
        }
        const jsize object_values = env->GetArrayLength(object_points_array);
        const jsize image_values = env->GetArrayLength(image_points_array);
        if (object_values < 12 || object_values % 3 != 0 ||
            object_values / 3 > std::numeric_limits<jsize>::max() / 2 ||
            image_values != (object_values / 3) * 2 ||
            env->GetArrayLength(intrinsics_array) != 5) {
            return EmptyResult(env);
        }

        std::vector<double> object_values_copy(static_cast<size_t>(object_values));
        std::vector<double> image_values_copy(static_cast<size_t>(image_values));
        double intrinsics_values[5] = {};
        env->GetDoubleArrayRegion(object_points_array, 0, object_values, object_values_copy.data());
        env->GetDoubleArrayRegion(image_points_array, 0, image_values, image_values_copy.data());
        env->GetDoubleArrayRegion(intrinsics_array, 0, 5, intrinsics_values);
        if (env->ExceptionCheck()) return nullptr;

        std::vector<Eigen::Vector3d> world_points(static_cast<size_t>(object_values / 3));
        std::vector<Eigen::Vector2d> image_points(static_cast<size_t>(image_values / 2));
        for (size_t i = 0; i < world_points.size(); ++i) {
            world_points[i] = Eigen::Vector3d(object_values_copy[i * 3],
                                              object_values_copy[i * 3 + 1],
                                              object_values_copy[i * 3 + 2]);
            image_points[i] = Eigen::Vector2d(image_values_copy[i * 2],
                                              image_values_copy[i * 2 + 1]);
        }
        const tracking::EpnpCameraIntrinsics intrinsics{
                intrinsics_values[0], intrinsics_values[1], intrinsics_values[2],
                intrinsics_values[3], intrinsics_values[4]};
        tracking::EpnpPoseEstimate estimate;
        if (!tracking::EstimateEpnpPose(image_points, world_points, intrinsics, &estimate)) {
            return EmptyResult(env);
        }

        jdouble output[14] = {};
        output[0] = 1.0;
        output[1] = estimate.reprojection_error;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                output[2 + row * 3 + column] = estimate.rotation(row, column);
            }
            output[11 + row] = estimate.translation[row];
        }
        for (double value : output) {
            if (!std::isfinite(value)) return EmptyResult(env);
        }

        jdoubleArray result = env->NewDoubleArray(14);
        if (result == nullptr) return nullptr;
        env->SetDoubleArrayRegion(result, 0, 14, output);
        if (env->ExceptionCheck()) return nullptr;
        return result;
    } catch (const std::exception&) {
        return EmptyResult(env);
    } catch (...) {
        return EmptyResult(env);
    }
}
