#include <jni.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <vector>

namespace {

struct PoseReprojectionResidual {
    PoseReprojectionResidual(const double* world_point,
                             double observed_x,
                             double observed_y,
                             const double* intrinsics)
            : observed_x(observed_x),
              observed_y(observed_y),
              fx(intrinsics[0]),
              skew(intrinsics[1]),
              cx(intrinsics[2]),
              fy(intrinsics[3]),
              cy(intrinsics[4]) {
        std::copy(world_point, world_point + 3, this->world_point);
    }

    template <typename T>
    bool operator()(const T* angle_axis, const T* translation, T* residuals) const {
        const T point[3] = {
                T(world_point[0]), T(world_point[1]), T(world_point[2])};
        T rotated[3];
        ceres::AngleAxisRotatePoint(angle_axis, point, rotated);
        const T x = rotated[0] + translation[0];
        const T y = rotated[1] + translation[1];
        const T z = rotated[2] + translation[2];
        const T safe_z = z > T(1e-8) ? z : T(1e-8);
        residuals[0] = T(fx) * x / safe_z + T(skew) * y / safe_z + T(cx) - T(observed_x);
        residuals[1] = T(fy) * y / safe_z + T(cy) - T(observed_y);
        return true;
    }

    double world_point[3];
    double observed_x;
    double observed_y;
    double fx;
    double skew;
    double cx;
    double fy;
    double cy;
};

bool AllFinite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
}

jdoubleArray EmptyResult(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_CameraPoseRefiner_nativeRefine(
        JNIEnv* env,
        jobject,
        jdoubleArray object_points_array,
        jdoubleArray image_points_array,
        jdoubleArray intrinsics_array,
        jdoubleArray initial_pose_array,
        jdouble huber_scale_px,
        jint max_iterations,
        jdouble gradient_tolerance) {
    try {
        if (!object_points_array || !image_points_array || !intrinsics_array ||
            !initial_pose_array) {
            return EmptyResult(env);
        }

        const jsize object_values = env->GetArrayLength(object_points_array);
        const jsize image_values = env->GetArrayLength(image_points_array);
        const jsize point_count = object_values / 3;
        if (object_values < 12 || object_values % 3 != 0 ||
            image_values != point_count * 2 ||
            env->GetArrayLength(intrinsics_array) != 5 ||
            env->GetArrayLength(initial_pose_array) != 6 ||
            max_iterations <= 0 || !std::isfinite(huber_scale_px) || huber_scale_px <= 0.0 ||
            !std::isfinite(gradient_tolerance) || gradient_tolerance <= 0.0) {
            return EmptyResult(env);
        }

        std::vector<double> object_points(object_values);
        std::vector<double> image_points(image_values);
        std::vector<double> intrinsics(5);
        std::vector<double> initial_pose(6);
        env->GetDoubleArrayRegion(object_points_array, 0, object_values, object_points.data());
        env->GetDoubleArrayRegion(image_points_array, 0, image_values, image_points.data());
        env->GetDoubleArrayRegion(intrinsics_array, 0, 5, intrinsics.data());
        env->GetDoubleArrayRegion(initial_pose_array, 0, 6, initial_pose.data());
        if (env->ExceptionCheck() || !AllFinite(object_points) || !AllFinite(image_points) ||
            !AllFinite(intrinsics) || !AllFinite(initial_pose) ||
            intrinsics[0] <= 0.0 || intrinsics[3] <= 0.0) {
            return EmptyResult(env);
        }

        std::vector<double> angle_axis(initial_pose.begin(), initial_pose.begin() + 3);
        std::vector<double> translation(initial_pose.begin() + 3, initial_pose.end());
        ceres::Problem problem;
        for (jsize index = 0; index < point_count; ++index) {
            const double* world_point = object_points.data() + index * 3;
            const double observed_x = image_points[index * 2];
            const double observed_y = image_points[index * 2 + 1];
            auto* cost = new ceres::AutoDiffCostFunction<PoseReprojectionResidual, 2, 3, 3>(
                    new PoseReprojectionResidual(world_point, observed_x, observed_y,
                                                 intrinsics.data()));
            problem.AddResidualBlock(cost, new ceres::HuberLoss(huber_scale_px),
                                     angle_axis.data(), translation.data());
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.max_num_iterations = max_iterations;
        options.num_threads = 1;
        options.gradient_tolerance = gradient_tolerance;
        options.minimizer_progress_to_stdout = false;
        options.logging_type = ceres::SILENT;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        const double allowed_cost_increase =
                1e-9 * std::max(1.0, summary.initial_cost);
        if (!summary.IsSolutionUsable() || !std::isfinite(summary.initial_cost) ||
            !std::isfinite(summary.final_cost) ||
            summary.final_cost > summary.initial_cost + allowed_cost_increase ||
            !AllFinite(angle_axis) || !AllFinite(translation)) {
            return EmptyResult(env);
        }

        const double output_values[10] = {
                1.0,
                summary.initial_cost,
                summary.final_cost,
                static_cast<double>(summary.iterations.size()),
                angle_axis[0], angle_axis[1], angle_axis[2],
                translation[0], translation[1], translation[2]};
        jdoubleArray result = env->NewDoubleArray(10);
        if (result) env->SetDoubleArrayRegion(result, 0, 10, output_values);
        return result;
    } catch (const std::exception&) {
        return EmptyResult(env);
    } catch (...) {
        return EmptyResult(env);
    }
}
