#include <jni.h>

#include "colmap_p4pf_absolute_pose_core.h"

#include <cmath>
#include <exception>
#include <limits>
#include <vector>

namespace {

constexpr jsize kResultHeaderSize = 21;

jdoubleArray EmptyResult(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

bool IsInteger(double value) {
    return std::isfinite(value) && value >= 0.0 && std::floor(value) == value &&
           value <= static_cast<double>(std::numeric_limits<int>::max());
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_colmap_ColmapP4pfAbsolutePoseEstimator_nativeEstimate(
        JNIEnv* env,
        jobject,
        jdoubleArray object_points_array,
        jdoubleArray image_points_array,
        jdoubleArray intrinsics_array,
        jdoubleArray options_array,
        jint random_seed) {
    try {
        if (!object_points_array || !image_points_array || !intrinsics_array || !options_array) {
            return EmptyResult(env);
        }
        const jsize object_values = env->GetArrayLength(object_points_array);
        const jsize image_values = env->GetArrayLength(image_points_array);
        if (object_values < 12 || object_values % 3 != 0 ||
            object_values / 3 > std::numeric_limits<jsize>::max() / 2 ||
            image_values != (object_values / 3) * 2 ||
            env->GetArrayLength(intrinsics_array) != 5 ||
            env->GetArrayLength(options_array) != 8) {
            return EmptyResult(env);
        }

        std::vector<double> object_values_copy(static_cast<size_t>(object_values));
        std::vector<double> image_values_copy(static_cast<size_t>(image_values));
        double intrinsics_values[5] = {};
        double options_values[8] = {};
        env->GetDoubleArrayRegion(object_points_array, 0, object_values,
                                  object_values_copy.data());
        env->GetDoubleArrayRegion(image_points_array, 0, image_values,
                                  image_values_copy.data());
        env->GetDoubleArrayRegion(intrinsics_array, 0, 5, intrinsics_values);
        env->GetDoubleArrayRegion(options_array, 0, 8, options_values);
        if (env->ExceptionCheck()) return nullptr;
        if (!IsInteger(options_values[4]) || !IsInteger(options_values[5]) ||
            random_seed < -1) {
            return EmptyResult(env);
        }

        std::vector<Eigen::Vector3d> world_points(static_cast<size_t>(object_values / 3));
        std::vector<Eigen::Vector2d> image_points(static_cast<size_t>(image_values / 2));
        for (size_t index = 0; index < world_points.size(); ++index) {
            world_points[index] = Eigen::Vector3d(object_values_copy[index * 3],
                                                   object_values_copy[index * 3 + 1],
                                                   object_values_copy[index * 3 + 2]);
            image_points[index] = Eigen::Vector2d(image_values_copy[index * 2],
                                                   image_values_copy[index * 2 + 1]);
        }

        const tracking::P4pfCameraIntrinsics intrinsics{
                intrinsics_values[0], intrinsics_values[1], intrinsics_values[2],
                intrinsics_values[3], intrinsics_values[4]};
        tracking::P4pfAbsolutePoseOptions options;
        options.max_error_px = options_values[0];
        options.min_inlier_ratio = options_values[1];
        options.confidence = options_values[2];
        options.dynamic_trial_multiplier = options_values[3];
        options.min_num_trials = static_cast<int>(options_values[4]);
        options.max_num_trials = static_cast<int>(options_values[5]);
        options.refinement_loss_scale = options_values[6];
        options.refinement_gradient_tolerance = options_values[7];
        options.random_seed = random_seed;

        tracking::P4pfAbsolutePoseResult estimate;
        if (!tracking::EstimateAbsolutePoseP4PF(image_points, world_points, intrinsics,
                                               options, &estimate)) {
            return EmptyResult(env);
        }
        if (estimate.inlier_indices.size() >
            static_cast<size_t>(std::numeric_limits<jsize>::max() - kResultHeaderSize)) {
            return EmptyResult(env);
        }

        const jsize result_size = kResultHeaderSize +
                                  static_cast<jsize>(estimate.inlier_indices.size());
        std::vector<jdouble> output(static_cast<size_t>(result_size), 0.0);
        output[0] = 1.0;
        output[1] = static_cast<double>(estimate.num_trials);
        output[2] = estimate.initial_focal_length;
        output[3] = estimate.refined_fx;
        output[4] = estimate.refined_fy;
        output[5] = estimate.initial_cost;
        output[6] = estimate.final_cost;
        output[7] = static_cast<double>(estimate.refinement_iterations);
        output[8] = static_cast<double>(estimate.inlier_indices.size());
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                output[9 + row * 3 + column] = estimate.rotation(row, column);
            }
            output[18 + row] = estimate.translation[row];
        }
        for (size_t index = 0; index < estimate.inlier_indices.size(); ++index) {
            output[static_cast<size_t>(kResultHeaderSize) + index] =
                    static_cast<double>(estimate.inlier_indices[index]);
        }
        for (double value : output) {
            if (!std::isfinite(value)) return EmptyResult(env);
        }

        jdoubleArray result = env->NewDoubleArray(result_size);
        if (result == nullptr) return nullptr;
        env->SetDoubleArrayRegion(result, 0, result_size, output.data());
        if (env->ExceptionCheck()) return nullptr;
        return result;
    } catch (const std::exception&) {
        return EmptyResult(env);
    } catch (...) {
        return EmptyResult(env);
    }
}
