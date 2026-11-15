#include <jni.h>
#include <android/log.h>

#include "global_bundle_adjuster_core.h"
#include "glomap_progress_callback.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <vector>

namespace {

jdoubleArray EmptyResult(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_glomap_GlomapGlobalBundleAdjuster_nativeSolve(
        JNIEnv* env,
        jobject,
        jdoubleArray camera_centers_array,
        jdoubleArray rotations_array,
        jdoubleArray points_array,
        jintArray frame_indices_array,
        jintArray track_indices_array,
        jdoubleArray image_points_array,
        jdoubleArray intrinsics_array,
        jboolean refine_principal_point,
        jint image_width,
        jint image_height,
        jint max_iterations,
        jint num_threads,
        jobject progress_listener) {
    try {
        if (!camera_centers_array || !rotations_array || !points_array ||
            !frame_indices_array || !track_indices_array || !image_points_array ||
            !intrinsics_array) {
            return EmptyResult(env);
        }

        const jsize center_values = env->GetArrayLength(camera_centers_array);
        const jsize rotation_values = env->GetArrayLength(rotations_array);
        const jsize point_values = env->GetArrayLength(points_array);
        const jsize observation_count = env->GetArrayLength(frame_indices_array);
        const jsize max_size = std::numeric_limits<jsize>::max();
        if (center_values == 0 || center_values % 3 != 0 ||
            center_values / 3 > max_size / 9 ||
            rotation_values != (center_values / 3) * 9 ||
            point_values == 0 || point_values % 3 != 0 ||
            observation_count < 2 || observation_count > max_size / 2 ||
            env->GetArrayLength(track_indices_array) != observation_count ||
            env->GetArrayLength(image_points_array) != observation_count * 2 ||
            env->GetArrayLength(intrinsics_array) != 5 ||
            (refine_principal_point != JNI_FALSE &&
             (image_width <= 0 || image_height <= 0)) ||
            max_iterations <= 0 || num_threads <= 0) {
            return EmptyResult(env);
        }

        std::vector<double> centers(static_cast<size_t>(center_values));
        std::vector<double> rotations(static_cast<size_t>(rotation_values));
        std::vector<double> points(static_cast<size_t>(point_values));
        std::vector<int> frame_indices(static_cast<size_t>(observation_count));
        std::vector<int> track_indices(static_cast<size_t>(observation_count));
        std::vector<double> image_points(static_cast<size_t>(observation_count) * 2);
        std::vector<double> intrinsics(5);

        env->GetDoubleArrayRegion(camera_centers_array, 0, center_values, centers.data());
        env->GetDoubleArrayRegion(rotations_array, 0, rotation_values, rotations.data());
        env->GetDoubleArrayRegion(points_array, 0, point_values, points.data());
        env->GetIntArrayRegion(frame_indices_array, 0, observation_count, frame_indices.data());
        env->GetIntArrayRegion(track_indices_array, 0, observation_count, track_indices.data());
        env->GetDoubleArrayRegion(image_points_array, 0, observation_count * 2, image_points.data());
        env->GetDoubleArrayRegion(intrinsics_array, 0, 5, intrinsics.data());
        if (env->ExceptionCheck()) return nullptr;

        trackstudio::JniProgressCallback progress_callback(
                env, progress_listener, std::min(max_iterations, 60));
        if (env->ExceptionCheck()) return nullptr;
        std::function<bool(int, int)> notify_progress;
        if (progress_callback.valid()) {
            notify_progress = [&progress_callback](int iteration, int total) {
                return progress_callback.Notify(iteration, total);
            };
        }

        tracking::GlobalBundleAdjustmentResult solution;
        if (!tracking::RunGlobalBundleAdjustment(
                    centers, rotations, points, frame_indices, track_indices,
                    image_points, intrinsics, max_iterations, num_threads,
                    &solution, notify_progress,
                    refine_principal_point != JNI_FALSE,
                    image_width, image_height)) {
            if (env->ExceptionCheck()) return nullptr;
            return EmptyResult(env);
        }
        if (progress_callback.aborted()) return nullptr;

        const size_t maximum_output_size =
                static_cast<size_t>(std::numeric_limits<jsize>::max());
        size_t output_size = 5;
        for (size_t field_size : {solution.camera_centers.size(), solution.rotations.size(),
                                  solution.points.size(), solution.intrinsics.size()}) {
            if (field_size > maximum_output_size - output_size) return EmptyResult(env);
            output_size += field_size;
        }
        std::vector<double> output;
        output.reserve(output_size);
        output.push_back(1.0);
        output.push_back(solution.initial_cost);
        output.push_back(solution.final_cost);
        output.push_back(static_cast<double>(solution.iterations));
        output.push_back(solution.positive_depth_ratio);
        output.insert(output.end(), solution.camera_centers.begin(), solution.camera_centers.end());
        output.insert(output.end(), solution.rotations.begin(), solution.rotations.end());
        output.insert(output.end(), solution.points.begin(), solution.points.end());
        output.insert(output.end(), solution.intrinsics.begin(), solution.intrinsics.end());

        jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output.size()));
        if (result == nullptr) return nullptr;
        env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output.size()), output.data());
        if (env->ExceptionCheck()) return nullptr;
        return result;
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, "TrackStudioGlobalBA",
                            "Native solve exception: %s", error.what());
        return EmptyResult(env);
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "TrackStudioGlobalBA",
                            "Unknown native solve exception.");
        return EmptyResult(env);
    }
}
