#include <jni.h>
#include <android/log.h>

#include "global_positioning_core.h"
#include "glomap_progress_callback.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace {

jdoubleArray EmptyResult(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_glomap_GlomapNativeSolver_nativeSolve(
        JNIEnv* env,
        jobject,
        jdoubleArray frame_centers_array,
        jdoubleArray points_array,
        jintArray frame_indices_array,
        jintArray track_indices_array,
        jdoubleArray directions_array,
        jdoubleArray initial_scales_array,
        jdoubleArray loss_scales_array,
        jint max_iterations,
        jint num_threads,
        jobject progress_listener) {
    try {
        if (!frame_centers_array || !points_array || !frame_indices_array ||
            !track_indices_array || !directions_array || !initial_scales_array ||
            !loss_scales_array) {
            return EmptyResult(env);
        }

        const jsize frame_value_count = env->GetArrayLength(frame_centers_array);
        const jsize point_value_count = env->GetArrayLength(points_array);
        const jsize constraint_count = env->GetArrayLength(frame_indices_array);
        const jsize max_size = std::numeric_limits<jsize>::max();
        if (frame_value_count == 0 || frame_value_count % 3 != 0 ||
            point_value_count == 0 || point_value_count % 3 != 0 ||
            constraint_count == 0 || constraint_count > max_size / 3 ||
            env->GetArrayLength(track_indices_array) != constraint_count ||
            env->GetArrayLength(directions_array) != constraint_count * 3 ||
            env->GetArrayLength(initial_scales_array) != constraint_count ||
            env->GetArrayLength(loss_scales_array) != constraint_count ||
            max_iterations <= 0 || num_threads <= 0) {
            return EmptyResult(env);
        }

        std::vector<double> frame_centers(static_cast<size_t>(frame_value_count));
        std::vector<double> points(static_cast<size_t>(point_value_count));
        std::vector<int> frame_indices(static_cast<size_t>(constraint_count));
        std::vector<int> track_indices(static_cast<size_t>(constraint_count));
        std::vector<double> directions(static_cast<size_t>(constraint_count) * 3);
        std::vector<double> scales(static_cast<size_t>(constraint_count));
        std::vector<double> loss_scales(static_cast<size_t>(constraint_count));

        env->GetDoubleArrayRegion(frame_centers_array, 0, frame_value_count,
                                  frame_centers.data());
        env->GetDoubleArrayRegion(points_array, 0, point_value_count, points.data());
        env->GetIntArrayRegion(frame_indices_array, 0, constraint_count,
                               frame_indices.data());
        env->GetIntArrayRegion(track_indices_array, 0, constraint_count,
                               track_indices.data());
        env->GetDoubleArrayRegion(directions_array, 0, constraint_count * 3,
                                  directions.data());
        env->GetDoubleArrayRegion(initial_scales_array, 0, constraint_count,
                                  scales.data());
        env->GetDoubleArrayRegion(loss_scales_array, 0, constraint_count,
                                  loss_scales.data());
        if (env->ExceptionCheck()) return nullptr;

        trackstudio::JniProgressCallback progress_callback(
                env, progress_listener, std::min(max_iterations, 100));
        if (env->ExceptionCheck()) return nullptr;
        std::function<bool(int, int)> notify_progress;
        if (progress_callback.valid()) {
            notify_progress = [&progress_callback](int iteration, int total) {
                return progress_callback.Notify(iteration, total);
            };
        }

        tracking::GlobalPositioningResult solution;
        if (!tracking::RunGlobalPositioning(
                    frame_centers, points, frame_indices, track_indices,
                    directions, scales, loss_scales, max_iterations, num_threads,
                    &solution, notify_progress)) {
            if (env->ExceptionCheck()) return nullptr;
            if (progress_callback.aborted()) return nullptr;
            return EmptyResult(env);
        }
        if (progress_callback.aborted()) return nullptr;
        __android_log_print(ANDROID_LOG_INFO, "TrackStudioGlobalPosition", "%s",
                            solution.solver_report.c_str());

        const size_t maximum_output =
                static_cast<size_t>(std::numeric_limits<jsize>::max());
        size_t output_size = 4;
        for (size_t size : {solution.frame_centers.size(), solution.points.size(),
                            solution.scales.size()}) {
            if (size > maximum_output - output_size) return EmptyResult(env);
            output_size += size;
        }
        std::vector<double> output;
        output.reserve(output_size);
        output.push_back(1.0);
        output.push_back(solution.initial_cost);
        output.push_back(solution.final_cost);
        output.push_back(static_cast<double>(solution.iterations));
        output.insert(output.end(), solution.frame_centers.begin(),
                      solution.frame_centers.end());
        output.insert(output.end(), solution.points.begin(), solution.points.end());
        output.insert(output.end(), solution.scales.begin(), solution.scales.end());

        jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output.size()));
        if (result == nullptr) return nullptr;
        env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output.size()),
                                  output.data());
        if (env->ExceptionCheck()) return nullptr;
        return result;
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, "TrackStudioGlobalPosition",
                            "Native solve exception: %s", error.what());
        return EmptyResult(env);
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "TrackStudioGlobalPosition",
                            "Unknown native solve exception.");
        return EmptyResult(env);
    }
}
