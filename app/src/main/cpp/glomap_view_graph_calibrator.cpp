#include <jni.h>
#include <android/log.h>

#include <exception>

#include "view_graph_calibration_core.h"

#include <limits>
#include <vector>

namespace {

jdoubleArray EmptyResult(JNIEnv* env) {
    return env->NewDoubleArray(0);
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_glomap_GlomapViewGraphCalibrator_nativeSolve(
        JNIEnv* env,
        jobject,
        jdoubleArray fundamental_matrices_array,
        jdouble principal_x,
        jdouble principal_y,
        jdouble initial_focal_length,
        jint max_num_iterations,
        jint num_threads) {
    try {
        if (fundamental_matrices_array == nullptr) return EmptyResult(env);
        const jsize value_count = env->GetArrayLength(fundamental_matrices_array);
        if (value_count <= 0 || value_count % 9 != 0 ||
            max_num_iterations <= 0 || num_threads <= 0) {
            return EmptyResult(env);
        }
        std::vector<double> fundamentals(static_cast<size_t>(value_count));
        env->GetDoubleArrayRegion(fundamental_matrices_array, 0, value_count,
                                  fundamentals.data());
        if (env->ExceptionCheck()) return nullptr;

        tracking::ViewGraphCalibrationResult solution;
        if (!tracking::RunViewGraphFocalCalibration(
                    fundamentals, principal_x, principal_y,
                    initial_focal_length, max_num_iterations, num_threads,
                    &solution)) {
            return EmptyResult(env);
        }

        const size_t maximum_output =
                static_cast<size_t>(std::numeric_limits<jsize>::max());
        const size_t output_size = 5 + solution.pair_error_norms.size();
        if (output_size > maximum_output) return EmptyResult(env);
        std::vector<double> output;
        output.reserve(output_size);
        output.push_back(1.0);
        output.push_back(solution.initial_cost);
        output.push_back(solution.final_cost);
        output.push_back(static_cast<double>(solution.iterations));
        output.push_back(solution.focal_length);
        output.insert(output.end(), solution.pair_error_norms.begin(),
                      solution.pair_error_norms.end());

        jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output.size()));
        if (result == nullptr) return nullptr;
        env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output.size()),
                                  output.data());
        if (env->ExceptionCheck()) return nullptr;
        return result;
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, "TrackStudioViewGraphCalibration",
                            "Native solve exception: %s", error.what());
        return EmptyResult(env);
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "TrackStudioViewGraphCalibration",
                            "Unknown native solve exception.");
        return EmptyResult(env);
    }
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_glomap_GlomapViewGraphCalibrator_nativeSolveMulti(
        JNIEnv* env,
        jobject,
        jdoubleArray fundamental_matrices_array,
        jintArray pair_camera_indices_array,
        jdoubleArray principal_points_array,
        jdoubleArray initial_focal_lengths_array,
        jbooleanArray has_prior_focal_length_array,
        jint max_num_iterations,
        jint num_threads) {
    try {
        if (fundamental_matrices_array == nullptr || pair_camera_indices_array == nullptr ||
            principal_points_array == nullptr || initial_focal_lengths_array == nullptr ||
            has_prior_focal_length_array == nullptr) {
            return EmptyResult(env);
        }
        const jsize fundamental_value_count = env->GetArrayLength(fundamental_matrices_array);
        const jsize pair_index_count = env->GetArrayLength(pair_camera_indices_array);
        const jsize principal_point_count = env->GetArrayLength(principal_points_array);
        const jsize camera_count = env->GetArrayLength(initial_focal_lengths_array);
        const jsize prior_count = env->GetArrayLength(has_prior_focal_length_array);
        if (fundamental_value_count <= 0 || fundamental_value_count % 9 != 0 ||
            pair_index_count != (fundamental_value_count / 9) * 2 ||
            camera_count <= 0 || principal_point_count != camera_count * 2 ||
            prior_count != camera_count || max_num_iterations <= 0 || num_threads <= 0) {
            return EmptyResult(env);
        }

        std::vector<double> fundamentals(static_cast<size_t>(fundamental_value_count));
        std::vector<jint> pair_camera_indices_jni(static_cast<size_t>(pair_index_count));
        std::vector<double> principal_points(static_cast<size_t>(principal_point_count));
        std::vector<double> initial_focal_lengths(static_cast<size_t>(camera_count));
        std::vector<jboolean> has_prior_jni(static_cast<size_t>(prior_count));
        env->GetDoubleArrayRegion(fundamental_matrices_array, 0, fundamental_value_count,
                                  fundamentals.data());
        env->GetIntArrayRegion(pair_camera_indices_array, 0, pair_index_count,
                               pair_camera_indices_jni.data());
        env->GetDoubleArrayRegion(principal_points_array, 0, principal_point_count,
                                  principal_points.data());
        env->GetDoubleArrayRegion(initial_focal_lengths_array, 0, camera_count,
                                  initial_focal_lengths.data());
        env->GetBooleanArrayRegion(has_prior_focal_length_array, 0, prior_count,
                                   has_prior_jni.data());
        if (env->ExceptionCheck()) return nullptr;

        std::vector<int> pair_camera_indices(pair_camera_indices_jni.begin(),
                                             pair_camera_indices_jni.end());
        std::vector<std::uint8_t> has_prior_focal_length(static_cast<size_t>(prior_count));
        for (size_t index = 0; index < has_prior_focal_length.size(); ++index) {
            has_prior_focal_length[index] = has_prior_jni[index] == JNI_FALSE ? 0 : 1;
        }

        tracking::MultiCameraViewGraphCalibrationResult solution;
        if (!tracking::RunMultiCameraViewGraphFocalCalibration(
                    fundamentals, pair_camera_indices, principal_points,
                    initial_focal_lengths, has_prior_focal_length,
                    max_num_iterations, num_threads, &solution)) {
            return EmptyResult(env);
        }

        const size_t maximum_output =
                static_cast<size_t>(std::numeric_limits<jsize>::max());
        const size_t output_size = 6 + solution.focal_lengths.size() +
                                   solution.pair_error_norms.size();
        if (output_size > maximum_output) return EmptyResult(env);
        std::vector<double> output;
        output.reserve(output_size);
        output.push_back(1.0);
        output.push_back(solution.initial_cost);
        output.push_back(solution.final_cost);
        output.push_back(static_cast<double>(solution.iterations));
        output.push_back(static_cast<double>(solution.focal_lengths.size()));
        output.push_back(static_cast<double>(solution.pair_error_norms.size()));
        output.insert(output.end(), solution.focal_lengths.begin(),
                      solution.focal_lengths.end());
        output.insert(output.end(), solution.pair_error_norms.begin(),
                      solution.pair_error_norms.end());

        jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output.size()));
        if (result == nullptr) return nullptr;
        env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output.size()),
                                  output.data());
        if (env->ExceptionCheck()) return nullptr;
        return result;
    } catch (const std::exception& error) {
        __android_log_print(ANDROID_LOG_ERROR, "TrackStudioViewGraphCalibration",
                            "Native multi-camera solve exception: %s", error.what());
        return EmptyResult(env);
    } catch (...) {
        __android_log_print(ANDROID_LOG_ERROR, "TrackStudioViewGraphCalibration",
                            "Unknown native multi-camera solve exception.");
        return EmptyResult(env);
    }
}
