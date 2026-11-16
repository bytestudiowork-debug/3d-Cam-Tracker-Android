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
