#include <jni.h>

#include "least_absolute_deviations_core.h"

#include <Eigen/SparseCore>

#include <cmath>
#include <limits>
#include <vector>

static jdoubleArray NativeSolveImpl(
        JNIEnv* env,
        jint frame_count,
        jintArray first_frames_array,
        jintArray second_frames_array,
        jdoubleArray residuals_array,
        jdoubleArray anchor_residual_array,
        jint max_iterations) {
    if (frame_count < 2 || first_frames_array == nullptr ||
        second_frames_array == nullptr || residuals_array == nullptr ||
        anchor_residual_array == nullptr || max_iterations <= 0) {
        return nullptr;
    }

    const jsize edge_count = env->GetArrayLength(first_frames_array);
    if (edge_count <= 0 || env->GetArrayLength(second_frames_array) != edge_count ||
        static_cast<jlong>(edge_count) * 3 !=
                static_cast<jlong>(env->GetArrayLength(residuals_array)) ||
        env->GetArrayLength(anchor_residual_array) != 3) {
        return nullptr;
    }
    const jlong edge_row_count_long = static_cast<jlong>(edge_count) * 3;
    const jlong row_count_long = (static_cast<jlong>(edge_count) + 1) * 3;
    const jlong column_count_long = static_cast<jlong>(frame_count) * 3;
    const jlong output_count_long = static_cast<jlong>(frame_count) * 3;
    if (row_count_long > std::numeric_limits<int>::max() ||
        row_count_long > std::numeric_limits<jsize>::max() ||
        column_count_long > std::numeric_limits<int>::max() ||
        output_count_long > std::numeric_limits<jsize>::max() ||
        static_cast<size_t>(edge_count) >
                (std::numeric_limits<size_t>::max() - 3) / 6) {
        return nullptr;
    }

    const int edge_row_count = static_cast<int>(edge_row_count_long);
    const int row_count = static_cast<int>(row_count_long);
    const int column_count = static_cast<int>(column_count_long);
    std::vector<jint> first_frames(static_cast<size_t>(edge_count));
    std::vector<jint> second_frames(static_cast<size_t>(edge_count));
    std::vector<double> residual_values(static_cast<size_t>(row_count));
    std::vector<double> anchor_residual(3);
    env->GetIntArrayRegion(first_frames_array, 0, edge_count, first_frames.data());
    env->GetIntArrayRegion(second_frames_array, 0, edge_count, second_frames.data());
    env->GetDoubleArrayRegion(residuals_array, 0, edge_row_count, residual_values.data());
    env->GetDoubleArrayRegion(anchor_residual_array, 0, 3, anchor_residual.data());
    if (env->ExceptionCheck()) return nullptr;
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(anchor_residual[axis])) return nullptr;
        residual_values[static_cast<size_t>(edge_row_count + axis)] = anchor_residual[axis];
    }

    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<size_t>(edge_count) * 6 + 3);
    for (int edge = 0; edge < edge_count; ++edge) {
        const int first = first_frames[edge];
        const int second = second_frames[edge];
        if (first < 0 || first >= frame_count || second < 0 ||
            second >= frame_count || first == second) {
            return nullptr;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const int row = edge * 3 + axis;
            if (!std::isfinite(residual_values[static_cast<size_t>(row)])) return nullptr;
            entries.emplace_back(row, first * 3 + axis, -1.0);
            entries.emplace_back(row, second * 3 + axis, 1.0);
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        entries.emplace_back(edge_row_count + axis, axis, 1.0);
    }

    Eigen::SparseMatrix<double> incidence(row_count, column_count);
    incidence.setFromTriplets(entries.begin(), entries.end());
    Eigen::VectorXd observations(row_count);
    for (int row = 0; row < row_count; ++row) {
        observations[row] = residual_values[static_cast<size_t>(row)];
    }
    Eigen::VectorXd solution = Eigen::VectorXd::Zero(column_count);
    tracking::LeastAbsoluteDeviationsOptions options;
    options.max_num_iterations = max_iterations;
    if (!tracking::SolveLeastAbsoluteDeviations(
                options, incidence, observations, &solution)) {
        return nullptr;
    }

    std::vector<double> output(static_cast<size_t>(output_count_long));
    for (int frame = 0; frame < frame_count; ++frame) {
        for (int axis = 0; axis < 3; ++axis) {
            output[static_cast<size_t>(frame) * 3 + axis] = solution[frame * 3 + axis];
        }
    }
    jdoubleArray result = env->NewDoubleArray(static_cast<jsize>(output.size()));
    if (result == nullptr) return nullptr;
    env->SetDoubleArrayRegion(result, 0, static_cast<jsize>(output.size()), output.data());
    if (env->ExceptionCheck()) return nullptr;
    return result;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_NativeL1Solver_nativeSolve(
        JNIEnv* env,
        jobject,
        jint frame_count,
        jintArray first_frames_array,
        jintArray second_frames_array,
        jdoubleArray residuals_array,
        jdoubleArray anchor_residual_array,
        jint max_iterations) {
    try {
        return NativeSolveImpl(env, frame_count, first_frames_array,
                               second_frames_array, residuals_array,
                               anchor_residual_array, max_iterations);
    } catch (...) {
        return nullptr;
    }
}
