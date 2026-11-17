#include "glomap_relative_pose_estimator_core.h"

#include <jni.h>

#include <cstddef>
#include <vector>

namespace {

bool ReadDoubleArray(JNIEnv* env, jdoubleArray source, std::vector<double>* values) {
    if (source == nullptr || values == nullptr) return false;
    const jsize length = env->GetArrayLength(source);
    if (length <= 0) return false;
    values->resize(static_cast<size_t>(length));
    env->GetDoubleArrayRegion(source, 0, length, values->data());
    return !env->ExceptionCheck();
}

bool ReadIntArray(JNIEnv* env, jintArray source, std::vector<jint>* values) {
    if (source == nullptr || values == nullptr) return false;
    const jsize length = env->GetArrayLength(source);
    if (length <= 0) return false;
    values->resize(static_cast<size_t>(length));
    env->GetIntArrayRegion(source, 0, length, values->data());
    return !env->ExceptionCheck();
}

}  // namespace

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_glomap_GlomapRelativePoseEstimator_nativeEstimate(
        JNIEnv* env,
        jobject /* thiz */,
        jdoubleArray first_pixels_xy,
        jdoubleArray second_pixels_xy,
        jdoubleArray camera1_values,
        jintArray camera1_dimensions,
        jdoubleArray camera2_values,
        jintArray camera2_dimensions,
        jint max_iterations) {
    std::vector<double> first_pixels;
    std::vector<double> second_pixels;
    std::vector<double> camera1_data;
    std::vector<double> camera2_data;
    std::vector<jint> camera1_size;
    std::vector<jint> camera2_size;
    if (!ReadDoubleArray(env, first_pixels_xy, &first_pixels) ||
        !ReadDoubleArray(env, second_pixels_xy, &second_pixels) ||
        !ReadDoubleArray(env, camera1_values, &camera1_data) ||
        !ReadDoubleArray(env, camera2_values, &camera2_data) ||
        !ReadIntArray(env, camera1_dimensions, &camera1_size) ||
        !ReadIntArray(env, camera2_dimensions, &camera2_size) ||
        first_pixels.size() != second_pixels.size() || first_pixels.size() % 2 != 0 ||
        camera1_data.size() != 5 || camera2_data.size() != 5 ||
        camera1_size.size() != 2 || camera2_size.size() != 2 || max_iterations <= 0) {
        return nullptr;
    }

    const tracking::GlomapRelativePoseCamera camera1{
            camera1_data[0], camera1_data[1], camera1_data[2],
            camera1_data[3], camera1_data[4], camera1_size[0], camera1_size[1]};
    const tracking::GlomapRelativePoseCamera camera2{
            camera2_data[0], camera2_data[1], camera2_data[2],
            camera2_data[3], camera2_data[4], camera2_size[0], camera2_size[1]};
    tracking::GlomapRelativePoseOptions options;
    options.max_iterations = static_cast<size_t>(max_iterations);
    tracking::GlomapRelativePoseResult estimate;
    if (!tracking::EstimateGlomapRelativePose(
                first_pixels.data(), second_pixels.data(), first_pixels.size() / 2,
                camera1, camera2, options, &estimate)) {
        return nullptr;
    }

    constexpr size_t kResultHeaderSize = 15;
    std::vector<jdouble> packed(kResultHeaderSize + first_pixels.size() / 2, 0.0);
    packed[0] = 1.0;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            packed[1 + row * 3 + column] = estimate.rotation(row, column);
        }
    }
    packed[10] = estimate.translation.x();
    packed[11] = estimate.translation.y();
    packed[12] = estimate.translation.z();
    packed[13] = static_cast<jdouble>(estimate.inlier_indices.size());
    packed[14] = static_cast<jdouble>(estimate.iterations);
    for (int index : estimate.inlier_indices) {
        if (index >= 0 && static_cast<size_t>(index) < first_pixels.size() / 2) {
            packed[kResultHeaderSize + static_cast<size_t>(index)] = 1.0;
        }
    }

    jdoubleArray output = env->NewDoubleArray(static_cast<jsize>(packed.size()));
    if (output == nullptr) return nullptr;
    env->SetDoubleArrayRegion(output, 0, static_cast<jsize>(packed.size()), packed.data());
    return env->ExceptionCheck() ? nullptr : output;
}
