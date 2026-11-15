/*
Copyright (c), ETH Zurich and UNC Chapel Hill.
All rights reserved.
*/

#include "sift_feature_core.h"

#include <jni.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace {

constexpr jint kDescriptorLength = 128;
constexpr jlong kMaximumImagePixels = 32LL * 1024LL * 1024LL;
std::mutex g_extractor_mutex;
std::unique_ptr<tracking::CpuSiftFeatureExtractor> g_extractor;
int g_maximum_features = 0;

jobject make_output(JNIEnv* env,
                    const std::vector<tracking::SiftFeature>& features) {
    if (features.size() > static_cast<std::size_t>(
                              std::numeric_limits<jsize>::max() / kDescriptorLength) ||
        features.size() > static_cast<std::size_t>(
                              std::numeric_limits<jsize>::max() / 4)) {
        return nullptr;
    }

    const jsize count = static_cast<jsize>(features.size());
    jfloatArray keypoint_array = env->NewFloatArray(count * 4);
    if (keypoint_array == nullptr) {
        return nullptr;
    }
    jbyteArray descriptor_array = env->NewByteArray(count * kDescriptorLength);
    if (descriptor_array == nullptr) {
        env->DeleteLocalRef(keypoint_array);
        return nullptr;
    }

    std::vector<jfloat> keypoint_data(static_cast<std::size_t>(count) * 4);
    std::vector<jbyte> descriptor_data(
        static_cast<std::size_t>(count) * kDescriptorLength);
    for (jsize index = 0; index < count; ++index) {
        const tracking::SiftFeature& feature = features[static_cast<std::size_t>(index)];
        const std::size_t point_offset = static_cast<std::size_t>(index) * 4;
        keypoint_data[point_offset] = feature.x;
        keypoint_data[point_offset + 1] = feature.y;
        keypoint_data[point_offset + 2] = feature.scale;
        keypoint_data[point_offset + 3] = feature.orientation_radians;
        std::memcpy(descriptor_data.data() +
                        static_cast<std::size_t>(index) * kDescriptorLength,
                    feature.descriptor.data(), kDescriptorLength);
    }

    if (count > 0) {
        env->SetFloatArrayRegion(keypoint_array, 0, count * 4, keypoint_data.data());
        env->SetByteArrayRegion(
            descriptor_array, 0, count * kDescriptorLength, descriptor_data.data());
    }
    if (env->ExceptionCheck()) {
        env->DeleteLocalRef(keypoint_array);
        env->DeleteLocalRef(descriptor_array);
        return nullptr;
    }

    jclass output_class =
        env->FindClass("com/trackstudio/app/tracking/NativeSiftOutput");
    if (output_class == nullptr) {
        env->DeleteLocalRef(keypoint_array);
        env->DeleteLocalRef(descriptor_array);
        return nullptr;
    }
    jmethodID constructor = env->GetMethodID(output_class, "<init>", "([F[B)V");
    if (constructor == nullptr) {
        env->DeleteLocalRef(output_class);
        env->DeleteLocalRef(keypoint_array);
        env->DeleteLocalRef(descriptor_array);
        return nullptr;
    }
    jobject output = env->NewObject(
        output_class, constructor, keypoint_array, descriptor_array);
    env->DeleteLocalRef(output_class);
    env->DeleteLocalRef(keypoint_array);
    env->DeleteLocalRef(descriptor_array);
    return output;
}

}

extern "C" JNIEXPORT jobject JNICALL
Java_com_trackstudio_app_tracking_NativeSiftExtractor_nativeExtract(
    JNIEnv* env,
    jobject,
    jbyteArray grayscale_image,
    jint width,
    jint height,
    jint maximum_features) {
    if (grayscale_image == nullptr || width <= 0 || height <= 0 ||
        maximum_features <= 0) {
        return nullptr;
    }
    const jlong image_pixels = static_cast<jlong>(width) * height;
    if (image_pixels <= 0 || image_pixels > kMaximumImagePixels ||
        image_pixels != env->GetArrayLength(grayscale_image)) {
        return nullptr;
    }

    try {
        std::lock_guard<std::mutex> lock(g_extractor_mutex);
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(image_pixels));
        env->GetByteArrayRegion(
            grayscale_image, 0, static_cast<jsize>(image_pixels),
            reinterpret_cast<jbyte*>(pixels.data()));
        if (env->ExceptionCheck()) {
            return nullptr;
        }

        if (g_extractor == nullptr || g_maximum_features != maximum_features) {
            g_extractor = std::make_unique<tracking::CpuSiftFeatureExtractor>(
                maximum_features);
            g_maximum_features = maximum_features;
        }
        const std::vector<tracking::SiftFeature> features =
            g_extractor->extract(pixels.data(), width, height);
        return make_output(env, features);
    } catch (...) {
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_trackstudio_app_tracking_NativeSiftExtractor_nativeClear(
    JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_extractor_mutex);
    g_extractor.reset();
    g_maximum_features = 0;
}
