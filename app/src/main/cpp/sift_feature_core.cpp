/*
Copyright (c), ETH Zurich and UNC Chapel Hill.
All rights reserved.
*/

#include "sift_feature_core.h"

#include "sift_core/sift.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>
#include <utility>

namespace tracking {
namespace {

constexpr int kDescriptorLength = 128;
constexpr int kOctaveCount = 4;
constexpr int kOctaveResolution = 3;
constexpr int kFirstOctave = -1;
constexpr int kMaximumOrientations = 2;
constexpr float kDescriptorScale = 512.0f;
constexpr double kPeakThreshold = 0.02 / kOctaveResolution;
constexpr double kEdgeThreshold = 10.0;
constexpr std::array<int, 8> kOrientationBinPermutation{{0, 7, 6, 5, 4, 3, 2, 1}};
constexpr std::size_t kMaximumImagePixels = 32u * 1024u * 1024u;

struct FeatureGroup {
    int detected_keypoint_count = 0;
    std::vector<SiftFeature> oriented_features;
};

std::array<std::uint8_t, kDescriptorLength> to_output_descriptor(
    const vl_sift_pix* raw_descriptor) {
    float l1_norm = 0.0f;
    for (int index = 0; index < kDescriptorLength; ++index) {
        l1_norm += std::fabs(raw_descriptor[index]);
    }

    std::array<std::uint8_t, kDescriptorLength> vl_order{};
    if (l1_norm > 0.0f && std::isfinite(l1_norm)) {
        const float inverse_norm = 1.0f / l1_norm;
        for (int index = 0; index < kDescriptorLength; ++index) {
            const float rooted = std::sqrt(raw_descriptor[index] * inverse_norm);
            const float rounded = std::round(kDescriptorScale * rooted);
            const float truncated = std::min(255.0f, std::max(0.0f, rounded));
            vl_order[index] = static_cast<std::uint8_t>(truncated);
        }
    }

    std::array<std::uint8_t, kDescriptorLength> output{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const int cell = 8 * (column + 4 * row);
            for (int orientation = 0; orientation < 8; ++orientation) {
                output[cell + kOrientationBinPermutation[orientation]] =
                    vl_order[cell + orientation];
            }
        }
    }
    return output;
}

}

CpuSiftFeatureExtractor::CpuSiftFeatureExtractor(int maximum_features)
    : maximum_features_(maximum_features) {
    if (maximum_features_ <= 0) {
        throw std::invalid_argument("maximum_features must be positive");
    }
}

CpuSiftFeatureExtractor::~CpuSiftFeatureExtractor() {
    clear();
}

void CpuSiftFeatureExtractor::ensure_filter(int width, int height) {
    if (filter_ != nullptr && width_ == width && height_ == height) {
        return;
    }
    clear();
    filter_ = vl_sift_new(
        width, height, kOctaveCount, kOctaveResolution, kFirstOctave);
    if (filter_ == nullptr) {
        throw std::runtime_error("native feature filter allocation failed");
    }
    width_ = width;
    height_ = height;
}

std::vector<SiftFeature> CpuSiftFeatureExtractor::extract(
    const std::uint8_t* grayscale_pixels,
    int width,
    int height) {
    if (grayscale_pixels == nullptr || width <= 0 || height <= 0) {
        throw std::invalid_argument("invalid grayscale image");
    }
    const std::size_t image_pixels = static_cast<std::size_t>(width) *
                                     static_cast<std::size_t>(height);
    if (image_pixels > kMaximumImagePixels || image_pixels > INT_MAX) {
        throw std::invalid_argument("grayscale image exceeds the native input limit");
    }

    ensure_filter(width, height);
    vl_sift_set_peak_thresh(filter_, kPeakThreshold);
    vl_sift_set_edge_thresh(filter_, kEdgeThreshold);

    std::vector<float> image_float(image_pixels);
    for (std::size_t index = 0; index < image_pixels; ++index) {
        image_float[index] = static_cast<float>(grayscale_pixels[index]) / 255.0f;
    }

    std::vector<FeatureGroup> groups;
    bool first_octave = true;
    while (true) {
        if (first_octave) {
            if (vl_sift_process_first_octave(filter_, image_float.data()) != 0) {
                break;
            }
            first_octave = false;
        } else if (vl_sift_process_next_octave(filter_) != 0) {
            break;
        }

        vl_sift_detect(filter_);
        const VlSiftKeypoint* keypoints = vl_sift_get_keypoints(filter_);
        const int keypoint_count = vl_sift_get_nkeypoints(filter_);
        if (keypoint_count == 0) {
            continue;
        }

        int previous_level = -1;
        for (int index = 0; index < keypoint_count; ++index) {
            const VlSiftKeypoint& keypoint = keypoints[index];
            if (keypoint.is != previous_level) {
                groups.emplace_back();
                previous_level = keypoint.is;
            }
            FeatureGroup& group = groups.back();
            ++group.detected_keypoint_count;

            double orientations[4];
            const int orientation_count = vl_sift_calc_keypoint_orientations(
                filter_, orientations, &keypoint);
            const int used_orientations =
                std::min(orientation_count, kMaximumOrientations);

            for (int orientation_index = 0;
                 orientation_index < used_orientations;
                 ++orientation_index) {
                vl_sift_pix raw_descriptor[kDescriptorLength];
                vl_sift_calc_keypoint_descriptor(
                    filter_, raw_descriptor, &keypoint,
                    orientations[orientation_index]);

                SiftFeature feature{};
                feature.x = keypoint.x + 0.5f;
                feature.y = keypoint.y + 0.5f;
                feature.scale = keypoint.sigma;
                feature.orientation_radians =
                    static_cast<float>(orientations[orientation_index]);
                feature.descriptor = to_output_descriptor(raw_descriptor);
                group.oriented_features.push_back(feature);
            }
        }
    }

    std::size_t first_group_to_keep = 0;
    int retained_keypoint_count = 0;
    std::size_t retained_oriented_count = 0;
    for (std::size_t reverse_index = groups.size(); reverse_index > 0;
         --reverse_index) {
        const std::size_t index = reverse_index - 1;
        retained_keypoint_count += groups[index].detected_keypoint_count;
        retained_oriented_count += groups[index].oriented_features.size();
        if (retained_keypoint_count > maximum_features_) {
            first_group_to_keep = index;
            break;
        }
    }

    std::vector<SiftFeature> output;
    output.reserve(retained_oriented_count);
    for (std::size_t index = first_group_to_keep; index < groups.size(); ++index) {
        output.insert(output.end(),
                      groups[index].oriented_features.begin(),
                      groups[index].oriented_features.end());
    }
    return output;
}

void CpuSiftFeatureExtractor::clear() noexcept {
    if (filter_ != nullptr) {
        vl_sift_delete(filter_);
        filter_ = nullptr;
    }
    width_ = 0;
    height_ = 0;
}

}
