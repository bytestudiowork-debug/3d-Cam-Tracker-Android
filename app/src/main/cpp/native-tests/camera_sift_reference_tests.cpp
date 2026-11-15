/*
Copyright (c), ETH Zurich and UNC Chapel Hill.
All rights reserved.
*/

#include "sift_feature_core.h"

#include "../sift_core/sift.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 256;
constexpr int kHeight = 256;
constexpr int kMaximumFeatures = 400;
constexpr int kDescriptorLength = 128;
constexpr std::size_t kExpectedFeatureCount = 275;
constexpr std::uint64_t kExpectedFeatureHash = 0x709ccd0d8eeda160ULL;
constexpr double kPeakThreshold = 0.02 / 3.0;
constexpr double kEdgeThreshold = 10.0;
constexpr int kBinPermutation[8] = {0, 7, 6, 5, 4, 3, 2, 1};

struct Group {
    int detected_keypoint_count = 0;
    std::vector<tracking::SiftFeature> features;
};

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

std::vector<std::uint8_t> make_fixture() {
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kWidth) * kHeight);
    std::uint32_t state = 0x243f6a88U;
    for (std::uint8_t& pixel : pixels) {
        state = state * 1664525U + 1013904223U;
        pixel = static_cast<std::uint8_t>(state >> 24);
    }
    return pixels;
}

std::array<std::uint8_t, kDescriptorLength> reference_descriptor(
    const vl_sift_pix* raw) {
    float l1_norm = 0.0f;
    for (int index = 0; index < kDescriptorLength; ++index) {
        l1_norm += std::fabs(raw[index]);
    }

    std::array<std::uint8_t, kDescriptorLength> vl_order{};
    if (l1_norm > 0.0f && std::isfinite(l1_norm)) {
        const float inverse_norm = 1.0f / l1_norm;
        for (int index = 0; index < kDescriptorLength; ++index) {
            const float rooted = std::sqrt(raw[index] * inverse_norm);
            const float rounded = std::round(512.0f * rooted);
            vl_order[index] = static_cast<std::uint8_t>(
                std::min(255.0f, std::max(0.0f, rounded)));
        }
    }

    std::array<std::uint8_t, kDescriptorLength> output{};
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            const int cell = 8 * (column + 4 * row);
            for (int orientation = 0; orientation < 8; ++orientation) {
                output[cell + kBinPermutation[orientation]] =
                    vl_order[cell + orientation];
            }
        }
    }
    return output;
}

std::vector<tracking::SiftFeature> direct_reference_extract(
    const std::vector<std::uint8_t>& pixels,
    int width,
    int height,
    int maximum_features) {
    std::unique_ptr<VlSiftFilt, decltype(&vl_sift_delete)> filter(
        vl_sift_new(width, height, 4, 3, -1), &vl_sift_delete);
    if (!filter) return {};
    vl_sift_set_peak_thresh(filter.get(), kPeakThreshold);
    vl_sift_set_edge_thresh(filter.get(), kEdgeThreshold);

    std::vector<float> float_image(pixels.size());
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        float_image[index] = static_cast<float>(pixels[index]) / 255.0f;
    }

    std::vector<Group> groups;
    bool first_octave = true;
    while (true) {
        if (first_octave) {
            if (vl_sift_process_first_octave(filter.get(), float_image.data()) != 0) {
                break;
            }
            first_octave = false;
        } else if (vl_sift_process_next_octave(filter.get()) != 0) {
            break;
        }

        vl_sift_detect(filter.get());
        const VlSiftKeypoint* keypoints = vl_sift_get_keypoints(filter.get());
        const int keypoint_count = vl_sift_get_nkeypoints(filter.get());
        if (keypoint_count == 0) continue;

        int previous_level = -1;
        for (int index = 0; index < keypoint_count; ++index) {
            const VlSiftKeypoint& keypoint = keypoints[index];
            if (keypoint.is != previous_level) {
                groups.emplace_back();
                previous_level = keypoint.is;
            }
            Group& group = groups.back();
            ++group.detected_keypoint_count;

            double angles[4];
            const int angle_count = vl_sift_calc_keypoint_orientations(
                filter.get(), angles, &keypoint);
            const int used_angles = std::min(angle_count, 2);
            for (int angle_index = 0; angle_index < used_angles; ++angle_index) {
                vl_sift_pix raw[kDescriptorLength];
                vl_sift_calc_keypoint_descriptor(
                    filter.get(), raw, &keypoint, angles[angle_index]);
                tracking::SiftFeature feature{};
                feature.x = keypoint.x + 0.5f;
                feature.y = keypoint.y + 0.5f;
                feature.scale = keypoint.sigma;
                feature.orientation_radians = static_cast<float>(angles[angle_index]);
                feature.descriptor = reference_descriptor(raw);
                group.features.push_back(feature);
            }
        }
    }

    std::size_t first_group_to_keep = 0;
    int detected_count = 0;
    for (std::size_t reverse_index = groups.size(); reverse_index > 0;
         --reverse_index) {
        const std::size_t index = reverse_index - 1;
        detected_count += groups[index].detected_keypoint_count;
        if (detected_count > maximum_features) {
            first_group_to_keep = index;
            break;
        }
    }

    std::vector<tracking::SiftFeature> output;
    for (std::size_t index = first_group_to_keep; index < groups.size(); ++index) {
        output.insert(output.end(),
                      groups[index].features.begin(),
                      groups[index].features.end());
    }
    return output;
}

std::uint64_t golden_hash(const std::vector<tracking::SiftFeature>& features) {
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&hash](std::uint8_t byte) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    };
    for (const tracking::SiftFeature& feature : features) {
        const float values[4] = {
            feature.x, feature.y, feature.scale, feature.orientation_radians};
        for (float value : values) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (int shift = 0; shift < 32; shift += 8) {
                mix(static_cast<std::uint8_t>(bits >> shift));
            }
        }
        for (std::uint8_t byte : feature.descriptor) mix(byte);
    }
    return hash;
}

bool same_features(const std::vector<tracking::SiftFeature>& actual,
                   const std::vector<tracking::SiftFeature>& expected) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const tracking::SiftFeature& left = actual[index];
        const tracking::SiftFeature& right = expected[index];
        if (left.x != right.x || left.y != right.y || left.scale != right.scale ||
            left.orientation_radians != right.orientation_radians ||
            left.descriptor != right.descriptor) {
            std::cerr << "reference mismatch at feature " << index << '\n';
            return false;
        }
    }
    return true;
}

}

int main(int argc, char** argv) {
    const std::vector<std::uint8_t> pixels = make_fixture();
    tracking::CpuSiftFeatureExtractor extractor(kMaximumFeatures);
    const std::vector<tracking::SiftFeature> actual =
        extractor.extract(pixels.data(), kWidth, kHeight);
    const std::vector<tracking::SiftFeature> expected = direct_reference_extract(
        pixels, kWidth, kHeight, kMaximumFeatures);
    const std::vector<tracking::SiftFeature> repeated =
        extractor.extract(pixels.data(), kWidth, kHeight);

    if (!check(!actual.empty(), "the fixed fixture produced no features") ||
        !check(same_features(actual, expected),
               "the native extractor differs from the direct reference") ||
        !check(same_features(repeated, actual),
               "repeated extraction was not deterministic")) {
        return 1;
    }

    for (const tracking::SiftFeature& feature : actual) {
        if (!check(std::isfinite(feature.x) && std::isfinite(feature.y) &&
                       std::isfinite(feature.scale) && feature.scale > 0.0f &&
                       std::isfinite(feature.orientation_radians),
                   "the fixed fixture contains an invalid keypoint")) {
            return 1;
        }
    }

    const std::uint64_t hash = golden_hash(expected);
    if (!check(expected.size() == kExpectedFeatureCount,
               "the pinned reference feature count changed") ||
        !check(hash == kExpectedFeatureHash,
               "the pinned reference descriptor/keypoint golden changed") ||
        !check(golden_hash(actual) == kExpectedFeatureHash,
               "the production adapter differs from the pinned golden")) {
        return 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--dump") {
        std::cout << "FEATURE_GOLDEN count=" << actual.size()
                  << " hash=0x" << std::hex << hash << std::dec << '\n';
        const std::size_t samples = std::min<std::size_t>(actual.size(), 8);
        for (std::size_t index = 0; index < samples; ++index) {
            const tracking::SiftFeature& feature = actual[index];
            std::cout << "FEATURE_SAMPLE " << index << ' '
                      << std::setprecision(9) << feature.x << ' ' << feature.y
                      << ' ' << feature.scale << ' '
                      << feature.orientation_radians << ' '
                      << std::hex << static_cast<int>(feature.descriptor[0])
                      << std::dec << '\n';
        }
    }
    return 0;
}
