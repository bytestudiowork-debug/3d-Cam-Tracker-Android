#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "sift_core/sift.h"

namespace tracking {

struct SiftFeature {
    float x;
    float y;
    float scale;
    float orientation_radians;
    std::array<std::uint8_t, 128> descriptor;
};

class CpuSiftFeatureExtractor {
public:
    explicit CpuSiftFeatureExtractor(int maximum_features);
    ~CpuSiftFeatureExtractor();

    CpuSiftFeatureExtractor(const CpuSiftFeatureExtractor&) = delete;
    CpuSiftFeatureExtractor& operator=(const CpuSiftFeatureExtractor&) = delete;

    std::vector<SiftFeature> extract(
        const std::uint8_t* grayscale_pixels,
        int width,
        int height);
    void clear() noexcept;

private:
    void ensure_filter(int width, int height);

    int maximum_features_;
    int width_ = 0;
    int height_ = 0;
    VlSiftFilt* filter_ = nullptr;
};

}
