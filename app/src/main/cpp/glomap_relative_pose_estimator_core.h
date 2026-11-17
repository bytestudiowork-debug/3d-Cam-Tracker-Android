// Android adapter for GLOMAP's EstimateRelativePoses at
// 0edb1b8435e0f9a594318908b81a31f078a51bf7, using its pinned PoseLib API.
#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <vector>

namespace tracking {

struct GlomapRelativePoseCamera {
    double fx;
    double fy;
    double skew;
    double cx;
    double cy;
    int width;
    int height;
};

struct GlomapRelativePoseOptions {
    // GLOMAP sets max_iterations to 50000. Android may use a lower runtime cap;
    // native tests verify the default and that bound.
    size_t max_iterations = 50000;
};

struct GlomapRelativePoseResult {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    std::vector<int> inlier_indices;
    size_t iterations = 0;
    double inlier_ratio = 0.0;
};

// Estimates camera2_from_camera1 from matched raw pixel coordinates. The
// shared PINHOLE input is converted to PoseLib's zero-skew PINHOLE convention.
bool EstimateGlomapRelativePose(const double* first_pixels_xy,
                                const double* second_pixels_xy,
                                size_t num_matches,
                                const GlomapRelativePoseCamera& camera1,
                                const GlomapRelativePoseCamera& camera2,
                                const GlomapRelativePoseOptions& options,
                                GlomapRelativePoseResult* result);

}  // namespace tracking
