// Copyright (c) 2024, ETH Zurich. All rights reserved.
// Adapted from GLOMAP glomap/estimators/relpose_estimation.cc at
// 0edb1b8435e0f9a594318908b81a31f078a51bf7. PoseLib is pinned separately at
// 7e9f5f53372e43f89655040d4dfc4a00e5ace11c; see its dedicated license asset.
#include "glomap_relative_pose_estimator_core.h"

#include <PoseLib/robust.h>

#include <cmath>
#include <exception>
#include <vector>

namespace tracking {
namespace {

bool IsValidCamera(const GlomapRelativePoseCamera& camera) {
    return std::isfinite(camera.fx) && camera.fx > 0.0 &&
           std::isfinite(camera.fy) && camera.fy > 0.0 &&
           std::isfinite(camera.skew) && std::isfinite(camera.cx) &&
           std::isfinite(camera.cy) && camera.width > 0 && camera.height > 0;
}

bool IsValidPixels(const double* pixels_xy, size_t num_matches) {
    if (pixels_xy == nullptr) return false;
    for (size_t index = 0; index < num_matches * 2; ++index) {
        if (!std::isfinite(pixels_xy[index])) return false;
    }
    return true;
}

void ConvertPixelsToPoseLib(const double* pixels_xy,
                            size_t num_matches,
                            const GlomapRelativePoseCamera& camera,
                            std::vector<poselib::Point2D>* converted) {
    converted->clear();
    if (converted->capacity() < num_matches) converted->reserve(num_matches);
    for (size_t index = 0; index < num_matches; ++index) {
        const double x = pixels_xy[index * 2];
        const double y = pixels_xy[index * 2 + 1];
        const double zero_skew_x = x - camera.skew * ((y - camera.cy) / camera.fy);
        converted->emplace_back(zero_skew_x, y);
    }
}

}  // namespace

bool EstimateGlomapRelativePose(const double* first_pixels_xy,
                                const double* second_pixels_xy,
                                size_t num_matches,
                                const GlomapRelativePoseCamera& camera1,
                                const GlomapRelativePoseCamera& camera2,
                                const GlomapRelativePoseOptions& options,
                                GlomapRelativePoseResult* result) {
    if (result == nullptr) return false;
    *result = GlomapRelativePoseResult{};
    if (num_matches < 5 || options.max_iterations == 0 ||
        !IsValidCamera(camera1) || !IsValidCamera(camera2) ||
        !IsValidPixels(first_pixels_xy, num_matches) ||
        !IsValidPixels(second_pixels_xy, num_matches)) {
        return false;
    }

    // Match GLOMAP's thread-local point/inlier buffers while keeping the
    // single-camera Android path free of retained frame or descriptor data.
    thread_local std::vector<poselib::Point2D> points1;
    thread_local std::vector<poselib::Point2D> points2;
    thread_local std::vector<char> inliers;
    ConvertPixelsToPoseLib(first_pixels_xy, num_matches, camera1, &points1);
    ConvertPixelsToPoseLib(second_pixels_xy, num_matches, camera2, &points2);
    inliers.clear();

    const poselib::Camera pose_camera1(
            "PINHOLE", {camera1.fx, camera1.fy, camera1.cx, camera1.cy},
            camera1.width, camera1.height);
    const poselib::Camera pose_camera2(
            "PINHOLE", {camera2.fx, camera2.fy, camera2.cx, camera2.cy},
            camera2.width, camera2.height);
    poselib::RansacOptions ransac_options;
    // GLOMAP overrides only max_iterations; PoseLib's pinned defaults supply
    // the 1-pixel Sampson threshold, sampling, confidence and stopping policy.
    ransac_options.max_iterations = options.max_iterations;
    const poselib::BundleOptions bundle_options;
    poselib::CameraPose relative_pose;
    poselib::RansacStats stats;
    try {
        stats = poselib::estimate_relative_pose(
                points1, points2, pose_camera1, pose_camera2,
                ransac_options, bundle_options, &relative_pose, &inliers);
    } catch (const std::exception&) {
        return false;
    }

    if (inliers.size() != num_matches || stats.num_inliers < 5) return false;
    const Eigen::Matrix3d rotation = relative_pose.R();
    Eigen::Vector3d translation = relative_pose.t;
    const double translation_norm = translation.norm();
    if (!rotation.allFinite() || !translation.allFinite() ||
        !std::isfinite(translation_norm) || translation_norm <= 1e-12) {
        return false;
    }
    translation /= translation_norm;

    result->rotation = rotation;
    result->translation = translation;
    result->iterations = stats.iterations;
    result->inlier_indices.reserve(stats.num_inliers);
    for (size_t index = 0; index < inliers.size(); ++index) {
        if (inliers[index] != 0) result->inlier_indices.push_back(static_cast<int>(index));
    }
    if (result->inlier_indices.size() < 5) {
        *result = GlomapRelativePoseResult{};
        return false;
    }
    result->inlier_ratio = static_cast<double>(result->inlier_indices.size()) /
                          static_cast<double>(num_matches);
    return true;
}

}  // namespace tracking
