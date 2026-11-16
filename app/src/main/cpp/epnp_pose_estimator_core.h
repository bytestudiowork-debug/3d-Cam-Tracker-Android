// Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
// Adapted from COLMAP src/colmap/estimators/absolute_pose.h at commit
// c5f9cefc87e5dd596b638e4cee0ff543c7d14755. See the matching license asset.
#pragma once

#include <Eigen/Core>

#include <vector>

namespace tracking {

struct EpnpCameraIntrinsics {
    double fx = 0.0;
    double skew = 0.0;
    double cx = 0.0;
    double fy = 0.0;
    double cy = 0.0;
};

struct EpnpPoseEstimate {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double reprojection_error = 0.0;
};

// Estimate one world-to-camera pose from at least four 2D-3D correspondences.
// The EPnP control-point, beta-initialization, and pose-recovery math follows
// the pinned COLMAP implementation; pixels use the supplied pinhole intrinsics.
bool EstimateEpnpPose(const std::vector<Eigen::Vector2d>& image_points,
                      const std::vector<Eigen::Vector3d>& world_points,
                      const EpnpCameraIntrinsics& intrinsics,
                      EpnpPoseEstimate* estimate);

}  // namespace tracking
