// Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
// Adapted from COLMAP src/colmap/estimators/pose.h and
// src/colmap/optim/{ransac.h,loransac.h} at commit
// c5f9cefc87e5dd596b638e4cee0ff543c7d14755. The minimal P3P solver is adapted
// from the separately licensed PoseLib revision listed in its own notice.
#pragma once

#include "epnp_pose_estimator_core.h"
#include "p3p_pose_estimator_core.h"

#include <cstddef>
#include <vector>

namespace tracking {

struct AbsolutePoseLORansacOptions {
    double max_error_px = 12.0;
    double min_inlier_ratio = 0.1;
    double confidence = 0.99999;
    double dynamic_trial_multiplier = 3.0;
    int min_num_trials = 100;
    int max_num_trials = 10000;
    int random_seed = -1;
    int max_local_trials = 10;
};

struct AbsolutePoseLORansacResult {
    bool success = false;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    std::vector<int> inlier_indices;
    int num_trials = 0;
    int num_local_trials = 0;
    double inlier_residual_sum = 0.0;
};

// COLMAP-style P3P hypotheses + EPnP local estimation for fixed pinhole
// intrinsics. Android retains its own registration gates after this estimator.
bool EstimateAbsolutePoseLORansac(
        const std::vector<Eigen::Vector2d>& image_points,
        const std::vector<Eigen::Vector3d>& world_points,
        const EpnpCameraIntrinsics& intrinsics,
        const AbsolutePoseLORansacOptions& options,
        AbsolutePoseLORansacResult* result);

// Source-equivalent finite-population adaptive trial estimate used by RANSAC.
size_t ComputeAbsolutePoseRansacNumTrials(size_t num_inliers,
                                          size_t num_samples,
                                          double confidence,
                                          double multiplier);

}  // namespace tracking
