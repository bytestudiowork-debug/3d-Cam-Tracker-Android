// Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
// Adapted from COLMAP src/colmap/estimators/{pose.h,pose.cc,absolute_pose.h,
// absolute_pose.cc}, src/colmap/optim/{ransac.h,support_measurement.h}, and
// PoseLib/PoseLib/solvers/p4pf.h at the revisions documented in the license assets.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstddef>
#include <vector>

namespace tracking {

struct P4pfCameraIntrinsics {
    double fx = 0.0;
    double fy = 0.0;
    double skew = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

struct P4pfAbsolutePoseOptions {
    double max_error_px = 12.0;
    double min_inlier_ratio = 0.1;
    double confidence = 0.99999;
    double dynamic_trial_multiplier = 3.0;
    int min_num_trials = 100;
    int max_num_trials = 10000;
    int random_seed = -1;
    int max_refinement_iterations = 100;
    double refinement_loss_scale = 1.0;
    double refinement_gradient_tolerance = 1.0;
};

struct P4pfAbsolutePoseResult {
    bool success = false;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double initial_focal_length = 0.0;
    double refined_fx = 0.0;
    double refined_fy = 0.0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int num_trials = 0;
    int refinement_iterations = 0;
    std::vector<int> inlier_indices;
};

// Finite-population RANSAC trial count for the pinned 4-point P4PF solver.
size_t ComputeP4pfRansacNumTrials(size_t num_inliers,
                                  size_t num_samples,
                                  double confidence,
                                  double multiplier);

// Estimates pose and focal length from absolute pixel coordinates. This
// COLMAP-style path assumes zero skew and a shared focal estimate. Principal
// point is fixed, matching COLMAP's P4PF branch; only pose and focal lengths
// are refined from the RANSAC inliers.
bool EstimateAbsolutePoseP4PF(const std::vector<Eigen::Vector2d>& image_points,
                              const std::vector<Eigen::Vector3d>& world_points,
                              const P4pfCameraIntrinsics& initial_intrinsics,
                              const P4pfAbsolutePoseOptions& options,
                              P4pfAbsolutePoseResult* result);

}  // namespace tracking
