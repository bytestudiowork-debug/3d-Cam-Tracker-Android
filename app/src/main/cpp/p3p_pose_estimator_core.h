// Copyright (c) 2020, Viktor Larsson. All rights reserved.
// Adapted from PoseLib PoseLib/solvers/p3p.h at commit
// f119951fca625133112acde48daffa5f20eba451, pinned by COLMAP commit
// c5f9cefc87e5dd596b638e4cee0ff543c7d14755. See the PoseLib license asset.
#pragma once

#include <Eigen/Core>

#include <vector>

namespace tracking {

struct P3PPoseEstimate {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

// Returns every real positive-depth pose produced by the pinned PoseLib P3P
// solver for exactly three normalized camera rays and corresponding world points.
bool EstimateP3PPoseCandidates(const std::vector<Eigen::Vector3d>& camera_rays,
                                const std::vector<Eigen::Vector3d>& world_points,
                                std::vector<P3PPoseEstimate>* candidates);

}  // namespace tracking
