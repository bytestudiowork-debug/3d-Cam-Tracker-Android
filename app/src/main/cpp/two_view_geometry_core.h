// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "homography_estimator_core.h"
#include "two_view_epipolar_estimator_core.h"

#include <cstddef>
#include <vector>

namespace tracking::source_two_view {

enum class GeometryConfiguration {
    kDegenerate,
    kCalibrated,
    kUncalibrated,
    kPlanarOrPanoramic,
};

enum class InlierModel {
    kNone,
    kEssential,
    kFundamental,
    kHomography,
};

enum class PoseSource {
    kNone,
    kEssential,
    kFundamental,
    kHomography,
};

struct ModelSupport {
    bool success = false;
    std::size_t num_inliers = 0;
    std::vector<char> inlier_mask;
};

struct TwoViewDispatchOptions {
    std::size_t minimum_inliers = 15;
    double minimum_essential_to_fundamental_ratio = 0.95;
    double maximum_homography_ratio = 0.8;
};

struct TwoViewGeometryDecision {
    GeometryConfiguration configuration = GeometryConfiguration::kDegenerate;
    InlierModel selected_model = InlierModel::kNone;
    PoseSource pose_source = PoseSource::kNone;
    std::size_t num_inliers = 0;
    std::vector<char> selected_inlier_mask;
};

struct InitialPairPoseResult {
    bool success = false;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    std::size_t selected_candidate = 0;
    std::size_t positive_depth_count = 0;
    std::vector<char> selected_inlier_mask;
    std::vector<char> cheirality_mask;
    double median_triangulation_angle_radians = 0.0;
    bool passes_initial_pair_gates = false;
};

struct InitialPairGeometryReport {
    EpipolarGeometryReport epipolar;
    LORansacReport<Eigen::Matrix3d> homography;
    TwoViewGeometryDecision decision;
};

TwoViewGeometryDecision DispatchTwoViewGeometry(
        bool both_cameras_have_focal_priors,
        const ModelSupport& essential,
        const ModelSupport& fundamental,
        const ModelSupport& homography,
        const TwoViewDispatchOptions& options = {});

ModelSupport ToModelSupport(
        const LORansacReport<Eigen::Matrix3d>& report);

bool EstimateInitialPairGeometry(
        const std::vector<Eigen::Vector2d>& pixels1,
        const std::vector<Eigen::Vector2d>& pixels2,
        const Eigen::Matrix3d& camera_matrix,
        bool both_cameras_have_focal_priors,
        double max_error_pixels,
        InitialPairGeometryReport* report,
        const RansacOptions& epipolar_options = RansacOptions{},
        const RansacOptions& homography_options = RansacOptions{},
        const TwoViewDispatchOptions& dispatch_options =
                TwoViewDispatchOptions{});

bool RecoverInitialPairPose(
        const InitialPairGeometryReport& geometry,
        const std::vector<Eigen::Vector2d>& pixels1,
        const std::vector<Eigen::Vector2d>& pixels2,
        const Eigen::Matrix3d& camera_matrix,
        const source_homography::InitialPairGates& homography_gates,
        InitialPairPoseResult* result);

}  // namespace tracking::source_two_view
