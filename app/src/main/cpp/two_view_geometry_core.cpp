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
#include "two_view_geometry_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <Eigen/LU>

namespace tracking::source_two_view {
namespace {

void ValidateSupport(const ModelSupport& support, const char* label) {
    if (support.success &&
        (support.inlier_mask.empty() ||
         static_cast<std::size_t>(std::count_if(
                 support.inlier_mask.begin(), support.inlier_mask.end(),
                 [](const char inlier) { return inlier != 0; })) !=
                 support.num_inliers)) {
        throw std::invalid_argument(std::string("invalid ") + label +
                                    " support report");
    }
}

TwoViewGeometryDecision MakeDecision(
        const GeometryConfiguration configuration,
        const InlierModel model,
        const std::size_t num_inliers,
        const ModelSupport& report) {
    TwoViewGeometryDecision decision;
    decision.configuration = configuration;
    decision.selected_model = model;
    decision.num_inliers = num_inliers;
    if (model != InlierModel::kNone) {
        decision.selected_inlier_mask = report.inlier_mask;
    }
    switch (configuration) {
        case GeometryConfiguration::kCalibrated:
            decision.pose_source = PoseSource::kEssential;
            break;
        case GeometryConfiguration::kUncalibrated:
            decision.pose_source = PoseSource::kFundamental;
            break;
        case GeometryConfiguration::kPlanarOrPanoramic:
            decision.pose_source = PoseSource::kHomography;
            break;
        case GeometryConfiguration::kDegenerate:
            decision.pose_source = PoseSource::kNone;
            break;
    }
    return decision;
}

bool IsFiniteVector(const Eigen::Vector2d& point) {
    return std::isfinite(point.x()) && std::isfinite(point.y());
}

bool IsFiniteMatrix(const Eigen::Matrix3d& matrix) {
    return matrix.array().isFinite().all();
}

}  // namespace

ModelSupport ToModelSupport(
        const LORansacReport<Eigen::Matrix3d>& report) {
    return {report.success, report.support.num_inliers, report.inlier_mask};
}

TwoViewGeometryDecision DispatchTwoViewGeometry(
        const bool both_cameras_have_focal_priors,
        const ModelSupport& essential,
        const ModelSupport& fundamental,
        const ModelSupport& homography,
        const TwoViewDispatchOptions& options) {
    if (options.minimum_inliers == 0 ||
        !std::isfinite(options.minimum_essential_to_fundamental_ratio) ||
        options.minimum_essential_to_fundamental_ratio < 0.0 ||
        options.minimum_essential_to_fundamental_ratio > 1.0 ||
        !std::isfinite(options.maximum_homography_ratio) ||
        options.maximum_homography_ratio < 0.0 ||
        options.maximum_homography_ratio > 1.0) {
        throw std::invalid_argument("invalid two-view dispatch options");
    }
    ValidateSupport(essential, "essential");
    ValidateSupport(fundamental, "fundamental");
    ValidateSupport(homography, "homography");
    const bool any_model_succeeded = essential.success || fundamental.success ||
                                     homography.success;
    const bool any_model_has_support =
            essential.num_inliers >= options.minimum_inliers ||
            fundamental.num_inliers >= options.minimum_inliers ||
            homography.num_inliers >= options.minimum_inliers;
    if (!any_model_succeeded || !any_model_has_support) {
        return MakeDecision(GeometryConfiguration::kDegenerate,
                            InlierModel::kNone, 0, essential);
    }
    if (both_cameras_have_focal_priors) {
        const double essential_to_fundamental_ratio =
                static_cast<double>(essential.num_inliers) /
                static_cast<double>(fundamental.num_inliers);
        const double homography_to_essential_ratio =
                static_cast<double>(homography.num_inliers) /
                static_cast<double>(essential.num_inliers);
        if (essential.success &&
            essential_to_fundamental_ratio >
                    options.minimum_essential_to_fundamental_ratio &&
            essential.num_inliers >= options.minimum_inliers) {
            const bool essential_is_best =
                    essential.num_inliers >= fundamental.num_inliers;
            const ModelSupport& best_epipolar =
                    essential_is_best ? essential : fundamental;
            const InlierModel best_model = essential_is_best
                    ? InlierModel::kEssential
                    : InlierModel::kFundamental;
            if (homography_to_essential_ratio >
                options.maximum_homography_ratio) {
                if (homography.num_inliers > best_epipolar.num_inliers) {
                    return MakeDecision(
                            GeometryConfiguration::kPlanarOrPanoramic,
                            InlierModel::kHomography,
                            homography.num_inliers, homography);
                }
                return MakeDecision(
                        GeometryConfiguration::kPlanarOrPanoramic,
                        best_model, best_epipolar.num_inliers,
                        best_epipolar);
            }
            return MakeDecision(GeometryConfiguration::kCalibrated,
                                best_model, best_epipolar.num_inliers,
                                best_epipolar);
        }
        if (fundamental.success &&
            fundamental.num_inliers >= options.minimum_inliers) {
            const double homography_to_fundamental_ratio =
                    static_cast<double>(homography.num_inliers) /
                    static_cast<double>(fundamental.num_inliers);
            if (homography_to_fundamental_ratio >
                options.maximum_homography_ratio) {
                if (homography.num_inliers > fundamental.num_inliers) {
                    return MakeDecision(
                            GeometryConfiguration::kPlanarOrPanoramic,
                            InlierModel::kHomography,
                            homography.num_inliers, homography);
                }
                return MakeDecision(
                        GeometryConfiguration::kPlanarOrPanoramic,
                        InlierModel::kFundamental,
                        fundamental.num_inliers, fundamental);
            }
            return MakeDecision(GeometryConfiguration::kUncalibrated,
                                InlierModel::kFundamental,
                                fundamental.num_inliers, fundamental);
        }
        if (homography.success &&
            homography.num_inliers >= options.minimum_inliers) {
            return MakeDecision(GeometryConfiguration::kPlanarOrPanoramic,
                                InlierModel::kHomography,
                                homography.num_inliers, homography);
        }
        return MakeDecision(GeometryConfiguration::kDegenerate,
                            InlierModel::kNone, 0, essential);
    }
    const double homography_to_fundamental_ratio =
            static_cast<double>(homography.num_inliers) /
            static_cast<double>(fundamental.num_inliers);
    if (fundamental.success &&
        fundamental.num_inliers >= options.minimum_inliers) {
        if (homography_to_fundamental_ratio >
            options.maximum_homography_ratio) {
            if (homography.num_inliers > fundamental.num_inliers) {
                return MakeDecision(GeometryConfiguration::kPlanarOrPanoramic,
                                    InlierModel::kHomography,
                                    homography.num_inliers, homography);
            }
            return MakeDecision(GeometryConfiguration::kPlanarOrPanoramic,
                                InlierModel::kFundamental,
                                fundamental.num_inliers, fundamental);
        }
        return MakeDecision(GeometryConfiguration::kUncalibrated,
                            InlierModel::kFundamental,
                            fundamental.num_inliers, fundamental);
    }
    if (homography.success &&
        homography.num_inliers >= options.minimum_inliers) {
        return MakeDecision(GeometryConfiguration::kPlanarOrPanoramic,
                            InlierModel::kHomography,
                            homography.num_inliers, homography);
    }
    return MakeDecision(GeometryConfiguration::kDegenerate,
                        InlierModel::kNone, 0, essential);
}

bool EstimateInitialPairGeometry(
        const std::vector<Eigen::Vector2d>& pixels1,
        const std::vector<Eigen::Vector2d>& pixels2,
        const Eigen::Matrix3d& camera_matrix,
        const bool both_cameras_have_focal_priors,
        const double max_error_pixels,
        InitialPairGeometryReport* report,
        const RansacOptions& epipolar_options,
        const RansacOptions& homography_options,
        const TwoViewDispatchOptions& dispatch_options) {
    if (report == nullptr || pixels1.size() != pixels2.size() ||
        pixels1.size() < 7 || pixels1.size() > kMaxAndroidInputObservations ||
        !std::isfinite(max_error_pixels) || max_error_pixels <= 0.0 ||
        !IsFiniteMatrix(camera_matrix)) {
        return false;
    }
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        if (!IsFiniteVector(pixels1[i]) || !IsFiniteVector(pixels2[i])) {
            return false;
        }
    }
    if (!EstimateTwoViewEpipolarGeometry(
                pixels1, pixels2, camera_matrix, max_error_pixels,
                &report->epipolar, epipolar_options)) {
        return false;
    }
    report->homography = source_homography::RunHomographyLORansac(
            pixels1, pixels2, homography_options);
    report->decision = DispatchTwoViewGeometry(
            both_cameras_have_focal_priors,
            ToModelSupport(report->epipolar.essential),
            ToModelSupport(report->epipolar.fundamental),
            ToModelSupport(report->homography), dispatch_options);
    return true;
}

bool RecoverInitialPairPose(
        const InitialPairGeometryReport& geometry,
        const std::vector<Eigen::Vector2d>& pixels1,
        const std::vector<Eigen::Vector2d>& pixels2,
        const Eigen::Matrix3d& camera_matrix,
        const source_homography::InitialPairGates& homography_gates,
        InitialPairPoseResult* result) {
    if (result == nullptr || pixels1.size() != pixels2.size() ||
        pixels1.empty() || pixels1.size() > kMaxAndroidInputObservations ||
        geometry.decision.selected_inlier_mask.size() != pixels1.size() ||
        geometry.decision.num_inliers != static_cast<std::size_t>(
                std::count_if(geometry.decision.selected_inlier_mask.begin(),
                              geometry.decision.selected_inlier_mask.end(),
                              [](const char inlier) { return inlier != 0; })) ||
        !IsFiniteMatrix(camera_matrix)) {
        return false;
    }
    *result = InitialPairPoseResult{};
    result->selected_inlier_mask = geometry.decision.selected_inlier_mask;
    result->cheirality_mask.assign(pixels1.size(), 0);
    if (geometry.decision.pose_source == PoseSource::kHomography) {
        if (!geometry.homography.success || !geometry.homography.model.has_value()) {
            return false;
        }
        std::vector<std::size_t> selected_indices;
        std::vector<source_homography::Ray> rays1;
        std::vector<source_homography::Ray> rays2;
        selected_indices.reserve(geometry.decision.num_inliers);
        rays1.reserve(geometry.decision.num_inliers);
        rays2.reserve(geometry.decision.num_inliers);
        const Eigen::Matrix3d inverse_camera = camera_matrix.inverse();
        if (!IsFiniteMatrix(inverse_camera)) {
            return false;
        }
        for (std::size_t i = 0; i < pixels1.size(); ++i) {
            if (geometry.decision.selected_inlier_mask[i] == 0) {
                continue;
            }
            if (!IsFiniteVector(pixels1[i]) || !IsFiniteVector(pixels2[i])) {
                return false;
            }
            Eigen::Vector3d ray1 = inverse_camera * pixels1[i].homogeneous();
            Eigen::Vector3d ray2 = inverse_camera * pixels2[i].homogeneous();
            const double norm1 = ray1.norm();
            const double norm2 = ray2.norm();
            if (!std::isfinite(norm1) || !std::isfinite(norm2) ||
                norm1 <= std::numeric_limits<double>::epsilon() ||
                norm2 <= std::numeric_limits<double>::epsilon()) {
                return false;
            }
            rays1.push_back(ray1 / norm1);
            rays2.push_back(ray2 / norm2);
            selected_indices.push_back(i);
        }
        const source_homography::HomographyPoseReport pose =
                source_homography::RecoverHomographyPose(
                        *geometry.homography.model, camera_matrix,
                        camera_matrix, rays1, rays2);
        if (pose.candidates.empty() ||
            pose.selected_index >= pose.candidates.size()) {
            return false;
        }
        const source_homography::HomographyPoseCandidate& selected =
                pose.candidates[pose.selected_index];
        if (selected.positive_depth_mask.size() != selected_indices.size()) {
            return false;
        }
        result->rotation = selected.rotation.toRotationMatrix();
        result->translation = selected.translation;
        result->selected_candidate = pose.selected_index;
        result->positive_depth_count = selected.positive_depth_count;
        result->median_triangulation_angle_radians =
                pose.median_triangulation_angle;
        for (std::size_t i = 0; i < selected_indices.size(); ++i) {
            if (selected.positive_depth_mask[i] != 0) {
                result->cheirality_mask[selected_indices[i]] = 1;
            }
        }
        result->success = true;
        result->passes_initial_pair_gates =
                source_homography::PassInitialPairGates(
                        geometry.decision.num_inliers, pose, homography_gates);
        return true;
    }
    Eigen::Matrix3d essential;
    if (geometry.decision.pose_source == PoseSource::kEssential) {
        if (!geometry.epipolar.essential.success ||
            !geometry.epipolar.essential.model.has_value()) {
            return false;
        }
        essential = *geometry.epipolar.essential.model;
    } else if (geometry.decision.pose_source == PoseSource::kFundamental) {
        if (!geometry.epipolar.fundamental.success ||
            !geometry.epipolar.fundamental.model.has_value()) {
            return false;
        }
        essential = camera_matrix.transpose() *
                    *geometry.epipolar.fundamental.model * camera_matrix;
        const double scale = essential.cwiseAbs().maxCoeff();
        if (!std::isfinite(scale) || scale <= 1e-15) {
            return false;
        }
        essential /= scale;
    } else {
        return false;
    }
    std::vector<char> mask = geometry.decision.selected_inlier_mask;
    EssentialPoseResult pose;
    if (!RecoverEssentialPoseFromPixels(
                essential, pixels1, pixels2, camera_matrix, mask, &pose)) {
        return false;
    }
    if (pose.cheirality_mask.size() != pixels1.size()) {
        return false;
    }
    result->rotation = pose.rotation;
    result->translation = pose.translation;
    result->selected_candidate = pose.selected_candidate;
    result->positive_depth_count = pose.positive_depth_count;
    result->cheirality_mask = std::move(pose.cheirality_mask);
    result->success = true;
    result->passes_initial_pair_gates = true;
    return true;
}

}  // namespace tracking::source_two_view
