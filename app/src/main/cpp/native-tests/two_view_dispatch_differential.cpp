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
#include "two_view_dispatch_differential.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tracking::two_view_dispatch_test {
namespace {

void Validate(const ModelSupport& report, const char* label) {
    if (report.success &&
        (report.inlier_mask.empty() ||
         static_cast<std::size_t>(std::count_if(
                 report.inlier_mask.begin(), report.inlier_mask.end(),
                 [](const char inlier) { return inlier != 0; })) !=
                 report.num_inliers)) {
        throw std::invalid_argument(std::string("invalid ") + label +
                                    " support report");
    }
}

Decision MakeDecision(const Configuration configuration,
                      const Model model,
                      const std::size_t num_inliers,
                      const ModelSupport& report) {
    Decision decision;
    decision.configuration = configuration;
    decision.selected_model = model;
    decision.num_inliers = num_inliers;
    if (model != Model::kNone) {
        decision.selected_inlier_mask = report.inlier_mask;
    }
    switch (configuration) {
        case Configuration::kCalibrated:
            decision.pose_source = PoseSource::kEssential;
            break;
        case Configuration::kUncalibrated:
            decision.pose_source = PoseSource::kFundamental;
            break;
        case Configuration::kPlanarOrPanoramic:
            decision.pose_source = PoseSource::kHomography;
            break;
        case Configuration::kDegenerate:
            decision.pose_source = PoseSource::kNone;
            break;
    }
    return decision;
}

}  // namespace

Decision Dispatch(const bool both_cameras_have_focal_priors,
                  const ModelSupport& essential,
                  const ModelSupport& fundamental,
                  const ModelSupport& homography,
                  const Options& options) {
    if (options.minimum_inliers == 0 ||
        options.minimum_essential_to_fundamental_ratio < 0.0 ||
        options.minimum_essential_to_fundamental_ratio > 1.0 ||
        options.maximum_homography_ratio < 0.0 ||
        options.maximum_homography_ratio > 1.0) {
        throw std::invalid_argument("invalid two-view dispatch options");
    }
    Validate(essential, "essential");
    Validate(fundamental, "fundamental");
    Validate(homography, "homography");

    const bool any_model_succeeded = essential.success || fundamental.success ||
                                     homography.success;
    const bool any_model_has_support =
            essential.num_inliers >= options.minimum_inliers ||
            fundamental.num_inliers >= options.minimum_inliers ||
            homography.num_inliers >= options.minimum_inliers;
    if (!any_model_succeeded || !any_model_has_support) {
        return MakeDecision(Configuration::kDegenerate, Model::kNone, 0,
                            essential);
    }

    if (both_cameras_have_focal_priors) {
        const double essential_to_fundamental_ratio =
                static_cast<double>(essential.num_inliers) /
                fundamental.num_inliers;
        const double homography_to_essential_ratio =
                static_cast<double>(homography.num_inliers) /
                essential.num_inliers;

        if (essential.success &&
            essential_to_fundamental_ratio >
                    options.minimum_essential_to_fundamental_ratio &&
            essential.num_inliers >= options.minimum_inliers) {
            // The pinned calibrated branch retains the larger E/F support;
            // E wins ties. H changes the configuration on a strict ratio
            // comparison, but replaces the support mask only when strictly
            // larger than the retained E/F support.
            const bool essential_is_best =
                    essential.num_inliers >= fundamental.num_inliers;
            const ModelSupport& best_epipolar =
                    essential_is_best ? essential : fundamental;
            const Model best_model = essential_is_best ? Model::kEssential
                                                       : Model::kFundamental;
            if (homography_to_essential_ratio >
                options.maximum_homography_ratio) {
                if (homography.num_inliers > best_epipolar.num_inliers) {
                    return MakeDecision(Configuration::kPlanarOrPanoramic,
                                        Model::kHomography,
                                        homography.num_inliers, homography);
                }
                return MakeDecision(Configuration::kPlanarOrPanoramic,
                                    best_model, best_epipolar.num_inliers,
                                    best_epipolar);
            }
            return MakeDecision(Configuration::kCalibrated, best_model,
                                best_epipolar.num_inliers, best_epipolar);
        }

        if (fundamental.success &&
            fundamental.num_inliers >= options.minimum_inliers) {
            const double homography_to_fundamental_ratio =
                    static_cast<double>(homography.num_inliers) /
                    fundamental.num_inliers;
            if (homography_to_fundamental_ratio >
                options.maximum_homography_ratio) {
                if (homography.num_inliers > fundamental.num_inliers) {
                    return MakeDecision(Configuration::kPlanarOrPanoramic,
                                        Model::kHomography,
                                        homography.num_inliers, homography);
                }
                return MakeDecision(Configuration::kPlanarOrPanoramic,
                                    Model::kFundamental,
                                    fundamental.num_inliers, fundamental);
            }
            return MakeDecision(Configuration::kUncalibrated,
                                Model::kFundamental,
                                fundamental.num_inliers, fundamental);
        }

        if (homography.success &&
            homography.num_inliers >= options.minimum_inliers) {
            return MakeDecision(Configuration::kPlanarOrPanoramic,
                                Model::kHomography, homography.num_inliers,
                                homography);
        }
        return MakeDecision(Configuration::kDegenerate, Model::kNone, 0,
                            essential);
    }

    const double homography_to_fundamental_ratio =
            static_cast<double>(homography.num_inliers) /
            fundamental.num_inliers;
    if (fundamental.success &&
        fundamental.num_inliers >= options.minimum_inliers) {
        if (homography_to_fundamental_ratio >
            options.maximum_homography_ratio) {
            if (homography.num_inliers > fundamental.num_inliers) {
                return MakeDecision(Configuration::kPlanarOrPanoramic,
                                    Model::kHomography,
                                    homography.num_inliers, homography);
            }
            return MakeDecision(Configuration::kPlanarOrPanoramic,
                                Model::kFundamental,
                                fundamental.num_inliers, fundamental);
        }
        return MakeDecision(Configuration::kUncalibrated,
                            Model::kFundamental, fundamental.num_inliers,
                            fundamental);
    }

    if (homography.success &&
        homography.num_inliers >= options.minimum_inliers) {
        return MakeDecision(Configuration::kPlanarOrPanoramic,
                            Model::kHomography, homography.num_inliers,
                            homography);
    }
    return MakeDecision(Configuration::kDegenerate, Model::kNone, 0,
                        essential);
}

const char* ToString(const Configuration configuration) {
    switch (configuration) {
        case Configuration::kDegenerate: return "DEGENERATE";
        case Configuration::kCalibrated: return "CALIBRATED";
        case Configuration::kUncalibrated: return "UNCALIBRATED";
        case Configuration::kPlanarOrPanoramic: return "PLANAR_OR_PANORAMIC";
    }
    return "UNKNOWN_CONFIGURATION";
}

const char* ToString(const Model model) {
    switch (model) {
        case Model::kNone: return "NONE";
        case Model::kEssential: return "ESSENTIAL";
        case Model::kFundamental: return "FUNDAMENTAL";
        case Model::kHomography: return "HOMOGRAPHY";
    }
    return "UNKNOWN_MODEL";
}

const char* ToString(const PoseSource source) {
    switch (source) {
        case PoseSource::kNone: return "NONE";
        case PoseSource::kEssential: return "ESSENTIAL";
        case PoseSource::kFundamental: return "FUNDAMENTAL";
        case PoseSource::kHomography: return "HOMOGRAPHY";
    }
    return "UNKNOWN_POSE_SOURCE";
}

}  // namespace tracking::two_view_dispatch_test
