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
#include "../two_view_geometry_core.h"

namespace tracking::two_view_dispatch_test {
namespace {
tracking::source_two_view::ModelSupport ToProductionSupport(
        const ModelSupport& support) {
    return {support.success, support.num_inliers, support.inlier_mask};
}
Configuration ToTestConfiguration(
        const tracking::source_two_view::GeometryConfiguration value) {
    switch (value) {
        case tracking::source_two_view::GeometryConfiguration::kDegenerate:
            return Configuration::kDegenerate;
        case tracking::source_two_view::GeometryConfiguration::kCalibrated:
            return Configuration::kCalibrated;
        case tracking::source_two_view::GeometryConfiguration::kUncalibrated:
            return Configuration::kUncalibrated;
        case tracking::source_two_view::GeometryConfiguration::kPlanarOrPanoramic:
            return Configuration::kPlanarOrPanoramic;
    }
    return Configuration::kDegenerate;
}
Model ToTestModel(const tracking::source_two_view::InlierModel value) {
    switch (value) {
        case tracking::source_two_view::InlierModel::kNone: return Model::kNone;
        case tracking::source_two_view::InlierModel::kEssential: return Model::kEssential;
        case tracking::source_two_view::InlierModel::kFundamental: return Model::kFundamental;
        case tracking::source_two_view::InlierModel::kHomography: return Model::kHomography;
    }
    return Model::kNone;
}
PoseSource ToTestPoseSource(const tracking::source_two_view::PoseSource value) {
    switch (value) {
        case tracking::source_two_view::PoseSource::kNone: return PoseSource::kNone;
        case tracking::source_two_view::PoseSource::kEssential: return PoseSource::kEssential;
        case tracking::source_two_view::PoseSource::kFundamental: return PoseSource::kFundamental;
        case tracking::source_two_view::PoseSource::kHomography: return PoseSource::kHomography;
    }
    return PoseSource::kNone;
}
}  // namespace

Decision Dispatch(const bool both_cameras_have_focal_priors,
                  const ModelSupport& essential,
                  const ModelSupport& fundamental,
                  const ModelSupport& homography,
                  const Options& options) {
    const tracking::source_two_view::TwoViewDispatchOptions production_options{
            options.minimum_inliers,
            options.minimum_essential_to_fundamental_ratio,
            options.maximum_homography_ratio};
    const tracking::source_two_view::TwoViewGeometryDecision production =
            tracking::source_two_view::DispatchTwoViewGeometry(
                    both_cameras_have_focal_priors,
                    ToProductionSupport(essential),
                    ToProductionSupport(fundamental),
                    ToProductionSupport(homography), production_options);
    return {ToTestConfiguration(production.configuration),
            ToTestModel(production.selected_model),
            ToTestPoseSource(production.pose_source),
            production.num_inliers,
            production.selected_inlier_mask};
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
