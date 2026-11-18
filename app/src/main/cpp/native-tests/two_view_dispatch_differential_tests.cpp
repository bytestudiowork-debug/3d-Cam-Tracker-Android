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
#include "homography_differential.h"
#include "../two_view_epipolar_estimator_core.h"
#include "../two_view_geometry_core.h"

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using tracking::homography_test::Matrix;
using tracking::homography_test::Point;
using tracking::homography_test::Ray;
using tracking::two_view_dispatch_test::Decision;
using tracking::two_view_dispatch_test::ModelSupport;
using tracking::two_view_dispatch_test::Options;

void Require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct SyntheticScene {
    std::vector<Point> points1;
    std::vector<Point> points2;
    std::vector<Ray> rays1;
    std::vector<Ray> rays2;
};

Point Project(const Matrix& intrinsics, const Ray& point) {
    const Ray projected = intrinsics * point;
    return projected.head<2>() / projected.z();
}

SyntheticScene MakeScene(const std::string& kind) {
    SyntheticScene scene;
    Matrix intrinsics;
    intrinsics << 800.0, 0.0, 640.0,
                  0.0, 800.0, 360.0,
                  0.0, 0.0, 1.0;
    const Matrix rotation = Eigen::AngleAxisd(
            0.13, Ray(0.25, 0.6, 0.31).normalized()).toRotationMatrix();
    const Ray translation(kind == "mixed_high"
                                  ? Ray(1.5, -0.08, 0.03)
                                  : Ray(0.65, -0.08, 0.03));
    constexpr std::size_t kPointCount = 120;
    scene.points1.reserve(kPointCount);
    scene.points2.reserve(kPointCount);
    scene.rays1.reserve(kPointCount);
    scene.rays2.reserve(kPointCount);

    for (std::size_t i = 0; i < kPointCount; ++i) {
        const double index = static_cast<double>(i);
        const double x = -2.0 + 0.037 * index + 0.31 * std::sin(0.31 * index);
        const double y = -1.0 + 0.16 * static_cast<double>(i % 11) +
                         0.17 * std::cos(0.47 * index);
        double depth = 4.5;
        if (kind == "nonplanar") {
            depth = 2.8 + 0.19 * static_cast<double>(i % 17);
        } else if (kind == "mixed" || kind == "mixed_high") {
            depth = i < 100 ? 4.5 : 2.2 + 0.71 * static_cast<double>(i % 6);
        }
        const Ray point1(x, y, depth);
        const Ray point2 = rotation * point1 + translation;
        const Point pixel1 = Project(intrinsics, point1);
        const Point pixel2 = Project(intrinsics, point2);
        scene.points1.push_back(pixel1);
        scene.points2.push_back(pixel2);
        scene.rays1.push_back(
                (intrinsics.inverse() * pixel1.homogeneous()).normalized());
        scene.rays2.push_back(
                (intrinsics.inverse() * pixel2.homogeneous()).normalized());
    }
    return scene;
}

std::string MaskString(const std::vector<char>& mask) {
    std::string result;
    result.reserve(mask.size());
    for (const char value : mask) result.push_back(value ? '1' : '0');
    return result;
}

ModelSupport ToSupport(const tracking::source_two_view::LORansacReport<Matrix>& report) {
    return {report.success, report.support.num_inliers, report.inlier_mask};
}

struct FixtureResult {
    tracking::source_two_view::EpipolarGeometryReport epipolar;
    tracking::homography_test::LORansacReport<Matrix> homography;
    Decision decision;
    tracking::source_two_view::InitialPairGeometryReport integrated_geometry;
    tracking::source_two_view::InitialPairPoseResult integrated_pose;
    bool integrated_pose_recovered = false;
};

FixtureResult RunFixture(const std::string& kind,
                         const bool both_cameras_have_focal_priors) {
    const SyntheticScene scene = MakeScene(kind);
    Matrix intrinsics;
    intrinsics << 800.0, 0.0, 640.0,
                  0.0, 800.0, 360.0,
                  0.0, 0.0, 1.0;

    tracking::source_two_view::RansacOptions epipolar_options;
    epipolar_options.max_error = 4.0;
    epipolar_options.min_inlier_ratio = 0.25;
    epipolar_options.confidence = 0.999;
    epipolar_options.dyn_num_trials_multiplier = 3.0;
    epipolar_options.min_num_trials = 100;
    epipolar_options.max_num_trials = 10000;
    epipolar_options.random_seed = 20261118;
    FixtureResult result;
    Require(tracking::source_two_view::EstimateTwoViewEpipolarGeometry(
                    scene.points1, scene.points2, intrinsics, 4.0,
                    &result.epipolar, epipolar_options),
            "native E/F estimator rejected deterministic dispatch fixture");

    tracking::source_two_view::RansacOptions homography_options;
    homography_options.max_error = 4.0;
    homography_options.min_inlier_ratio = 0.25;
    homography_options.confidence = 0.999;
    homography_options.dyn_num_trials_multiplier = 3.0;
    homography_options.min_num_trials = 100;
    homography_options.max_num_trials = 10000;
    homography_options.random_seed = 20261118;
    result.homography = tracking::homography_test::RunHomographyLORansac(
            scene.points1, scene.points2, homography_options);

    result.decision = tracking::two_view_dispatch_test::Dispatch(
            both_cameras_have_focal_priors,
            ToSupport(result.epipolar.essential),
            ToSupport(result.epipolar.fundamental),
            ToSupport(result.homography), Options{});
    Require(tracking::source_two_view::EstimateInitialPairGeometry(
                    scene.points1, scene.points2, intrinsics,
                    both_cameras_have_focal_priors, 4.0,
                    &result.integrated_geometry, epipolar_options,
                    homography_options),
            "integrated production estimator rejected pinned fixture");
    const auto& integrated = result.integrated_geometry;
    Require(integrated.epipolar.essential.inlier_mask ==
                            result.epipolar.essential.inlier_mask &&
                    integrated.epipolar.fundamental.inlier_mask ==
                            result.epipolar.fundamental.inlier_mask &&
                    integrated.homography.inlier_mask ==
                            result.homography.inlier_mask &&
                    integrated.decision.selected_inlier_mask ==
                            result.decision.selected_inlier_mask,
            "integrated path changed a model or selected mask");
    result.integrated_pose_recovered =
            tracking::source_two_view::RecoverInitialPairPose(
                    integrated, scene.points1, scene.points2, intrinsics,
                    tracking::source_homography::InitialPairGates{},
                    &result.integrated_pose);
    Require(result.integrated_pose_recovered,
            "integrated path failed to recover the dispatched pose source");
    Require(result.integrated_pose.success &&
                    result.integrated_pose.selected_inlier_mask ==
                            integrated.decision.selected_inlier_mask &&
                    result.integrated_pose.cheirality_mask.size() ==
                            scene.points1.size(),
            "integrated pose did not preserve the selected-mask contract");
    return result;
}

struct OracleExpectation {
    const char* name;
    const char* scene_kind;
    bool both_cameras_have_focal_priors;
    std::size_t essential_support;
    std::size_t fundamental_support;
    std::size_t homography_support;
    const char* homography_mask;
    tracking::two_view_dispatch_test::Configuration configuration;
    tracking::two_view_dispatch_test::Model selected_model;
    tracking::two_view_dispatch_test::PoseSource pose_source;
};

std::vector<char> MaskWithFirstNInliers(const std::size_t size,
                                        const std::size_t inlier_count) {
    Require(inlier_count <= size, "invalid synthetic report mask size");
    std::vector<char> mask(size, 0);
    for (std::size_t i = 0; i < inlier_count; ++i) mask[i] = 1;
    return mask;
}

void CheckPinnedReferenceFixture(const OracleExpectation& expected) {
    const FixtureResult result = RunFixture(
            expected.scene_kind, expected.both_cameras_have_focal_priors);
    const std::string label(expected.name);
    const std::string all_inliers(120, '1');
    Require(result.epipolar.essential.success &&
                    result.epipolar.essential.support.num_inliers ==
                            expected.essential_support &&
                    MaskString(result.epipolar.essential.inlier_mask) ==
                            all_inliers,
            label + ": essential report differs from the pinned reference");
    Require(result.epipolar.fundamental.success &&
                    result.epipolar.fundamental.support.num_inliers ==
                            expected.fundamental_support &&
                    MaskString(result.epipolar.fundamental.inlier_mask) ==
                            all_inliers,
            label + ": fundamental report differs from the pinned reference");
    Require(result.homography.success &&
                    result.homography.support.num_inliers ==
                            expected.homography_support &&
                    MaskString(result.homography.inlier_mask) ==
                            expected.homography_mask,
            label + ": homography report differs from the pinned reference");
    Require(result.decision.configuration == expected.configuration,
            label + ": selected configuration differs from the pinned reference");
    Require(result.decision.selected_model == expected.selected_model,
            label + ": selected inlier model differs from the pinned reference");
    Require(result.decision.pose_source == expected.pose_source,
            label + ": pose source differs from the pinned reference");
    Require(result.decision.num_inliers == 120 &&
                    MaskString(result.decision.selected_inlier_mask) ==
                            all_inliers,
            label + ": selected inlier mask differs from the pinned reference");
    Require(result.integrated_geometry.decision.configuration ==
                            (expected.pose_source ==
                                            tracking::two_view_dispatch_test::PoseSource::kHomography
                                    ? tracking::source_two_view::GeometryConfiguration::kPlanarOrPanoramic
                                    : expected.both_cameras_have_focal_priors
                                            ? tracking::source_two_view::GeometryConfiguration::kCalibrated
                                            : tracking::source_two_view::GeometryConfiguration::kUncalibrated) &&
                    result.integrated_geometry.decision.num_inliers == 120 &&
                    result.integrated_pose.success,
            label + ": integrated production geometry/pose differs from pinned dispatch");
    if (expected.pose_source ==
        tracking::two_view_dispatch_test::PoseSource::kHomography) {
        Require(result.integrated_geometry.decision.pose_source ==
                                tracking::source_two_view::PoseSource::kHomography,
                label + ": integrated path did not use H pose recovery");
        if (std::string(expected.scene_kind) == "mixed") {
            Require(!result.integrated_pose.passes_initial_pair_gates &&
                            result.integrated_pose.median_triangulation_angle_radians <=
                                    16.0 * 3.14159265358979323846 / 180.0,
                    label + ": mixed-depth H pose did not fail the strict parallax gate (angle=" +
                            std::to_string(result.integrated_pose
                                                   .median_triangulation_angle_radians *
                                           180.0 / 3.14159265358979323846) +
                            ", tz=" +
                            std::to_string(result.integrated_pose.translation.z()) + ")");
        } else {
            Require(!result.integrated_pose.passes_initial_pair_gates,
                    label + ": planar H pose unexpectedly passed initial-pair gates");
        }
    }
}

void TestPinnedDispatchDifferentialCases() {
    // Frozen reports and final selected masks were generated by compiling the
    // full independent reference at c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
    const std::string all_inliers(120, '1');
    const std::string nonplanar_h_mask =
            "000000000000000001111100000000000011111111110000000000001111111111"
            "000000000001111111100000000000000111000000000000000000";
    const std::string mixed_h_mask =
            "111111111111111111111111111111111111111111111111111111111111"
            "111111111111111111111111111111111111111100000100000100000100";
    const OracleExpectation cases[] = {
            {"calibrated-nonplanar", "nonplanar", true, 120, 120, 36,
             nonplanar_h_mask.c_str(),
             tracking::two_view_dispatch_test::Configuration::kCalibrated,
             tracking::two_view_dispatch_test::Model::kEssential,
             tracking::two_view_dispatch_test::PoseSource::kEssential},
            {"unknown-nonplanar", "nonplanar", false, 120, 120, 36,
             nonplanar_h_mask.c_str(),
             tracking::two_view_dispatch_test::Configuration::kUncalibrated,
             tracking::two_view_dispatch_test::Model::kFundamental,
             tracking::two_view_dispatch_test::PoseSource::kFundamental},
            {"calibrated-mixed", "mixed", true, 120, 120, 103,
             mixed_h_mask.c_str(),
             tracking::two_view_dispatch_test::Configuration::kPlanarOrPanoramic,
             tracking::two_view_dispatch_test::Model::kEssential,
             tracking::two_view_dispatch_test::PoseSource::kHomography},
            {"unknown-mixed", "mixed", false, 120, 120, 103,
             mixed_h_mask.c_str(),
             tracking::two_view_dispatch_test::Configuration::kPlanarOrPanoramic,
             tracking::two_view_dispatch_test::Model::kFundamental,
             tracking::two_view_dispatch_test::PoseSource::kHomography},
            {"calibrated-planar", "planar", true, 120, 120, 120,
             all_inliers.c_str(),
             tracking::two_view_dispatch_test::Configuration::kPlanarOrPanoramic,
             tracking::two_view_dispatch_test::Model::kEssential,
             tracking::two_view_dispatch_test::PoseSource::kHomography},
            {"unknown-planar", "planar", false, 120, 120, 120,
             all_inliers.c_str(),
             tracking::two_view_dispatch_test::Configuration::kPlanarOrPanoramic,
             tracking::two_view_dispatch_test::Model::kFundamental,
             tracking::two_view_dispatch_test::PoseSource::kHomography},
    };
    for (const OracleExpectation& expected : cases) {
        CheckPinnedReferenceFixture(expected);
    }
}

void TestMaskSelectionAndStrictThresholds() {
    using tracking::two_view_dispatch_test::Configuration;
    using tracking::two_view_dispatch_test::Dispatch;
    using tracking::two_view_dispatch_test::Model;
    using tracking::two_view_dispatch_test::PoseSource;

    const ModelSupport essential{true, 100, MaskWithFirstNInliers(130, 100)};
    const ModelSupport fundamental{true, 105, MaskWithFirstNInliers(130, 105)};
    const ModelSupport homography{true, 130, MaskWithFirstNInliers(130, 130)};
    const Decision homography_wins =
            Dispatch(true, essential, fundamental, homography);
    Require(homography_wins.configuration ==
                            Configuration::kPlanarOrPanoramic &&
                    homography_wins.selected_model == Model::kHomography &&
                    homography_wins.pose_source == PoseSource::kHomography &&
                    homography_wins.num_inliers == 130 &&
                    homography_wins.selected_inlier_mask ==
                            homography.inlier_mask,
            "strictly greater H support must select the H mask and H pose");

    const ModelSupport tied_homography{true, 105,
                                       MaskWithFirstNInliers(130, 105)};
    const Decision epipolar_wins_tie =
            Dispatch(true, essential, fundamental, tied_homography);
    Require(epipolar_wins_tie.configuration ==
                            Configuration::kPlanarOrPanoramic &&
                    epipolar_wins_tie.selected_model == Model::kFundamental &&
                    epipolar_wins_tie.pose_source == PoseSource::kHomography &&
                    epipolar_wins_tie.selected_inlier_mask ==
                            fundamental.inlier_mask,
            "equal H/E/F support must retain the E/F mask while using H pose");

    const Decision unknown_homography_wins = Dispatch(
            false, ModelSupport{},
            ModelSupport{true, 100, MaskWithFirstNInliers(130, 100)},
            ModelSupport{true, 120, MaskWithFirstNInliers(130, 120)});
    Require(unknown_homography_wins.selected_model == Model::kHomography &&
                    unknown_homography_wins.pose_source ==
                            PoseSource::kHomography &&
                    unknown_homography_wins.num_inliers == 120,
            "unknown-focal H support greater than F must select the H mask");

    const Decision essential_fundamental_boundary = Dispatch(
            true, ModelSupport{true, 95, MaskWithFirstNInliers(120, 95)},
            ModelSupport{true, 100, MaskWithFirstNInliers(120, 100)},
            ModelSupport{});
    Require(essential_fundamental_boundary.configuration ==
                            Configuration::kUncalibrated &&
                    essential_fundamental_boundary.selected_model ==
                            Model::kFundamental &&
                    essential_fundamental_boundary.pose_source ==
                            PoseSource::kFundamental,
            "the E/F ratio threshold must remain a strict greater-than");

    const Decision homography_ratio_boundary = Dispatch(
            true, ModelSupport{true, 100, MaskWithFirstNInliers(120, 100)},
            ModelSupport{true, 100, MaskWithFirstNInliers(120, 100)},
            ModelSupport{true, 80, MaskWithFirstNInliers(120, 80)});
    Require(homography_ratio_boundary.configuration ==
                            Configuration::kCalibrated &&
                    homography_ratio_boundary.selected_model ==
                            Model::kEssential &&
                    homography_ratio_boundary.pose_source ==
                            PoseSource::kEssential,
            "the H ratio threshold must remain a strict greater-than");
}

void TestHighParallaxIntegratedHomographyPose() {
    const std::string all_inliers(120, '1');
    for (const bool has_focal_prior : {true, false}) {
        const FixtureResult result = RunFixture("mixed_high", has_focal_prior);
        const std::string label = has_focal_prior ? "calibrated high-parallax" :
                                                    "unknown-focal high-parallax";
        const double median_degrees =
                result.integrated_pose.median_triangulation_angle_radians *
                180.0 / 3.14159265358979323846;
        Require(result.integrated_geometry.decision.configuration ==
                                tracking::source_two_view::GeometryConfiguration::kPlanarOrPanoramic &&
                        result.integrated_geometry.decision.pose_source ==
                                tracking::source_two_view::PoseSource::kHomography &&
                        result.integrated_geometry.decision.selected_inlier_mask.size() ==
                                all_inliers.size() &&
                        MaskString(result.integrated_geometry.decision.selected_inlier_mask) ==
                                all_inliers &&
                        result.integrated_pose.passes_initial_pair_gates,
                label + " did not preserve E/F correspondences and pass H gates (E=" +
                        std::to_string(result.epipolar.essential.support.num_inliers) +
                        ", F=" + std::to_string(result.epipolar.fundamental.support.num_inliers) +
                        ", H=" + std::to_string(result.homography.support.num_inliers) +
                        ", angle=" + std::to_string(median_degrees) +
                        ", tz=" + std::to_string(result.integrated_pose.translation.z()) + ")");
        const auto expected_model = has_focal_prior
                ? tracking::two_view_dispatch_test::Model::kEssential
                : tracking::two_view_dispatch_test::Model::kFundamental;
        Require(result.decision.selected_model == expected_model,
                label + " selected a different correspondence model");
    }
}

}  // namespace

int main() {
    try {
        TestPinnedDispatchDifferentialCases();
        TestMaskSelectionAndStrictThresholds();
        TestHighParallaxIntegratedHomographyPose();
        std::cout << "Two-view dispatch differential tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Two-view dispatch differential failure: " << error.what()
                  << '\n';
        return 1;
    }
}
