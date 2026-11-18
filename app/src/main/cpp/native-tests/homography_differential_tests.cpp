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
#include "homography_differential.h"
#include "homography_oracle_expectations.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace tracking::homography_test;
namespace oracle = homography_oracle_expectations;

void Require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct SyntheticScene {
    Matrix intrinsics = Matrix::Identity();
    Matrix rotation = Matrix::Identity();
    Ray translation = Ray::Zero();
    Matrix homography = Matrix::Identity();
    std::vector<Point> points1;
    std::vector<Point> points2;
    std::vector<Ray> rays1;
    std::vector<Ray> rays2;
    std::vector<char> true_inlier_mask;
};

Point Project(const Matrix& intrinsics, const Ray& point) {
    const Ray projected = intrinsics * point;
    return projected.head<2>() / projected.z();
}

SyntheticScene MakeScene(const std::string& kind,
                         const std::size_t inlier_count,
                         const std::size_t outlier_count,
                         const bool add_noise) {
    SyntheticScene scene;
    scene.intrinsics << 820.0, 0.0, 640.0,
                        0.0, 790.0, 360.0,
                        0.0, 0.0, 1.0;
    const Ray axis(0.25, 0.6, 0.31);
    const double angle = kind == "low" ? 0.025 : 0.16;
    scene.rotation = Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
    if (kind == "low") {
        scene.translation = Ray(0.045, -0.012, 0.003);
    } else if (kind == "wide") {
        scene.translation = Ray(3.2, -0.05, 0.02);
    } else if (kind == "pano") {
        scene.translation.setZero();
    } else {
        scene.translation = Ray(0.44, -0.065, 0.04);
    }
    const double plane_depth = kind == "wide" ? 2.7 : 5.0;
    scene.homography = scene.intrinsics *
            (scene.rotation - scene.translation * (-Ray::UnitZ()).transpose() /
                                      plane_depth) *
            scene.intrinsics.inverse();

    for (std::size_t i = 0; i < inlier_count; ++i) {
        const double depth = kind == "nonplanar"
                ? 3.4 + 0.21 * static_cast<double>(i % 17)
                : (kind == "wide" ? 2.7 : 5.0);
        const double x = -1.6 + 0.075 * static_cast<double>(i) +
                         0.3 * std::sin(0.31 * static_cast<double>(i));
        const double y = -0.9 + 0.13 * static_cast<double>(i % 9) +
                         0.12 * std::cos(0.47 * static_cast<double>(i));
        const Ray point1(x, y, depth);
        const Ray point2 = scene.rotation * point1 + scene.translation;
        Point pixel1 = Project(scene.intrinsics, point1);
        Point pixel2 = Project(scene.intrinsics, point2);
        if (add_noise) {
            pixel1.x() += 0.08 * std::sin(1.7 * static_cast<double>(i));
            pixel1.y() += 0.08 * std::cos(0.91 * static_cast<double>(i));
            pixel2.x() += 0.08 * std::sin(1.21 * static_cast<double>(i));
            pixel2.y() += 0.08 * std::cos(1.39 * static_cast<double>(i));
        }
        scene.points1.push_back(pixel1);
        scene.points2.push_back(pixel2);
        scene.rays1.push_back(
                (scene.intrinsics.inverse() * pixel1.homogeneous()).normalized());
        scene.rays2.push_back(
                (scene.intrinsics.inverse() * pixel2.homogeneous()).normalized());
        scene.true_inlier_mask.push_back(1);
    }
    for (std::size_t i = 0; i < outlier_count; ++i) {
        const Point pixel1(
                40.0 + std::fmod(137.0 * i + 71.0, 1200.0),
                30.0 + std::fmod(113.0 * i + 17.0, 680.0));
        const Point pixel2(
                40.0 + std::fmod(317.0 * i + 401.0, 1200.0),
                30.0 + std::fmod(193.0 * i + 319.0, 680.0));
        scene.points1.push_back(pixel1);
        scene.points2.push_back(pixel2);
        scene.rays1.push_back(
                (scene.intrinsics.inverse() * pixel1.homogeneous()).normalized());
        scene.rays2.push_back(
                (scene.intrinsics.inverse() * pixel2.homogeneous()).normalized());
        scene.true_inlier_mask.push_back(0);
    }
    return scene;
}

Matrix MatrixFromRowMajor(const double (&values)[9]) {
    Matrix result;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            result(row, col) = values[row * 3 + col];
        }
    }
    return result;
}

void RequireMatrix(const Matrix& actual,
                   const double (&expected)[9],
                   const std::string& label,
                   const double tolerance = 1e-8) {
    const Matrix expected_matrix = MatrixFromRowMajor(expected);
    Require(actual.allFinite() && (actual - expected_matrix).norm() < tolerance,
            label + " differs from its fixed pinned-source matrix");
}

std::string MaskString(const std::vector<char>& mask) {
    std::string result;
    result.reserve(mask.size());
    for (const char value : mask) {
        result.push_back(value ? '1' : '0');
    }
    return result;
}

void GatherPoseRays(const SyntheticScene& scene,
                    const std::vector<char>& mask,
                    std::vector<Ray>* rays1,
                    std::vector<Ray>* rays2) {
    rays1->clear();
    rays2->clear();
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            rays1->push_back(scene.rays1[i]);
            rays2->push_back(scene.rays2[i]);
        }
    }
}

void RequirePoseCandidates(
        const HomographyPoseReport& actual,
        const double (&expected_rotations)[2][9],
        const double (&expected_translations)[4][3],
        const std::size_t (&expected_positive_counts)[4],
        const double (&expected_residual_sums)[4],
        const std::size_t expected_selected_index,
        const std::string& label) {
    Require(actual.candidates.size() == 4,
            label + " decomposition did not preserve four ordered candidates");
    for (std::size_t i = 0; i < actual.candidates.size(); ++i) {
        const std::size_t rotation_index = i < 2 ? 0 : 1;
        const Matrix expected_rotation = MatrixFromRowMajor(
                expected_rotations[rotation_index]);
        const Matrix actual_rotation = actual.candidates[i].rotation.toRotationMatrix();
        Require((actual_rotation - expected_rotation).norm() < 1e-8,
                label + " pose candidate rotation/order differs from the oracle");
        const Ray expected_translation(expected_translations[i][0],
                                       expected_translations[i][1],
                                       expected_translations[i][2]);
        Require((actual.candidates[i].translation - expected_translation).norm() <
                        1e-8,
                label + " pose candidate translation/order differs from the oracle");
        Require(actual.candidates[i].positive_depth_count ==
                        expected_positive_counts[i],
                label + " pose candidate cheirality count differs from the oracle");
        Require(std::abs(actual.candidates[i].reprojection_residual_sum -
                         expected_residual_sums[i]) < 1e-8,
                label + " pose candidate reprojection score differs from the oracle");
    }
    Require(actual.selected_index == expected_selected_index,
            label + " selected pose index differs from source cheirality/reprojection ordering");
}

RansacOptions FixtureOptions() {
    RansacOptions options;
    options.max_error = 2.5;
    options.min_inlier_ratio = 0.2;
    options.confidence = 0.999;
    options.dyn_num_trials_multiplier = 3.0;
    options.min_num_trials = 20;
    options.max_num_trials = 180;
    options.random_seed = 20261118;
    return options;
}

void TestDltMatricesResidualsAndDegeneracy() {
    const SyntheticScene scene = MakeScene("planar", 40, 12, true);
    const std::vector<Point> first_four1(scene.points1.begin(), scene.points1.begin() + 4);
    const std::vector<Point> first_four2(scene.points2.begin(), scene.points2.begin() + 4);
    std::vector<Matrix> minimal_models;
    Require(EstimateDLT(first_four1, first_four2, &minimal_models) &&
                    minimal_models.size() == 1,
            "minimal four-point DLT failed the planar fixture");
    RequireMatrix(minimal_models.front(), oracle::kMinimalDlt,
                  "minimal four-point DLT");

    const std::vector<Point> first_twelve1(scene.points1.begin(), scene.points1.begin() + 12);
    const std::vector<Point> first_twelve2(scene.points2.begin(), scene.points2.begin() + 12);
    std::vector<Matrix> overdetermined_models;
    Require(EstimateDLT(first_twelve1, first_twelve2, &overdetermined_models) &&
                    overdetermined_models.size() == 1,
            "overdetermined DLT failed the planar fixture");
    RequireMatrix(overdetermined_models.front(), oracle::kOverdeterminedDlt,
                  "overdetermined DLT");
    std::vector<double> residuals;
    OneWaySquaredReprojectionResiduals(first_twelve1, first_twelve2,
                                       overdetermined_models.front(), &residuals);
    Require(residuals.size() == 12,
            "one-way residual vector has the wrong length");
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        Require(std::abs(residuals[i] - oracle::kFirstTwelveOneWayResiduals[i]) <
                        2e-12,
                "one-way squared reprojection residual differs from the pinned output");
    }

    std::vector<Matrix> degenerate_models;
    Require(!EstimateDLT({{0, 0}, {1, 0}, {2, 0}, {3, 0}},
                         {{1, 2}, {3, 2}, {5, 2}, {7, 2}},
                         &degenerate_models) && degenerate_models.empty(),
            "collinear four-point DLT should produce no model");
    Require(!EstimateDLT(std::vector<Point>(4, Point(1, 2)),
                         std::vector<Point>(4, Point(3, 4)),
                         &degenerate_models) && degenerate_models.empty(),
            "repeated-point four-point DLT should produce no model");
}

void TestPlanarLocalConsensusAndPoseCandidates() {
    const SyntheticScene scene = MakeScene("planar", 40, 12, true);
    const auto first = RunHomographyLORansac(scene.points1, scene.points2,
                                             FixtureOptions());
    const auto repeated = RunHomographyLORansac(scene.points1, scene.points2,
                                                FixtureOptions());
    Require(first.success && first.model.has_value(),
            "source-shaped homography LO-RANSAC rejected a planar fixture");
    Require(first.num_trials == oracle::kPlanarTrials &&
                    first.support.num_inliers == oracle::kPlanarInliers &&
                    std::abs(first.support.residual_sum - oracle::kPlanarResidualSum) <
                            1e-8,
            "planar LO-RANSAC support or adaptive trial result differs from the pinned output");
    Require(MaskString(first.inlier_mask) == oracle::kPlanarMask,
            "planar LO-RANSAC inlier mask/order differs from the pinned output");
    RequireMatrix(*first.model, oracle::kPlanarRansacModel,
                  "planar LO-RANSAC model");
    Require(first.best_model_is_local && first.accepted_local_refits > 0 &&
                    first.local_optimization_trials > 0,
            "planar LO-RANSAC did not exercise local DLT refinement");
    Require(repeated.success && repeated.num_trials == first.num_trials &&
                    repeated.inlier_mask == first.inlier_mask &&
                    repeated.model.has_value() &&
                    (*repeated.model - *first.model).norm() == 0.0 &&
                    repeated.support.residual_sum == first.support.residual_sum,
            "fixed-seed planar homography refit is not deterministic");

    std::vector<Point> refit1;
    std::vector<Point> refit2;
    for (std::size_t i = 0; i < first.inlier_mask.size(); ++i) {
        if (first.inlier_mask[i]) {
            refit1.push_back(scene.points1[i]);
            refit2.push_back(scene.points2[i]);
        }
    }
    std::vector<Matrix> final_refit;
    Require(EstimateDLT(refit1, refit2, &final_refit) && final_refit.size() == 1,
            "deterministic final inlier refit did not return a model");
    Require((final_refit.front() - *first.model).norm() < 1e-10,
            "LO-RANSAC best model is not the deterministic all-inlier DLT refit");

    std::vector<Ray> rays1;
    std::vector<Ray> rays2;
    GatherPoseRays(scene, first.inlier_mask, &rays1, &rays2);
    const HomographyPoseReport pose = RecoverHomographyPose(
            *first.model, scene.intrinsics, scene.intrinsics, rays1, rays2);
    RequirePoseCandidates(pose, oracle::kPlanarPoseRotations,
                          oracle::kPlanarPoseTranslations,
                          oracle::kPlanarPositiveDepthCounts,
                          oracle::kPlanarPoseResidualSums,
                          oracle::kPlanarSelectedPose, "planar");
    Require(std::abs(pose.median_triangulation_angle - 0.083157517744552531) <
                    1e-8,
            "planar median triangulation angle differs from the pinned pose path");
    Require(!PassInitialPairGates(oracle::kPlanarInliers, pose),
            "planar fixture below the 100-inlier/16-degree initial-pair gates was accepted");
}

void TestNonPlanarConsensusAndPoseCandidates() {
    const SyntheticScene scene = MakeScene("nonplanar", 48, 12, true);
    const auto report = RunHomographyLORansac(scene.points1, scene.points2,
                                              FixtureOptions());
    Require(report.success && report.model.has_value(),
            "homography LO-RANSAC did not return its source-shaped non-planar consensus");
    Require(report.num_trials == oracle::kNonPlanarTrials &&
                    report.support.num_inliers == oracle::kNonPlanarInliers &&
                    std::abs(report.support.residual_sum -
                             oracle::kNonPlanarResidualSum) < 1e-8,
            "non-planar LO-RANSAC support/trial result differs from the pinned output");
    Require(MaskString(report.inlier_mask) == oracle::kNonPlanarMask,
            "non-planar LO-RANSAC mask differs from the pinned output");
    RequireMatrix(*report.model, oracle::kNonPlanarRansacModel,
                  "non-planar LO-RANSAC model");

    std::vector<Point> refit1;
    std::vector<Point> refit2;
    std::vector<Ray> rays1;
    std::vector<Ray> rays2;
    for (std::size_t i = 0; i < report.inlier_mask.size(); ++i) {
        if (report.inlier_mask[i]) {
            refit1.push_back(scene.points1[i]);
            refit2.push_back(scene.points2[i]);
            rays1.push_back(scene.rays1[i]);
            rays2.push_back(scene.rays2[i]);
        }
    }
    std::vector<Matrix> refit;
    Require(EstimateDLT(refit1, refit2, &refit) && refit.size() == 1 &&
                    (refit.front() - *report.model).norm() < 1e-10,
            "non-planar deterministic local refit differs from the pinned H model");
    const HomographyPoseReport pose = RecoverHomographyPose(
            *report.model, scene.intrinsics, scene.intrinsics, rays1, rays2);
    RequirePoseCandidates(pose, oracle::kNonPlanarPoseRotations,
                          oracle::kNonPlanarPoseTranslations,
                          oracle::kNonPlanarPositiveDepthCounts,
                          oracle::kNonPlanarPoseResidualSums,
                          oracle::kNonPlanarSelectedPose, "non-planar");
    Require(std::abs(pose.median_triangulation_angle -
                     oracle::kNonPlanarMedianAngle) < 1e-8,
            "non-planar selected-pose median triangulation angle differs from the oracle");
}

void TestLowParallaxAndRotationOnlyPoseCases() {
    const SyntheticScene low = MakeScene("low", 32, 0, false);
    const HomographyPoseReport low_pose = RecoverHomographyPose(
            low.homography, low.intrinsics, low.intrinsics,
            low.rays1, low.rays2);
    RequirePoseCandidates(low_pose, oracle::kLowParallaxPoseRotations,
                          oracle::kLowParallaxPoseTranslations,
                          oracle::kLowParallaxPositiveDepthCounts,
                          oracle::kLowParallaxPoseResidualSums,
                          oracle::kLowParallaxSelectedPose, "low-parallax");
    Require(std::abs(low_pose.median_triangulation_angle -
                     oracle::kLowParallaxMedianAngle) < 1e-8 &&
                    !PassInitialPairGates(120, low_pose),
            "low-parallax pose did not retain its source angle and reject the 16-degree gate");

    const SyntheticScene pano = MakeScene("pano", 32, 0, false);
    const HomographyPoseReport pano_pose = RecoverHomographyPose(
            pano.homography, pano.intrinsics, pano.intrinsics,
            pano.rays1, pano.rays2);
    Require(pano_pose.candidates.size() == 1 &&
                    pano_pose.selected_index == 0 &&
                    pano_pose.candidates.front().translation.squaredNorm() == 0.0 &&
                    pano_pose.triangulated_points.empty() &&
                    pano_pose.median_triangulation_angle == 0.0,
            "rotation-only homography must preserve the single-candidate zero-translation path");
    Require((pano_pose.candidates.front().rotation.toRotationMatrix() -
             pano.rotation).norm() < 1e-8,
            "rotation-only decomposition changed the recovered rotation");
}

void TestExactInitialPairGateBoundaries() {
    const SyntheticScene wide = MakeScene("wide", 120, 0, false);
    const auto wide_fit = RunHomographyLORansac(
            wide.points1, wide.points2, FixtureOptions());
    Require(wide_fit.success && wide_fit.model.has_value() &&
                    wide_fit.num_trials == oracle::kWideParallaxTrials &&
                    wide_fit.support.num_inliers == oracle::kWideParallaxInliers &&
                    std::abs(wide_fit.support.residual_sum -
                             oracle::kWideParallaxResidualSum) < 1e-20 &&
                    std::all_of(wide_fit.inlier_mask.begin(),
                                wide_fit.inlier_mask.end(),
                                [](const char inlier) { return inlier != 0; }),
            "wide-parallax source H fit did not produce the fixed 120-inlier report");
    RequireMatrix(*wide_fit.model, oracle::kWideParallaxRansacModel,
                  "wide-parallax LO-RANSAC model");
    const HomographyPoseReport wide_pose = RecoverHomographyPose(
            *wide_fit.model, wide.intrinsics, wide.intrinsics,
            wide.rays1, wide.rays2);
    Require(wide_pose.triangulated_points.size() == oracle::kWideParallaxInliers &&
                    std::abs(wide_pose.median_triangulation_angle -
                             oracle::kWideParallaxMedianAngle) < 1e-8,
            "high-parallax triangulation points or median differ from the pinned output");
    Require((wide_pose.candidates[wide_pose.selected_index].translation -
             Ray(oracle::kWideParallaxSelectedTranslation[0],
                 oracle::kWideParallaxSelectedTranslation[1],
                 oracle::kWideParallaxSelectedTranslation[2])).norm() < 1e-8,
            "high-parallax selected translation differs from the pinned output");
    Require(PassInitialPairGates(wide_fit.support.num_inliers, wide_pose),
            "120-inlier, low-forward-motion, >16-degree source fixture must pass");
    Require(!PassInitialPairGates(99, wide_pose) &&
                    PassInitialPairGates(100, wide_pose),
            "the initial-pair minimum must be inclusive at 100 inliers");

    constexpr double kDegreesToRadians =
            0.0174532925199432954743716805978692718781530857086181640625;
    HomographyPoseReport at_forward_limit = wide_pose;
    at_forward_limit.candidates[at_forward_limit.selected_index].translation.z() = 0.95;
    Require(!PassInitialPairGates(100, at_forward_limit),
            "absolute z translation equal to 0.95 must be rejected");
    at_forward_limit.candidates[at_forward_limit.selected_index].translation.z() = -0.95;
    Require(!PassInitialPairGates(100, at_forward_limit),
            "negative absolute z translation equal to 0.95 must be rejected");
    at_forward_limit.candidates[at_forward_limit.selected_index].translation.z() = 0.949999;
    at_forward_limit.candidates[at_forward_limit.selected_index].translation.x() = 100.0;
    Require(PassInitialPairGates(100, at_forward_limit),
            "source gate must use abs(translation.z()), not translation norm or a separate ratio");

    HomographyPoseReport at_angle_limit = wide_pose;
    at_angle_limit.median_triangulation_angle = 16.0 * kDegreesToRadians;
    Require(!PassInitialPairGates(100, at_angle_limit),
            "median triangulation angle equal to 16 degrees must be rejected");
    at_angle_limit.median_triangulation_angle =
            std::nextafter(16.0 * kDegreesToRadians,
                           std::numeric_limits<double>::infinity());
    Require(PassInitialPairGates(100, at_angle_limit),
            "median triangulation angle strictly above 16 degrees must be accepted");
}

void RunAllTests() {
    TestDltMatricesResidualsAndDegeneracy();
    TestPlanarLocalConsensusAndPoseCandidates();
    TestNonPlanarConsensusAndPoseCandidates();
    TestLowParallaxAndRotationOnlyPoseCases();
    TestExactInitialPairGateBoundaries();
}

}  // namespace

int main() {
    try {
        RunAllTests();
        std::cout << "Homography differential fixtures passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Homography differential failure: " << error.what() << '\n';
        return 1;
    }
}
