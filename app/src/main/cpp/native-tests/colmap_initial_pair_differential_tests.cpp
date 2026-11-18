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
//
// Differential fixtures for COLMAP c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
// These tests intentionally do not invoke or modify production estimator code.
#include "colmap_initial_pair_differential.h"
#include "colmap_five_point_differential.h"
#include "colmap_five_point_pinned_expectations.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/SVD>

namespace {
using tracking::colmap_initial_pair_test::ComputeNumTrials;
using tracking::colmap_initial_pair_test::EvaluateInlierSupport;
using tracking::colmap_initial_pair_test::IsLeftSupportBetter;
using tracking::colmap_initial_pair_test::RecoverEssentialPoseHypotheses;
using tracking::colmap_initial_pair_test::RansacOptions;
using tracking::colmap_initial_pair_test::RunLORansac;
using tracking::colmap_initial_pair_test::SquaredSampsonError;

void Require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Eigen::Matrix3d Skew(const Eigen::Vector3d& vector) {
    Eigen::Matrix3d result;
    result << 0.0, -vector.z(), vector.y(),
              vector.z(), 0.0, -vector.x(),
              -vector.y(), vector.x(), 0.0;
    return result;
}

double SymmetricEpipolarError(const Eigen::Matrix3d& fundamental,
                              const Eigen::Vector3d& point1,
                              const Eigen::Vector3d& point2) {
    const Eigen::Vector3d line2 = fundamental * point1;
    const Eigen::Vector3d line1 = fundamental.transpose() * point2;
    const double numerator = std::abs(point2.dot(line2));
    return std::max(numerator / line2.head<2>().norm(),
                    numerator / line1.head<2>().norm());
}

void TestSquaredSampsonResidualAndPixelPathDivergence() {
    Eigen::Matrix3d essential;
    essential << 0.0, 0.0, 0.0,
                 0.0, 0.0, -1.0,
                 0.0, 1.0, 0.0;
    const Eigen::Vector3d ray1(0.25, 0.10, 1.0);
    const Eigen::Vector3d ray2(0.45, 0.106, 1.0);
    const double residual = SquaredSampsonError(ray1, ray2, essential);
    Require(std::abs(residual - 1.8e-5) < 1e-12,
            "pinned COLMAP squared Sampson residual changed from 1.8e-5");

    constexpr double focal = 800.0;
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();
    intrinsics(0, 0) = focal;
    intrinsics(1, 1) = focal;
    intrinsics(0, 2) = 640.0;
    intrinsics(1, 2) = 360.0;
    const Eigen::Matrix3d fundamental =
            intrinsics.inverse().transpose() * essential * intrinsics.inverse();
    const Eigen::Vector3d pixel1(focal * ray1.x() + 640.0,
                                 focal * ray1.y() + 360.0, 1.0);
    const Eigen::Vector3d pixel2(focal * ray2.x() + 640.0,
                                 focal * ray2.y() + 360.0, 1.0);
    const double android_pixel_residual =
            SymmetricEpipolarError(fundamental, pixel1, pixel2);
    Require(std::abs(android_pixel_residual - 4.8) < 1e-9,
            "Android-style symmetric pixel residual changed from 4.8 px");
    Require(android_pixel_residual > 4.0,
            "Android-style residual should reject at 4 px");
    const double normalized_threshold = 4.0 / focal;
    Require(residual <= normalized_threshold * normalized_threshold,
            "COLMAP-style normalized Sampson residual should accept at 4 px");

    const double degenerate = SquaredSampsonError(
            ray1, ray2, Eigen::Matrix3d::Zero());
    Require(degenerate == std::numeric_limits<double>::max(),
            "zero Sampson denominator must return the source sentinel");
}

void TestSquaredSampsonSupportMaskAndRanking() {
    Eigen::Matrix3d essential;
    essential << 0.0, 0.0, 0.0,
                 0.0, 0.0, -1.0,
                 0.0, 1.0, 0.0;
    std::vector<double> residuals;
    for (int i = 0; i < 8; ++i) {
        const double x = -0.3 + 0.1 * i;
        const double y = -0.15 + 0.03 * i;
        const double displacement = (i == 2 || i == 6) ? 0.02 : 0.0;
        residuals.push_back(SquaredSampsonError(
                Eigen::Vector3d(x, y, 1.0),
                Eigen::Vector3d(x + 0.2, y + displacement, 1.0), essential));
    }
    constexpr double max_error = 0.005;
    const double max_residual = max_error * max_error;
    const auto support = EvaluateInlierSupport(residuals, max_residual);
    Require(support.num_inliers == 6,
            "squared-Sampson support count differs from the synthetic oracle");
    std::vector<char> mask(residuals.size());
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        mask[i] = residuals[i] <= max_residual;
    }
    const std::vector<char> expected = {1, 1, 0, 1, 1, 1, 0, 1};
    Require(mask == expected,
            "squared-Sampson inlier mask differs from the fixed synthetic fixture");

    const auto tied_higher_residual =
            tracking::colmap_initial_pair_test::InlierSupport{6, 1.0};
    const auto tied_lower_residual =
            tracking::colmap_initial_pair_test::InlierSupport{6, 0.5};
    Require(IsLeftSupportBetter(tied_lower_residual, tied_higher_residual),
            "equal-cardinality support must prefer lower inlier residual sum");
    Require(!IsLeftSupportBetter(tied_higher_residual, tied_lower_residual),
            "support ranking must not prefer a larger residual sum on a tie");
}

struct LineModel {
    double slope = 0.0;
    double intercept = 0.0;
};

struct LineObservation {
    double x = 0.0;
    double y = 0.0;
};

std::vector<LineObservation> MakeLineFixture(const std::size_t inlier_count,
                                             const std::size_t outlier_count) {
    std::vector<LineObservation> observations;
    observations.reserve(inlier_count + outlier_count);
    for (std::size_t i = 0; i < inlier_count; ++i) {
        const double x = (static_cast<double>(i) -
                          0.5 * static_cast<double>(inlier_count)) * 0.4;
        const double noise = 0.12 * std::sin(static_cast<double>(i) * 1.7);
        observations.push_back({x, 1.7 * x - 2.0 + noise});
    }
    for (std::size_t i = 0; i < outlier_count; ++i) {
        const double x = (static_cast<double>(i) -
                          0.5 * static_cast<double>(outlier_count)) * 0.4;
        observations.push_back({x, 25.0 + 0.3 * static_cast<double>(i)});
    }
    return observations;
}

void EstimateLineFromPair(const std::vector<LineObservation>& observations,
                          const std::vector<std::size_t>& indices,
                          std::vector<LineModel>* models) {
    models->clear();
    if (indices.size() < 2) {
        return;
    }
    const auto& first = observations[indices[0]];
    const auto& second = observations[indices[1]];
    const double dx = second.x - first.x;
    if (std::abs(dx) <= 1e-12) {
        return;
    }
    const double slope = (second.y - first.y) / dx;
    models->push_back({slope, first.y - slope * first.x});
}

void RefitLineLeastSquares(const std::vector<LineObservation>& observations,
                           const std::vector<std::size_t>& indices,
                           std::vector<LineModel>* models) {
    models->clear();
    if (indices.size() < 2) {
        return;
    }
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    for (const std::size_t index : indices) {
        const double x = observations[index].x;
        const double y = observations[index].y;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    const double count = static_cast<double>(indices.size());
    const double denominator = count * sum_xx - sum_x * sum_x;
    if (std::abs(denominator) <= 1e-12) {
        return;
    }
    const double slope = (count * sum_xy - sum_x * sum_y) / denominator;
    models->push_back({slope, (sum_y - slope * sum_x) / count});
}

struct LineRun {
    tracking::colmap_initial_pair_test::LORansacReport<LineModel> report;
    std::vector<LineObservation> observations;
};

LineRun RunLineFixture(const std::size_t inlier_count,
                       const std::size_t outlier_count,
                       const RansacOptions& options) {
    LineRun run;
    run.observations = MakeLineFixture(inlier_count, outlier_count);
    const auto estimate = [&run](const std::vector<std::size_t>& indices,
                                 std::vector<LineModel>* models) {
        EstimateLineFromPair(run.observations, indices, models);
    };
    const auto local_estimate = [&run](const std::vector<std::size_t>& indices,
                                       std::vector<LineModel>* models) {
        RefitLineLeastSquares(run.observations, indices, models);
    };
    const auto residual = [&run](const LineModel& model,
                                 const std::size_t index) {
        const double error = run.observations[index].y -
                             (model.slope * run.observations[index].x +
                              model.intercept);
        return error * error;
    };
    run.report = RunLORansac<LineModel>(
            run.observations.size(), 2, 2, options, estimate, local_estimate,
            residual);
    return run;
}

void TestLORansacDeterministicMaskAndLocalRefit() {
    RansacOptions options;
    options.max_error = 0.3;
    options.min_inlier_ratio = 0.1;
    options.confidence = 0.999;
    options.dyn_num_trials_multiplier = 3.0;
    options.min_num_trials = 20;
    options.max_num_trials = 200;
    options.random_seed = 20261118;

    const LineRun first = RunLineFixture(90, 30, options);
    const LineRun repeated = RunLineFixture(90, 30, options);
    Require(first.report.success, "synthetic LO-RANSAC fixture did not succeed");
    Require(first.report.support.num_inliers == 90,
            "local refit did not recover all synthetic line inliers");
    Require(first.report.inlier_mask.size() == 120,
            "LO-RANSAC returned an inlier mask with the wrong length");
    for (std::size_t i = 0; i < first.report.inlier_mask.size(); ++i) {
        Require(first.report.inlier_mask[i] == (i < 90),
                "LO-RANSAC mask does not preserve original observation order");
    }
    Require(first.report.local_optimization_trials > 0 &&
                    first.report.accepted_local_refits > 0 &&
                    first.report.best_model_is_local,
            "LO-RANSAC did not accept a local refit in the synthetic fixture");
    Require(first.report.num_trials <=
                    static_cast<std::size_t>(options.max_num_trials),
            "LO-RANSAC exceeded the configured practical trial bound");
    Require(repeated.report.success &&
                    repeated.report.num_trials == first.report.num_trials &&
                    repeated.report.inlier_mask == first.report.inlier_mask &&
                    repeated.report.model->slope == first.report.model->slope &&
                    repeated.report.model->intercept == first.report.model->intercept,
            "fixed-seed LO-RANSAC changed its synthetic result across runs");

    constexpr std::size_t inliers = 800;
    constexpr std::size_t samples = 1000;
    constexpr std::size_t sample_size = 5;
    double probability_inlier = 1.0;
    for (std::size_t i = 0; i < sample_size; ++i) {
        probability_inlier *= static_cast<double>(inliers - i) /
                              static_cast<double>(samples - i);
    }
    const std::size_t expected_trials = static_cast<std::size_t>(std::ceil(
            std::log(1.0 - 0.99) /
            std::log(1.0 - probability_inlier) * 3.0));
    Require(ComputeNumTrials(inliers, samples, sample_size, 0.99, 3.0) ==
                    expected_trials,
            "adaptive RANSAC trials differ from COLMAP's finite-population formula");
    Require(ComputeNumTrials(samples, samples, sample_size, 0.99, 3.0) == 1,
            "all-inlier minimal samples should stop after one trial");
    Require(ComputeNumTrials(sample_size - 1, samples, sample_size, 0.99, 3.0) ==
                    std::numeric_limits<std::size_t>::max(),
            "insufficient inliers should retain the hard trial bound");
}

void TestEssentialPoseHypothesesAndCheirality() {
    const Eigen::Vector3d axis(0.2, 0.6, 0.3);
    const Eigen::Matrix3d expected_rotation =
            Eigen::AngleAxisd(0.23, axis.normalized()).toRotationMatrix();
    const Eigen::Vector3d expected_translation =
            Eigen::Vector3d(0.9, -0.2, 0.15).normalized();
    const Eigen::Matrix3d essential = Skew(expected_translation) * expected_rotation;

    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    for (int i = 0; i < 32; ++i) {
        const double x = -1.2 + 0.075 * i;
        const double y = -0.55 + 0.11 * (i % 11);
        const double z = 4.5 + 0.17 * (i % 13);
        const Eigen::Vector3d point1(x, y, z);
        const Eigen::Vector3d point2 =
                expected_rotation * point1 + expected_translation;
        rays1.push_back(point1.normalized());
        rays2.push_back(point2.normalized());
    }
    const auto hypotheses =
            RecoverEssentialPoseHypotheses(essential, rays1, rays2);
    Require(hypotheses.has_value() && hypotheses->candidates.size() == 4,
            "COLMAP essential decomposition did not produce four pose hypotheses");
    const auto& selected = hypotheses->candidates[hypotheses->selected_index];
    Require(selected.positive_depth_count == rays1.size(),
            "cheirality did not select the fully positive-depth pose hypothesis");
    Require((selected.rotation - expected_rotation).norm() < 1e-8,
            "selected pose hypothesis has the wrong relative rotation");
    Require(selected.translation.dot(expected_translation) > 1.0 - 1e-8,
            "selected pose hypothesis has the wrong translation direction");
    std::size_t maximum_positive_depth_count = 0;
    for (const auto& candidate : hypotheses->candidates) {
        maximum_positive_depth_count =
                std::max(maximum_positive_depth_count,
                         candidate.positive_depth_count);
    }
    Require(selected.positive_depth_count == maximum_positive_depth_count,
            "selected essential pose is not the maximum-cheirality hypothesis");
}

void RunBenchmarkCase(const std::size_t observation_count) {
    RansacOptions options;
    options.max_error = 0.3;
    options.min_inlier_ratio = 0.01;
    options.confidence = 0.999;
    options.dyn_num_trials_multiplier = 3.0;
    options.min_num_trials = 32;
    options.max_num_trials = 128;
    options.random_seed = 20261118;
    const std::size_t inlier_count = observation_count * 4 / 5;
    const auto start = std::chrono::steady_clock::now();
    const LineRun run = RunLineFixture(inlier_count,
                                       observation_count - inlier_count,
                                       options);
    const auto stop = std::chrono::steady_clock::now();
    const double milliseconds =
            std::chrono::duration<double, std::milli>(stop - start).count();
    Require(run.report.success && run.report.num_trials <= 128,
            "bounded benchmark case exceeded its success or trial limits");
    std::cout << "bounded_loransac observations=" << observation_count
              << " trials=" << run.report.num_trials
              << " inliers=" << run.report.support.num_inliers
              << " elapsed_ms=" << milliseconds << '\n';
}

struct FivePointRayFixture {
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

FivePointRayFixture MakeFivePointRayFixture(const std::size_t count) {
    FivePointRayFixture fixture;
    const Eigen::Vector3d axis(0.2, 0.6, 0.3);
    fixture.rotation =
            Eigen::AngleAxisd(0.23, axis.normalized()).toRotationMatrix();
    fixture.translation = Eigen::Vector3d(0.9, -0.2, 0.15).normalized();
    fixture.rays1.reserve(count);
    fixture.rays2.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double x = -1.3 + 0.37 * i + 0.09 * std::sin(0.7 * i);
        const double y = -0.8 + 0.23 * (i % 4) + 0.04 * i;
        const double z = 4.1 + 0.31 * (i % 5) + 0.11 * i;
        const Eigen::Vector3d ray1(x, y, z);
        fixture.rays1.push_back(ray1);
        fixture.rays2.push_back(
                (fixture.rotation * ray1 + fixture.translation).normalized());
    }
    return fixture;
}

template <std::size_t CandidateCount, std::size_t ObservationCount>
void CheckFivePointOracleFixture(
        const FivePointRayFixture& fixture,
        const double (&expected_matrices)[CandidateCount][9],
        const double (&expected_residuals)[CandidateCount][ObservationCount]) {
    Require(fixture.rays1.size() == ObservationCount &&
                    fixture.rays2.size() == ObservationCount,
            "five-point fixture observation count changed");
    std::vector<Eigen::Matrix3d> models;
    tracking::colmap_initial_pair_test::EstimateEssentialFivePoint(
            fixture.rays1, fixture.rays2, &models);
    Require(models.size() == CandidateCount,
            "five-point candidate count differs from pinned COLMAP");

    const Eigen::Matrix3d expected_pose_essential =
            Skew(fixture.translation) * fixture.rotation;
    const Eigen::Matrix3d normalized_expected_pose_essential =
            expected_pose_essential.normalized();
    bool found_pose_solution = false;
    for (std::size_t candidate = 0; candidate < CandidateCount; ++candidate) {
        Eigen::Matrix3d expected_matrix;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                expected_matrix(row, column) =
                        expected_matrices[candidate][row * 3 + column];
            }
        }
        Require(models[candidate].allFinite() &&
                        std::abs(models[candidate].norm() - 1.0) < 1e-12,
                "five-point candidate is non-finite or not normalized");
        const double candidate_matrix_error = std::min(
                (models[candidate] - expected_matrix).norm(),
                (models[candidate] + expected_matrix).norm());
        Require(candidate_matrix_error < 1e-8,
                "ordered five-point candidate differs from pinned COLMAP output");
        if (std::min((models[candidate] - normalized_expected_pose_essential).norm(),
                     (models[candidate] + normalized_expected_pose_essential).norm()) <
            1e-8) {
            found_pose_solution = true;
        }
        for (std::size_t observation = 0; observation < ObservationCount;
             ++observation) {
            const double actual_residual = SquaredSampsonError(
                    fixture.rays1[observation], fixture.rays2[observation],
                    models[candidate]);
            const double expected_residual =
                    expected_residuals[candidate][observation];
            const double tolerance =
                    2e-25 + 2e-8 * std::abs(expected_residual);
            Require(std::isfinite(actual_residual) &&
                            std::abs(actual_residual - expected_residual) <= tolerance,
                    "five-point candidate residual differs from pinned COLMAP output");
        }
    }
    Require(found_pose_solution,
            "five-point candidates do not include the known camera-pose essential matrix");
}

void TestFivePointSolverCandidatesAndResiduals() {
    const FivePointRayFixture minimal_fixture = MakeFivePointRayFixture(5);
    CheckFivePointOracleFixture(
            minimal_fixture,
            five_point_oracle_fixtures::kCandidateMatrices5,
            five_point_oracle_fixtures::kSquaredResiduals5);
    const FivePointRayFixture overdetermined_fixture = MakeFivePointRayFixture(8);
    CheckFivePointOracleFixture(
            overdetermined_fixture,
            five_point_oracle_fixtures::kCandidateMatrices8,
            five_point_oracle_fixtures::kSquaredResiduals8);
}

void TestFivePointSolverInvalidInputsAndDegeneracy() {
    const FivePointRayFixture fixture = MakeFivePointRayFixture(5);
    std::vector<Eigen::Matrix3d> models;
    bool threw = false;
    try {
        tracking::colmap_initial_pair_test::EstimateEssentialFivePoint(
                fixture.rays1, std::vector<Eigen::Vector3d>(4), &models);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Require(threw, "five-point solver did not reject mismatched ray vectors");

    threw = false;
    try {
        tracking::colmap_initial_pair_test::EstimateEssentialFivePoint(
                std::vector<Eigen::Vector3d>(4),
                std::vector<Eigen::Vector3d>(4), &models);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Require(threw, "five-point solver did not reject fewer than five rays");

    threw = false;
    try {
        tracking::colmap_initial_pair_test::EstimateEssentialFivePoint(
                fixture.rays1, fixture.rays2, nullptr);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Require(threw, "five-point solver did not reject a null output vector");

    const std::vector<Eigen::Vector3d> repeated_ray1(
            5, Eigen::Vector3d(0.1, 0.2, 1.0));
    const std::vector<Eigen::Vector3d> repeated_ray2(
            5, Eigen::Vector3d(-0.1, 0.05, 1.0));
    models.push_back(Eigen::Matrix3d::Identity());
    tracking::colmap_initial_pair_test::EstimateEssentialFivePoint(
            repeated_ray1, repeated_ray2, &models);
    Require(models.empty(),
            "degenerate repeated-ray fixture differs from pinned COLMAP no-candidate result");
}

void RunFivePointHostBenchmark(const std::size_t observation_count,
                               const int repetitions) {
    const FivePointRayFixture fixture = MakeFivePointRayFixture(observation_count);
    std::vector<Eigen::Matrix3d> models;
    const auto start = std::chrono::steady_clock::now();
    for (int run = 0; run < repetitions; ++run) {
        tracking::colmap_initial_pair_test::EstimateEssentialFivePoint(
                fixture.rays1, fixture.rays2, &models);
    }
    const auto stop = std::chrono::steady_clock::now();
    const double milliseconds =
            std::chrono::duration<double, std::milli>(stop - start).count();
    Require(!models.empty() && observation_count <= 8 && repetitions <= 16,
            "bounded five-point host benchmark did not complete its small fixture");
    std::cout << "host_bounded_five_point_solver observations="
              << observation_count << " repetitions=" << repetitions
              << " candidates=" << models.size()
              << " host_elapsed_ms=" << milliseconds << '\n';
}

void RunAllTests() {
    TestSquaredSampsonResidualAndPixelPathDivergence();
    TestSquaredSampsonSupportMaskAndRanking();
    TestLORansacDeterministicMaskAndLocalRefit();
    TestEssentialPoseHypothesesAndCheirality();
    TestFivePointSolverCandidatesAndResiduals();
    TestFivePointSolverInvalidInputsAndDegeneracy();
}

}  // namespace

int main(const int argc, char** argv) {
    try {
        RunAllTests();
        if (argc == 2 && std::string(argv[1]) == "--benchmark") {
            RunBenchmarkCase(1024);
            RunBenchmarkCase(4096);
            RunFivePointHostBenchmark(5, 8);
            RunFivePointHostBenchmark(8, 8);
        }
        std::cout << "COLMAP initial-pair differential fixtures passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "COLMAP initial-pair differential failure: "
                  << error.what() << '\n';
        return 1;
    }
}
