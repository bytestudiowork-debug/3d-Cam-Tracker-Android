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
#include "fundamental_matrix_solver_differential.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Core>
#include "fundamental_matrix_fixture_inputs.h"
#include "fundamental_matrix_oracle_expectations.h"
namespace {
using fundamental_matrix_oracle_expectations::CandidateFixture;
using fundamental_matrix_oracle_expectations::SolverFixture;
using tracking::fundamental_solver_test_fixtures::Correspondences;
using tracking::fundamental_solver_test_fixtures::Geometry;
using tracking::fundamental_solver_test_fixtures::MakeCorrespondences;
void Require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
template <typename Operation>
void RequireThrows(const Operation& operation, const std::string& message) {
    bool threw = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Require(threw, message);
}
void VerifyFixture(const SolverFixture& fixture,
                   const Correspondences& points,
                   const bool seven_point) {
    Require(points.points1.size() == fixture.observation_count &&
                    points.points2.size() == fixture.observation_count,
            std::string(fixture.name) + " input observation count changed");
    std::vector<Eigen::Matrix3d> models;
    if (seven_point) {
        tracking::fundamental_solver_test::EstimateFundamentalSevenPoint(
                points.points1, points.points2, &models);
    } else {
        tracking::fundamental_solver_test::EstimateFundamentalEightPoint(
                points.points1, points.points2, &models);
    }
    Require(models.size() == fixture.candidates.size(),
            std::string(fixture.name) + " candidate count/order changed");
    for (std::size_t candidate_index = 0;
         candidate_index < fixture.candidates.size(); ++candidate_index) {
        const CandidateFixture& expected = fixture.candidates[candidate_index];
        const Eigen::Matrix3d& actual = models[candidate_index];
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                const double value = expected.matrix[3 * row + col];
                const double tolerance = 2e-10 + 2e-9 * std::abs(value);
                Require(std::abs(actual(row, col) - value) <= tolerance,
                        std::string(fixture.name) + " candidate matrix differs");
            }
        }
        std::vector<double> residuals;
        tracking::fundamental_solver_test::FundamentalMatrixResiduals(
                points.points1, points.points2, actual, &residuals);
        Require(residuals.size() == expected.residuals.size(),
                std::string(fixture.name) + " residual count changed");
        for (std::size_t observation = 0;
             observation < expected.residuals.size(); ++observation) {
            const double expected_residual = expected.residuals[observation];
            const double tolerance = 1e-20 +
                    2e-10 * std::abs(expected_residual);
            Require(std::abs(residuals[observation] - expected_residual) <=
                            tolerance,
                    std::string(fixture.name) + " residual differs at observation " +
                            std::to_string(observation));
        }
    }
}
void TestPinnedFixtures() {
    using namespace fundamental_matrix_oracle_expectations;
    VerifyFixture(kSevenGeneral,
                  MakeCorrespondences(7, Geometry::kGeneral), true);
    VerifyFixture(kSevenPlanar,
                  MakeCorrespondences(7, Geometry::kPlanar), true);
    VerifyFixture(kSevenCollinear,
                  MakeCorrespondences(7, Geometry::kCollinear), true);
    VerifyFixture(kEightMinimal,
                  MakeCorrespondences(8, Geometry::kGeneral), false);
    VerifyFixture(kEightNoisyOverdetermined,
                  MakeCorrespondences(13, Geometry::kGeneral, 0.35), false);
    VerifyFixture(kEightPlanar,
                  MakeCorrespondences(8, Geometry::kPlanar), false);
    VerifyFixture(kEightCollinearOverdetermined,
                  MakeCorrespondences(10, Geometry::kCollinear), false);
}
void TestInvalidInputSemantics() {
    const Eigen::Matrix3d marker = Eigen::Matrix3d::Constant(42.0);
    std::vector<Eigen::Matrix3d> models{marker};
    auto seven = MakeCorrespondences(7, Geometry::kGeneral);
    auto six = MakeCorrespondences(6, Geometry::kGeneral);
    RequireThrows([&] {
        tracking::fundamental_solver_test::EstimateFundamentalSevenPoint(
                six.points1, seven.points2, &models);
    }, "seven-point accepted a non-minimal sample");
    Require(models.size() == 1 && (models[0] - marker).cwiseAbs().maxCoeff() == 0.0,
            "seven-point changed output before rejecting invalid input");
    RequireThrows([&] {
        tracking::fundamental_solver_test::EstimateFundamentalSevenPoint(
                seven.points1, seven.points2, nullptr);
    }, "seven-point accepted a null model output");
    auto eight = MakeCorrespondences(8, Geometry::kGeneral);
    auto seven2 = MakeCorrespondences(7, Geometry::kGeneral);
    RequireThrows([&] {
        tracking::fundamental_solver_test::EstimateFundamentalEightPoint(
                eight.points1, seven2.points2, &models);
    }, "eight-point accepted unequal input lengths");
    Require(models.size() == 1 && (models[0] - marker).cwiseAbs().maxCoeff() == 0.0,
            "eight-point changed output before rejecting mismatched input");
    RequireThrows([&] {
        tracking::fundamental_solver_test::EstimateFundamentalEightPoint(
                seven.points1, seven.points2, &models);
    }, "eight-point accepted fewer than eight correspondences");
    RequireThrows([&] {
        tracking::fundamental_solver_test::EstimateFundamentalEightPoint(
                eight.points1, eight.points2, nullptr);
    }, "eight-point accepted a null model output");
    std::vector<double> residuals{42.0};
    RequireThrows([&] {
        tracking::fundamental_solver_test::FundamentalMatrixResiduals(
                eight.points1, seven2.points2, marker, &residuals);
    }, "residual computation accepted unequal point counts");
    Require(residuals.size() == 1 && residuals[0] == 42.0,
            "residual output changed before rejecting mismatched input");
    RequireThrows([&] {
        tracking::fundamental_solver_test::FundamentalMatrixResiduals(
                eight.points1, eight.points2, marker, nullptr);
    }, "residual computation accepted a null output");
}
}  // namespace
int main() {
    try {
        TestPinnedFixtures();
        TestInvalidInputSemantics();
        std::cout << "Fundamental-matrix differential checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fundamental-matrix differential failure: " << error.what()
                  << '\n';
        return 1;
    }
}
