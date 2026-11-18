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

#include "colmap_initial_pair_differential.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace tracking::homography_test {

using tracking::colmap_initial_pair_test::LORansacReport;
using tracking::colmap_initial_pair_test::RansacOptions;
using tracking::colmap_initial_pair_test::RunLORansac;

using Point = Eigen::Vector2d;
using Ray = Eigen::Vector3d;
using Matrix = Eigen::Matrix3d;

struct HomographyPoseCandidate {
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Ray translation = Ray::Zero();
    Ray normal = Ray::Zero();
    std::size_t positive_depth_count = 0;
    double reprojection_residual_sum = 0.0;
    std::vector<char> positive_depth_mask;
};

struct HomographyPoseReport {
    std::vector<HomographyPoseCandidate> candidates;
    std::size_t selected_index = 0;
    std::vector<Ray> triangulated_points;
    double median_triangulation_angle = 0.0;
};

struct InitialPairGates {
    std::size_t minimum_inliers = 100;
    double maximum_absolute_z_translation = 0.95;
    double minimum_median_triangulation_angle_degrees = 16.0;
};

bool EstimateDLT(const std::vector<Point>& points1,
                 const std::vector<Point>& points2,
                 std::vector<Matrix>* models);

void OneWaySquaredReprojectionResiduals(const std::vector<Point>& points1,
                                        const std::vector<Point>& points2,
                                        const Matrix& homography,
                                        std::vector<double>* residuals);

LORansacReport<Matrix> RunHomographyLORansac(
        const std::vector<Point>& points1,
        const std::vector<Point>& points2,
        const RansacOptions& options);

std::vector<HomographyPoseCandidate> DecomposeHomography(
        const Matrix& homography,
        const Matrix& intrinsics1,
        const Matrix& intrinsics2);

HomographyPoseReport RecoverHomographyPose(
        const Matrix& homography,
        const Matrix& intrinsics1,
        const Matrix& intrinsics2,
        const std::vector<Ray>& rays1,
        const std::vector<Ray>& rays2);

double CalculateTriangulationAngle(const Ray& projection_center1,
                                   const Ray& projection_center2,
                                   const Ray& point3d);

double Median(std::vector<double> values);

bool PassInitialPairGates(std::size_t inlier_count,
                          const HomographyPoseReport& pose,
                          const InitialPairGates& gates = {});

}  // namespace tracking::homography_test
