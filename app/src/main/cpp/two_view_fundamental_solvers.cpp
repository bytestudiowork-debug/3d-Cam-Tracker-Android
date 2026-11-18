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
#include "two_view_epipolar_estimator_core.h"
// Source adaptation from pinned fundamental-matrix implementation at c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>
namespace tracking::source_two_view {
namespace {
void RequireInput(const bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}
Eigen::Vector2d NormalizeImagePoint(const Eigen::Vector2d& point,
                                    const Eigen::Matrix3d& transform) {
    return (transform * point.homogeneous()).hnormalized();
}
void CenterAndNormalizeImagePoints(
        const std::vector<Eigen::Vector2d>& points,
        std::vector<Eigen::Vector2d>* normalized_points,
        Eigen::Matrix3d* normalized_from_original) {
    const std::size_t num_points = points.size();
    RequireInput(num_points > 0, "point normalization requires at least one point");
    Eigen::Vector2d centroid(0.0, 0.0);
    for (const Eigen::Vector2d& point : points) {
        centroid += point;
    }
    centroid /= static_cast<double>(num_points);
    double rms_mean_dist = 0.0;
    for (const Eigen::Vector2d& point : points) {
        rms_mean_dist += (point - centroid).squaredNorm();
    }
    rms_mean_dist = std::sqrt(rms_mean_dist / static_cast<double>(num_points));
    const double norm_factor = std::sqrt(2.0) / rms_mean_dist;
    *normalized_from_original << norm_factor, 0.0, -norm_factor * centroid(0),
            0.0, norm_factor, -norm_factor * centroid(1),
            0.0, 0.0, 1.0;
    normalized_points->resize(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        (*normalized_points)[i] = NormalizeImagePoint(
                points[i], *normalized_from_original);
    }
}
int FindCubicPolynomialRoots(const double c2,
                             const double c1,
                             const double c0,
                             Eigen::Vector3d* real_roots) {
    constexpr double k2PiOver3 = 2.09439510239319526263557236234192;
    constexpr double k4PiOver3 = 4.18879020478639052527114472468384;
    const double c2_over_3 = c2 / 3.0;
    const double a = c1 - c2 * c2_over_3;
    double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double c = b * b / 4.0 + a * a * a / 27.0;
    int num_roots = 0;
    if (c > 0.0) {
        c = std::sqrt(c);
        b *= -0.5;
        (*real_roots)[0] = std::cbrt(b + c) + std::cbrt(b - c) - c2_over_3;
        num_roots = 1;
    } else {
        c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
        const double d = 2.0 * std::sqrt(-a / 3.0);
        const double acos_over_3 = std::acos(c) / 3.0;
        (*real_roots)[0] = d * std::cos(acos_over_3) - c2_over_3;
        (*real_roots)[1] = d * std::cos(acos_over_3 - k2PiOver3) - c2_over_3;
        (*real_roots)[2] = d * std::cos(acos_over_3 - k4PiOver3) - c2_over_3;
        num_roots = 3;
    }
    for (int i = 0; i < num_roots; ++i) {
        const double x = (*real_roots)[i];
        const double x2 = x * x;
        const double x3 = x * x2;
        const double dx = -(x3 + c2 * x2 + c1 * x + c0) /
                (3.0 * x2 + 2.0 * c2 * x + c1);
        (*real_roots)[i] += dx;
    }
    return num_roots;
}
}  // namespace
void EstimateFundamentalSevenPoint(
        const std::vector<Eigen::Vector2d>& points1,
        const std::vector<Eigen::Vector2d>& points2,
        std::vector<Eigen::Matrix3d>* models) {
    RequireInput(points1.size() == 7,
                 "seven-point estimation requires exactly seven first points");
    RequireInput(points2.size() == 7,
                 "seven-point estimation requires exactly seven second points");
    RequireInput(models != nullptr, "seven-point model output must not be null");
    models->clear();
    Eigen::Matrix<double, 9, 7> A;
    for (std::size_t i = 0; i < 7; ++i) {
        A.col(i) << points1[i].x() * points2[i].homogeneous(),
                    points1[i].y() * points2[i].homogeneous(),
                    points2[i].homogeneous();
    }
    const Eigen::Matrix<double, 9, 9> Q =
            A.fullPivHouseholderQr().matrixQ();
    Eigen::Matrix<double, 9, 1> f1 = Q.col(7);
    const Eigen::Matrix<double, 9, 1> f2 = Q.col(8);
    f1 -= f2;
    const double t0 = f1(4) * f1(8) - f1(5) * f1(7);
    const double t1 = f1(3) * f1(8) - f1(5) * f1(6);
    const double t2 = f1(3) * f1(7) - f1(4) * f1(6);
    const double t3 = f2(4) * f2(8) - f2(5) * f2(7);
    const double t4 = f2(3) * f2(8) - f2(5) * f2(6);
    const double t5 = f2(3) * f2(7) - f2(4) * f2(6);
    Eigen::Vector4d coeffs;
    coeffs(0) = f1(0) * t0 - f1(1) * t1 + f1(2) * t2;
    if (std::abs(coeffs(0)) < 1e-16) {
        return;
    }
    coeffs(1) = f2(0) * t0 - f2(1) * t1 + f2(2) * t2 -
            f2(3) * (f1(1) * f1(8) - f1(2) * f1(7)) +
            f2(4) * (f1(0) * f1(8) - f1(2) * f1(6)) -
            f2(5) * (f1(0) * f1(7) - f1(1) * f1(6)) +
            f2(6) * (f1(1) * f1(5) - f1(2) * f1(4)) -
            f2(7) * (f1(0) * f1(5) - f1(2) * f1(3)) +
            f2(8) * (f1(0) * f1(4) - f1(1) * f1(3));
    coeffs(2) = f1(0) * t3 - f1(1) * t4 + f1(2) * t5 -
            f1(3) * (f2(1) * f2(8) - f2(2) * f2(7)) +
            f1(4) * (f2(0) * f2(8) - f2(2) * f2(6)) -
            f1(5) * (f2(0) * f2(7) - f2(1) * f2(6)) +
            f1(6) * (f2(1) * f2(5) - f2(2) * f2(4)) -
            f1(7) * (f2(0) * f2(5) - f2(2) * f2(3)) +
            f1(8) * (f2(0) * f2(4) - f2(1) * f2(3));
    coeffs(3) = f2(0) * t3 - f2(1) * t4 + f2(2) * t5;
    coeffs.tail<3>() /= coeffs(0);
    Eigen::Vector3d roots;
    const int num_roots = FindCubicPolynomialRoots(
            coeffs(1), coeffs(2), coeffs(3), &roots);
    models->reserve(static_cast<std::size_t>(num_roots));
    for (int i = 0; i < num_roots; ++i) {
        const Eigen::Matrix<double, 9, 1> F =
                (f1 * roots[i] + f2).normalized();
        models->push_back(Eigen::Map<const Eigen::Matrix3d>(F.data()));
    }
}
void EstimateFundamentalEightPoint(
        const std::vector<Eigen::Vector2d>& points1,
        const std::vector<Eigen::Vector2d>& points2,
        std::vector<Eigen::Matrix3d>* models) {
    RequireInput(points1.size() == points2.size(),
                 "eight-point estimation requires equal point counts");
    RequireInput(points1.size() >= 8,
                 "eight-point estimation requires at least eight points");
    RequireInput(models != nullptr, "eight-point model output must not be null");
    models->clear();
    std::vector<Eigen::Vector2d> normalized_points1;
    std::vector<Eigen::Vector2d> normalized_points2;
    Eigen::Matrix3d normalized_from_original1;
    Eigen::Matrix3d normalized_from_original2;
    CenterAndNormalizeImagePoints(
            points1, &normalized_points1, &normalized_from_original1);
    CenterAndNormalizeImagePoints(
            points2, &normalized_points2, &normalized_from_original2);
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(
            static_cast<Eigen::Index>(points1.size()), 9);
    for (std::size_t i = 0; i < points1.size(); ++i) {
        A.row(static_cast<Eigen::Index>(i)) <<
                normalized_points2[i].x() *
                        normalized_points1[i].transpose().homogeneous(),
                normalized_points2[i].y() *
                        normalized_points1[i].transpose().homogeneous(),
                normalized_points1[i].transpose().homogeneous();
    }
    Eigen::Matrix3d Q;
    if (points1.size() == 8) {
        const Eigen::MatrixXd QQ =
                A.transpose().householderQr().householderQ();
        Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
                QQ.col(8).data());
    } else {
        const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
                A, Eigen::ComputeFullV);
        Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
                svd.matrixV().col(8).data());
    }
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = svd.singularValues();
    singular_values(2) = 0.0;
    const Eigen::Matrix3d F =
            svd.matrixU() * singular_values.asDiagonal() *
            svd.matrixV().transpose();
    models->resize(1);
    (*models)[0] = normalized_from_original2.transpose() * F *
                   normalized_from_original1;
}
void FundamentalMatrixResiduals(
        const std::vector<Eigen::Vector2d>& points1,
        const std::vector<Eigen::Vector2d>& points2,
        const Eigen::Matrix3d& fundamental,
        std::vector<double>* residuals) {
    RequireInput(points1.size() == points2.size(),
                 "residual computation requires equal point counts");
    RequireInput(residuals != nullptr, "residual output must not be null");
    residuals->resize(points1.size());
    for (std::size_t i = 0; i < points1.size(); ++i) {
        const Eigen::Vector3d ray1 = points1[i].homogeneous();
        const Eigen::Vector3d ray2 = points2[i].homogeneous();
        const Eigen::Vector3d epipolar_line1 = fundamental * ray1;
        const double numerator = ray2.dot(epipolar_line1);
        const Eigen::Vector4d denominator(
                ray2.dot(fundamental.col(0)),
                ray2.dot(fundamental.col(1)),
                epipolar_line1.x(), epipolar_line1.y());
        const double denominator_squared_norm = denominator.squaredNorm();
        if (denominator_squared_norm == 0.0) {
            (*residuals)[i] = std::numeric_limits<double>::max();
        } else {
            (*residuals)[i] = numerator * numerator /
                              denominator_squared_norm;
        }
    }
}
}  // namespace tracking::source_two_view
