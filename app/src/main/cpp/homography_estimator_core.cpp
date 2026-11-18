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
#include "homography_estimator_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <Eigen/LU>
#include <Eigen/SVD>

namespace tracking::source_homography {
namespace {

double OneWaySquaredResidual(const Point& point1,
                            const Point& point2,
                            const Matrix& H) {
    const double pd0 = H(0, 0) * point1.x() + H(0, 1) * point1.y() + H(0, 2);
    const double pd1 = H(1, 0) * point1.x() + H(1, 1) * point1.y() + H(1, 2);
    const double pd2 = H(2, 0) * point1.x() + H(2, 1) * point1.y() + H(2, 2);
    const double inv_pd2 = 1.0 / pd2;
    const double dd0 = point2.x() - pd0 * inv_pd2;
    const double dd1 = point2.y() - pd1 * inv_pd2;
    return dd0 * dd0 + dd1 * dd1;
}

void Gather(const std::vector<Point>& points,
            const std::vector<std::size_t>& indices,
            std::vector<Point>* selected) {
    selected->clear();
    selected->reserve(indices.size());
    for (const std::size_t index : indices) {
        selected->push_back(points[index]);
    }
}

double ComputeOppositeOfMinor(const Matrix& matrix,
                              const int row,
                              const int col) {
    const int col1 = col == 0 ? 1 : 0;
    const int col2 = col == 2 ? 1 : 2;
    const int row1 = row == 0 ? 1 : 0;
    const int row2 = row == 2 ? 1 : 2;
    return matrix(row1, col2) * matrix(row2, col1) -
           matrix(row1, col1) * matrix(row2, col2);
}

Matrix ComputeHomographyRotation(const Matrix& H_normalized,
                                 const Ray& tstar,
                                 const Ray& normal,
                                 const double v) {
    return H_normalized *
           (Matrix::Identity() - (2.0 / v) * tstar * normal.transpose());
}

bool TriangulateMidpoint(const HomographyPoseCandidate& pose,
                         const Ray& cam_ray1,
                         const Ray& cam_ray2,
                         Ray* point3d_in_cam1) {
    const Eigen::Quaterniond cam1_from_cam2_rotation = pose.rotation.inverse();
    const Ray cam_ray2_in_cam1 = cam1_from_cam2_rotation * cam_ray2;
    const Ray cam2_in_cam1 = cam1_from_cam2_rotation * -pose.translation;

    Eigen::Matrix3d system;
    system << cam_ray1(0), -cam_ray2_in_cam1(0), -cam2_in_cam1(0),
              cam_ray1(1), -cam_ray2_in_cam1(1), -cam2_in_cam1(1),
              cam_ray1(2), -cam_ray2_in_cam1(2), -cam2_in_cam1(2);
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(system, Eigen::ComputeFullV);
    if (svd.matrixV()(2, 2) == 0.0) {
        return false;
    }
    const Eigen::Vector2d lambda = svd.matrixV().col(2).hnormalized();
    if (lambda(0) <= std::numeric_limits<double>::epsilon() ||
        lambda(1) <= std::numeric_limits<double>::epsilon()) {
        return false;
    }
    *point3d_in_cam1 = 0.5 * (lambda(0) * cam_ray1 + cam2_in_cam1 +
                              lambda(1) * cam_ray2_in_cam1);
    return true;
}

void EvaluatePoseCandidate(HomographyPoseCandidate* pose,
                           const std::vector<Ray>& rays1,
                           const std::vector<Ray>& rays2,
                           std::vector<Ray>* points3d) {
    pose->positive_depth_mask.assign(rays1.size(), 0);
    pose->positive_depth_count = 0;
    pose->reprojection_residual_sum = 0.0;
    points3d->clear();
    for (std::size_t i = 0; i < rays1.size(); ++i) {
        Ray point3d_in_cam1;
        if (!TriangulateMidpoint(*pose, rays1[i], rays2[i], &point3d_in_cam1)) {
            continue;
        }
        pose->positive_depth_mask[i] = 1;
        ++pose->positive_depth_count;
        const Ray point3d_in_cam2 = pose->rotation * point3d_in_cam1 +
                                    pose->translation;
        const double error1 = 1.0 - std::clamp(
                rays1[i].dot(point3d_in_cam1.normalized()), -1.0, 1.0);
        const double error2 = 1.0 - std::clamp(
                rays2[i].dot(point3d_in_cam2.normalized()), -1.0, 1.0);
        pose->reprojection_residual_sum += error1 + error2;
        points3d->push_back(point3d_in_cam1);
    }
}

}  // namespace

bool EstimateDLT(const std::vector<Point>& points1,
                 const std::vector<Point>& points2,
                 std::vector<Matrix>* models) {
    if (points1.size() != points2.size() || points1.size() < 4 ||
        models == nullptr) {
        throw std::invalid_argument("invalid homography DLT input");
    }
    models->clear();
    const std::size_t num_points = points1.size();
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(2 * num_points, 9);
    for (std::size_t i = 0; i < num_points; ++i) {
        A.block<1, 3>(2 * i, 0) = points1[i].transpose().homogeneous();
        A.block<1, 3>(2 * i, 3).setZero();
        A.block<1, 3>(2 * i, 6) =
                -points2[i].x() * points1[i].transpose().homogeneous();
        A.block<1, 3>(2 * i + 1, 0).setZero();
        A.block<1, 3>(2 * i + 1, 3) = points1[i].transpose().homogeneous();
        A.block<1, 3>(2 * i + 1, 6) =
                -points2[i].y() * points1[i].transpose().homogeneous();
    }

    Matrix H;
    if (num_points == 4) {
        const Eigen::Matrix<double, 9, 1> h =
                A.block<8, 8>(0, 0).partialPivLu()
                        .solve(-A.block<8, 1>(0, 8))
                        .homogeneous();
        if (h.hasNaN()) {
            return false;
        }
        H = Eigen::Map<const Matrix>(h.data()).transpose();
    } else {
        const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
                A, Eigen::ComputeFullV);
        if (svd.rank() < 8) {
            return false;
        }
        const Eigen::VectorXd nullspace = svd.matrixV().col(8);
        H = Eigen::Map<const Matrix>(nullspace.data()).transpose();
    }
    if (std::abs(H.determinant()) < 1e-8) {
        return false;
    }
    models->push_back(H);
    return true;
}

void OneWaySquaredReprojectionResiduals(const std::vector<Point>& points1,
                                        const std::vector<Point>& points2,
                                        const Matrix& homography,
                                        std::vector<double>* residuals) {
    if (points1.size() != points2.size() || residuals == nullptr) {
        throw std::invalid_argument("invalid homography residual input");
    }
    residuals->resize(points1.size());
    for (std::size_t i = 0; i < points1.size(); ++i) {
        (*residuals)[i] = OneWaySquaredResidual(points1[i], points2[i], homography);
    }
}

LORansacReport<Matrix> RunHomographyLORansac(
        const std::vector<Point>& points1,
        const std::vector<Point>& points2,
        const RansacOptions& options) {
    if (points1.size() != points2.size()) {
        throw std::invalid_argument("homography LO-RANSAC input sizes differ");
    }
    const auto estimate = [&points1, &points2](
                                  const std::vector<std::size_t>& indices,
                                  std::vector<Matrix>* models) {
        std::vector<Point> selected1;
        std::vector<Point> selected2;
        Gather(points1, indices, &selected1);
        Gather(points2, indices, &selected2);
        EstimateDLT(selected1, selected2, models);
    };
    const auto local_estimate = [&points1, &points2](
                                        const std::vector<std::size_t>& indices,
                                        std::vector<Matrix>* models) {
        std::vector<Point> selected1;
        std::vector<Point> selected2;
        Gather(points1, indices, &selected1);
        Gather(points2, indices, &selected2);
        if (selected1.size() >= 4) {
            EstimateDLT(selected1, selected2, models);
        } else {
            models->clear();
        }
    };
    const auto residual = [&points1, &points2](const Matrix& model,
                                               const std::size_t index) {
        return OneWaySquaredResidual(points1[index], points2[index], model);
    };
    return RunLORansac<Matrix>(points1.size(), 4, 4, options, estimate,
                               local_estimate, residual);
}

std::vector<HomographyPoseCandidate> DecomposeHomography(
        const Matrix& homography,
        const Matrix& intrinsics1,
        const Matrix& intrinsics2) {
    Matrix H_normalized = intrinsics2.inverse() * homography * intrinsics1;
    const Eigen::JacobiSVD<Matrix> hmatrix_norm_svd(H_normalized);
    H_normalized.array() /= hmatrix_norm_svd.singularValues()[1];
    if (H_normalized.determinant() < 0.0) {
        H_normalized.array() *= -1.0;
    }
    const Matrix S = H_normalized.transpose() * H_normalized - Matrix::Identity();
    constexpr double kMinInfinityNorm = 1e-3;
    if (S.lpNorm<Eigen::Infinity>() < kMinInfinityNorm) {
        HomographyPoseCandidate candidate;
        candidate.rotation = Eigen::Quaterniond(H_normalized);
        candidate.translation.setZero();
        candidate.normal.setZero();
        return {candidate};
    }

    const double M00 = ComputeOppositeOfMinor(S, 0, 0);
    const double M11 = ComputeOppositeOfMinor(S, 1, 1);
    const double M22 = ComputeOppositeOfMinor(S, 2, 2);
    const double rtM00 = std::sqrt(std::max(M00, 0.0));
    const double rtM11 = std::sqrt(std::max(M11, 0.0));
    const double rtM22 = std::sqrt(std::max(M22, 0.0));
    const double M01 = ComputeOppositeOfMinor(S, 0, 1);
    const double M12 = ComputeOppositeOfMinor(S, 1, 2);
    const double M02 = ComputeOppositeOfMinor(S, 0, 2);
    const int e12 = (M12 > 0.0) - (M12 < 0.0);
    const int e02 = (M02 > 0.0) - (M02 < 0.0);
    const int e01 = (M01 > 0.0) - (M01 < 0.0);
    const std::array<double, 3> nS{{std::abs(S(0, 0)), std::abs(S(1, 1)),
                                    std::abs(S(2, 2))}};
    const std::size_t idx = static_cast<std::size_t>(
            std::distance(nS.begin(), std::max_element(nS.begin(), nS.end())));
    Ray np1;
    Ray np2;
    if (idx == 0) {
        np1[0] = S(0, 0); np2[0] = S(0, 0);
        np1[1] = S(0, 1) + rtM22; np2[1] = S(0, 1) - rtM22;
        np1[2] = S(0, 2) + e12 * rtM11; np2[2] = S(0, 2) - e12 * rtM11;
    } else if (idx == 1) {
        np1[0] = S(0, 1) + rtM22; np2[0] = S(0, 1) - rtM22;
        np1[1] = S(1, 1); np2[1] = S(1, 1);
        np1[2] = S(1, 2) - e02 * rtM00; np2[2] = S(1, 2) + e02 * rtM00;
    } else {
        np1[0] = S(0, 2) + e01 * rtM11; np2[0] = S(0, 2) - e01 * rtM11;
        np1[1] = S(1, 2) + rtM00; np2[1] = S(1, 2) - rtM00;
        np1[2] = S(2, 2); np2[2] = S(2, 2);
    }
    const double traceS = S.trace();
    const double v = 2.0 * std::sqrt(
            std::max(1.0 + traceS - M00 - M11 - M22, 0.0));
    const double ESii = (S(idx, idx) > 0.0) - (S(idx, idx) < 0.0);
    const double r_2 = 2.0 + traceS + v;
    const double nt_2 = 2.0 + traceS - v;
    const double r = std::sqrt(std::max(r_2, 0.0));
    const double n_t = std::sqrt(std::max(nt_2, 0.0));
    const Ray n1 = np1.normalized();
    const Ray n2 = np2.normalized();
    const double half_nt = 0.5 * n_t;
    const double esii_t_r = ESii * r;
    const Ray t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
    const Ray t2_star = half_nt * (esii_t_r * n1 - n_t * n2);
    const Matrix R1 = ComputeHomographyRotation(H_normalized, t1_star, n1, v);
    const Ray t1 = R1 * t1_star;
    const Matrix R2 = ComputeHomographyRotation(H_normalized, t2_star, n2, v);
    const Ray t2 = R2 * t2_star;

    std::vector<HomographyPoseCandidate> candidates(4);
    candidates[0] = {Eigen::Quaterniond(R1), t1, -n1};
    candidates[1] = {Eigen::Quaterniond(R1), -t1, n1};
    candidates[2] = {Eigen::Quaterniond(R2), t2, -n2};
    candidates[3] = {Eigen::Quaterniond(R2), -t2, n2};
    return candidates;
}

HomographyPoseReport RecoverHomographyPose(
        const Matrix& homography,
        const Matrix& intrinsics1,
        const Matrix& intrinsics2,
        const std::vector<Ray>& rays1,
        const std::vector<Ray>& rays2) {
    if (rays1.size() != rays2.size()) {
        throw std::invalid_argument("homography pose ray sizes differ");
    }
    HomographyPoseReport report;
    report.candidates = DecomposeHomography(homography, intrinsics1, intrinsics2);
    if (report.candidates.empty()) {
        throw std::runtime_error("homography decomposition returned no candidate");
    }
    std::vector<Ray> tentative_points;
    double best_reprojection_residual_sum = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < report.candidates.size(); ++i) {
        EvaluatePoseCandidate(&report.candidates[i], rays1, rays2,
                             &tentative_points);
        if (tentative_points.size() > report.triangulated_points.size() ||
            (tentative_points.size() == report.triangulated_points.size() &&
             report.candidates[i].reprojection_residual_sum <
                     best_reprojection_residual_sum)) {
            report.selected_index = i;
            best_reprojection_residual_sum =
                    report.candidates[i].reprojection_residual_sum;
            report.triangulated_points = tentative_points;
        }
    }
    if (!report.triangulated_points.empty()) {
        const HomographyPoseCandidate& selected =
                report.candidates[report.selected_index];
        const Ray center2 = selected.rotation.inverse() * -selected.translation;
        std::vector<double> angles;
        angles.reserve(report.triangulated_points.size());
        for (const Ray& point : report.triangulated_points) {
            angles.push_back(CalculateTriangulationAngle(
                    Ray::Zero(), center2, point));
        }
        report.median_triangulation_angle = Median(std::move(angles));
    }
    return report;
}

double CalculateTriangulationAngle(const Ray& projection_center1,
                                   const Ray& projection_center2,
                                   const Ray& point3d) {
    const double baseline_length_squared =
            (projection_center1 - projection_center2).squaredNorm();
    const double ray_length_squared1 = (point3d - projection_center1).squaredNorm();
    const double ray_length_squared2 = (point3d - projection_center2).squaredNorm();
    const double denominator =
            2.0 * std::sqrt(ray_length_squared1 * ray_length_squared2);
    if (denominator == 0.0) {
        return 0.0;
    }
    const double nominator = ray_length_squared1 + ray_length_squared2 -
                             baseline_length_squared;
    const double angle = std::acos(std::clamp(nominator / denominator, -1.0, 1.0));
    return std::min(angle, M_PI - angle);
}

double Median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("median requires at least one value");
    }
    const double idx_double = 0.5 * (values.size() - 1);
    const std::size_t left_idx = static_cast<std::size_t>(std::floor(idx_double));
    const std::size_t right_idx = static_cast<std::size_t>(std::ceil(idx_double));
    std::nth_element(values.begin(), values.begin() + right_idx, values.end());
    const double right = values[right_idx];
    if (left_idx == right_idx) {
        return right;
    }
    const double left = *std::max_element(values.begin(), values.begin() + right_idx);
    return (right_idx - idx_double) * left + (idx_double - left_idx) * right;
}

bool PassInitialPairGates(const std::size_t inlier_count,
                          const HomographyPoseReport& pose,
                          const InitialPairGates& gates) {
    const HomographyPoseCandidate& selected = pose.candidates.at(pose.selected_index);
    const double minimum_angle_radians =
            gates.minimum_median_triangulation_angle_degrees *
            0.0174532925199432954743716805978692718781530857086181640625;
    return inlier_count >= gates.minimum_inliers &&
           std::abs(selected.translation.z()) <
                   gates.maximum_absolute_z_translation &&
           pose.median_triangulation_angle > minimum_angle_radians;
}

}  // namespace tracking::source_homography
