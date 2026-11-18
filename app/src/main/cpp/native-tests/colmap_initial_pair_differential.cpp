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
// Test-only differential port from COLMAP c5f9cefc87e5dd596b638e4cee0ff543c7d14755:
// src/colmap/estimators/utils.cc, src/colmap/optim/support_measurement.cc,
// src/colmap/optim/ransac.h, src/colmap/optim/loransac.h, and
// src/colmap/geometry/{essential_matrix,triangulation}.cc.
#include "colmap_initial_pair_differential.h"

#include <array>

#include <Eigen/SVD>

namespace tracking::colmap_initial_pair_test {

InlierSupport EvaluateInlierSupport(const std::vector<double>& residuals,
                                    const double max_residual) {
    InlierSupport support;
    support.num_inliers = 0;
    support.residual_sum = 0.0;
    for (const double residual : residuals) {
        if (residual <= max_residual) {
            ++support.num_inliers;
            support.residual_sum += residual;
        }
    }
    return support;
}

bool IsLeftSupportBetter(const InlierSupport& left,
                         const InlierSupport& right) {
    if (left.num_inliers > right.num_inliers) {
        return true;
    }
    return left.num_inliers == right.num_inliers &&
           left.residual_sum < right.residual_sum;
}

std::size_t ComputeNumTrials(const std::size_t num_inliers,
                             const std::size_t num_samples,
                             const std::size_t min_num_samples,
                             const double confidence,
                             const double num_trials_multiplier) {
    if (min_num_samples == 0 || num_inliers < min_num_samples ||
        num_samples < min_num_samples) {
        // The mathematical probability of an all-inlier minimal sample is zero.
        // This also avoids unsigned underflow in the source formula's n - i terms.
        return std::numeric_limits<std::size_t>::max();
    }
    const double prob_failure = 1.0 - confidence;
    if (prob_failure <= 0.0) {
        return std::numeric_limits<std::size_t>::max();
    }
    double prob_inlier = 1.0;
    for (std::size_t i = 0; i < min_num_samples; ++i) {
        prob_inlier *= static_cast<double>(num_inliers - i) /
                       static_cast<double>(num_samples - i);
    }
    const double prob_outlier = 1.0 - prob_inlier;
    if (prob_outlier <= 0.0) {
        return 1;
    }
    if (prob_outlier == 1.0) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(std::ceil(
            std::log(prob_failure) / std::log(prob_outlier) *
            num_trials_multiplier));
}

double SquaredSampsonError(const Eigen::Vector3d& ray1,
                           const Eigen::Vector3d& ray2,
                           const Eigen::Matrix3d& matrix) {
    // Preserve COLMAP's ray residual arithmetic and zero-denominator behavior.
    const Eigen::Vector3d epipolar_line1 = matrix * ray1;
    const double numerator = ray2.dot(epipolar_line1);
    const Eigen::Vector4d denominator(ray2.dot(matrix.col(0)),
                                      ray2.dot(matrix.col(1)),
                                      epipolar_line1.x(),
                                      epipolar_line1.y());
    const double denominator_squared_norm = denominator.squaredNorm();
    if (denominator_squared_norm == 0.0) {
        return std::numeric_limits<double>::max();
    }
    return numerator * numerator / denominator_squared_norm;
}

namespace {

bool TriangulateMidpoint(const Eigen::Matrix3d& cam2_from_cam1_rotation,
                         const Eigen::Vector3d& cam2_from_cam1_translation,
                         const Eigen::Vector3d& cam_ray1,
                         const Eigen::Vector3d& cam_ray2,
                         Eigen::Vector3d* point3d_in_cam1) {
    const Eigen::Matrix3d cam1_from_cam2_rotation =
            cam2_from_cam1_rotation.transpose();
    const Eigen::Vector3d cam_ray2_in_cam1 =
            cam1_from_cam2_rotation * cam_ray2;
    const Eigen::Vector3d cam2_in_cam1 =
            cam1_from_cam2_rotation * -cam2_from_cam1_translation;

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

std::size_t CountCheiralPoints(RelativePoseHypothesis* pose,
                               const std::vector<Eigen::Vector3d>& rays1,
                               const std::vector<Eigen::Vector3d>& rays2) {
    pose->positive_depth_mask.assign(rays1.size(), 0);
    std::size_t count = 0;
    for (std::size_t i = 0; i < rays1.size(); ++i) {
        Eigen::Vector3d point3d_in_cam1;
        if (TriangulateMidpoint(pose->rotation, pose->translation,
                                rays1[i], rays2[i], &point3d_in_cam1)) {
            pose->positive_depth_mask[i] = 1;
            ++count;
        }
    }
    return count;
}

}  // namespace

std::optional<EssentialPoseHypotheses> RecoverEssentialPoseHypotheses(
        const Eigen::Matrix3d& essential,
        const std::vector<Eigen::Vector3d>& rays1,
        const std::vector<Eigen::Vector3d>& rays2) {
    if (rays1.size() != rays2.size()) {
        return std::nullopt;
    }
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            essential, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV().transpose();
    if (U.determinant() < 0.0) {
        U *= -1.0;
    }
    if (V.determinant() < 0.0) {
        V *= -1.0;
    }
    Eigen::Matrix3d W;
    W << 0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0,
         0.0, 0.0, 1.0;
    const Eigen::Matrix3d rotation1 = U * W * V;
    const Eigen::Matrix3d rotation2 = U * W.transpose() * V;
    const Eigen::Vector3d translation = U.col(2).normalized();

    EssentialPoseHypotheses result;
    result.candidates = {
        {rotation1, translation, 0},
        {rotation2, translation, 0},
        {rotation1, -translation, 0},
        {rotation2, -translation, 0}};
    for (std::size_t i = 0; i < result.candidates.size(); ++i) {
        result.candidates[i].positive_depth_count =
                CountCheiralPoints(&result.candidates[i], rays1, rays2);
        // COLMAP uses >=, so a later candidate wins an exact tie.
        if (result.candidates[i].positive_depth_count >=
            result.candidates[result.selected_index].positive_depth_count) {
            result.selected_index = i;
        }
    }
    return result;
}

}  // namespace tracking::colmap_initial_pair_test
