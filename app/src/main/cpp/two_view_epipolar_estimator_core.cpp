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
// Source-level estimator dispatch adapted from the pinned upstream revision
// c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
#include "two_view_epipolar_estimator_core.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/LU>

namespace tracking::source_two_view {
namespace {

bool IsFiniteVector(const Eigen::Vector2d& value) {
    return std::isfinite(value.x()) && std::isfinite(value.y());
}

bool IsFiniteMatrix(const Eigen::Matrix3d& matrix) {
    return matrix.array().isFinite().all();
}

bool IsValidCameraMatrix(const Eigen::Matrix3d& camera_matrix) {
    if (!IsFiniteMatrix(camera_matrix) || camera_matrix(0, 0) <= 0.0 ||
        camera_matrix(1, 1) <= 0.0) {
        return false;
    }
    const double determinant = camera_matrix.determinant();
    return std::isfinite(determinant) && std::abs(determinant) > 1e-15;
}

bool BuildUnitCameraRays(const std::vector<Eigen::Vector2d>& pixels,
                         const Eigen::Matrix3d& inverse_camera_matrix,
                         std::vector<Eigen::Vector3d>* rays) {
    rays->clear();
    rays->reserve(pixels.size());
    for (const Eigen::Vector2d& pixel : pixels) {
        if (!IsFiniteVector(pixel)) {
            return false;
        }
        Eigen::Vector3d ray = inverse_camera_matrix * pixel.homogeneous();
        const double norm = ray.norm();
        if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        ray /= norm;
        rays->push_back(ray);
    }
    return true;
}

template <typename Point>
void GatherByIndices(const std::vector<Point>& source,
                     const std::vector<std::size_t>& indices,
                     std::vector<Point>* selected) {
    selected->clear();
    selected->reserve(indices.size());
    for (const std::size_t index : indices) {
        selected->push_back(source[index]);
    }
}

}  // namespace

bool EstimateTwoViewEpipolarGeometry(
        const std::vector<Eigen::Vector2d>& pixels1,
        const std::vector<Eigen::Vector2d>& pixels2,
        const Eigen::Matrix3d& camera_matrix,
        const double max_error_pixels,
        EpipolarGeometryReport* report,
        const RansacOptions& options) {
    if (report == nullptr || pixels1.size() != pixels2.size() ||
        pixels1.size() < 7 || pixels1.size() > kMaxAndroidInputObservations ||
        !std::isfinite(max_error_pixels) || max_error_pixels <= 0.0 ||
        !IsValidCameraMatrix(camera_matrix)) {
        return false;
    }
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        if (!IsFiniteVector(pixels1[i]) || !IsFiniteVector(pixels2[i])) {
            return false;
        }
    }

    const Eigen::Matrix3d inverse_camera_matrix = camera_matrix.inverse();
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    if (!BuildUnitCameraRays(pixels1, inverse_camera_matrix, &rays1) ||
        !BuildUnitCameraRays(pixels2, inverse_camera_matrix, &rays2)) {
        return false;
    }

    const double mean_focal_length =
            0.5 * (camera_matrix(0, 0) + camera_matrix(1, 1));
    if (!std::isfinite(mean_focal_length) || mean_focal_length <= 0.0) {
        return false;
    }

    auto gather_rays = [&rays1, &rays2](
            const std::vector<std::size_t>& indices,
            std::vector<Eigen::Vector3d>* subset1,
            std::vector<Eigen::Vector3d>* subset2) {
        GatherByIndices(rays1, indices, subset1);
        GatherByIndices(rays2, indices, subset2);
    };
    auto gather_pixels = [&pixels1, &pixels2](
            const std::vector<std::size_t>& indices,
            std::vector<Eigen::Vector2d>* subset1,
            std::vector<Eigen::Vector2d>* subset2) {
        GatherByIndices(pixels1, indices, subset1);
        GatherByIndices(pixels2, indices, subset2);
    };

    RansacOptions essential_options = options;
    essential_options.max_error = max_error_pixels / mean_focal_length;
    auto estimate_essential = [&gather_rays](
            const std::vector<std::size_t>& indices,
            std::vector<Eigen::Matrix3d>* models) {
        std::vector<Eigen::Vector3d> subset1;
        std::vector<Eigen::Vector3d> subset2;
        gather_rays(indices, &subset1, &subset2);
        EstimateEssentialFivePoint(subset1, subset2, models);
        if (models->size() > kMaxEssentialCandidates) {
            models->resize(kMaxEssentialCandidates);
        }
    };
    auto essential_residual = [&rays1, &rays2](
            const Eigen::Matrix3d& model, const std::size_t index) {
        return SquaredSampsonError(rays1[index], rays2[index], model);
    };
    report->essential = RunLORansac<Eigen::Matrix3d>(
            pixels1.size(), 5, 5, essential_options,
            estimate_essential, estimate_essential, essential_residual);

    RansacOptions fundamental_options = options;
    fundamental_options.max_error = max_error_pixels;
    auto estimate_fundamental = [&gather_pixels](
            const std::vector<std::size_t>& indices,
            std::vector<Eigen::Matrix3d>* models) {
        std::vector<Eigen::Vector2d> subset1;
        std::vector<Eigen::Vector2d> subset2;
        gather_pixels(indices, &subset1, &subset2);
        EstimateFundamentalSevenPoint(subset1, subset2, models);
        if (models->size() > kMaxFundamentalCandidates) {
            models->resize(kMaxFundamentalCandidates);
        }
    };
    auto local_estimate_fundamental = [&gather_pixels](
            const std::vector<std::size_t>& indices,
            std::vector<Eigen::Matrix3d>* models) {
        std::vector<Eigen::Vector2d> subset1;
        std::vector<Eigen::Vector2d> subset2;
        gather_pixels(indices, &subset1, &subset2);
        EstimateFundamentalEightPoint(subset1, subset2, models);
    };
    auto fundamental_residual = [&pixels1, &pixels2](
            const Eigen::Matrix3d& model, const std::size_t index) {
        return SquaredSampsonError(pixels1[index].homogeneous(),
                                   pixels2[index].homogeneous(), model);
    };
    report->fundamental = RunLORansac<Eigen::Matrix3d>(
            pixels1.size(), 7, 8, fundamental_options,
            estimate_fundamental, local_estimate_fundamental,
            fundamental_residual);
    return true;
}

bool RecoverEssentialPoseFromPixels(
        const Eigen::Matrix3d& essential,
        const std::vector<Eigen::Vector2d>& pixels1,
        const std::vector<Eigen::Vector2d>& pixels2,
        const Eigen::Matrix3d& camera_matrix,
        const std::vector<char>& epipolar_inlier_mask,
        EssentialPoseResult* result) {
    if (result == nullptr || pixels1.size() != pixels2.size() ||
        pixels1.size() != epipolar_inlier_mask.size() ||
        pixels1.size() > kMaxAndroidInputObservations ||
        !IsFiniteMatrix(essential) || essential.norm() <= 0.0 ||
        !IsValidCameraMatrix(camera_matrix)) {
        return false;
    }

    const Eigen::Matrix3d inverse_camera_matrix = camera_matrix.inverse();
    std::vector<Eigen::Vector3d> rays1;
    std::vector<Eigen::Vector3d> rays2;
    std::vector<std::size_t> selected_indices;
    selected_indices.reserve(pixels1.size());
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        if (!IsFiniteVector(pixels1[i]) || !IsFiniteVector(pixels2[i])) {
            return false;
        }
        if (epipolar_inlier_mask[i] == 0) {
            continue;
        }
        Eigen::Vector3d ray1 = inverse_camera_matrix * pixels1[i].homogeneous();
        Eigen::Vector3d ray2 = inverse_camera_matrix * pixels2[i].homogeneous();
        const double norm1 = ray1.norm();
        const double norm2 = ray2.norm();
        if (!std::isfinite(norm1) || !std::isfinite(norm2) ||
            norm1 <= std::numeric_limits<double>::epsilon() ||
            norm2 <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        rays1.push_back(ray1 / norm1);
        rays2.push_back(ray2 / norm2);
        selected_indices.push_back(i);
    }
    if (selected_indices.size() < 5) {
        return false;
    }

    const auto hypotheses = RecoverEssentialPoseHypotheses(essential, rays1, rays2);
    if (!hypotheses.has_value() || hypotheses->candidates.size() != 4) {
        return false;
    }
    const RelativePoseHypothesis& selected =
            hypotheses->candidates[hypotheses->selected_index];
    if (selected.positive_depth_mask.size() != selected_indices.size() ||
        selected.positive_depth_count == 0) {
        return false;
    }
    result->rotation = selected.rotation;
    result->translation = selected.translation;
    result->selected_candidate = hypotheses->selected_index;
    result->positive_depth_count = selected.positive_depth_count;
    result->cheirality_mask.assign(pixels1.size(), 0);
    for (std::size_t i = 0; i < selected_indices.size(); ++i) {
        result->cheirality_mask[selected_indices[i]] =
                selected.positive_depth_mask[i];
    }
    return true;
}

}  // namespace tracking::source_two_view
