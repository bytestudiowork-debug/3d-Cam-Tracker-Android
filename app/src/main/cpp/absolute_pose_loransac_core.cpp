// Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
// Adapted from COLMAP src/colmap/estimators/pose.cc, pose.h,
// src/colmap/optim/{ransac.h,loransac.h,random_sampler.cc,
// support_measurement.cc}, and their declarations at commit
// c5f9cefc87e5dd596b638e4cee0ff543c7d14755. The sample generation, all-model
// scoring, support ordering, 100-trial floor, finite-population stopping, and
// recursive EPnP refits follow the pinned sources. The fixed pinhole projection,
// explicit 10,000/10 work ceilings, and deterministic-test seed are Android
// adaptations. See app/src/main/assets/licenses/colmap-p3p-loransac-license.txt.

#include "absolute_pose_loransac_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

namespace tracking {
namespace {

constexpr int kMinimalSampleSize = 3;
constexpr int kLocalEstimatorMinimumSamples = 4;
constexpr int kMaximumAndroidTrials = 10000;
constexpr int kMaximumAndroidLocalTrials = 10;

struct Support {
    size_t num_inliers = 0;
    double residual_sum = std::numeric_limits<double>::max();
    std::vector<double> residuals;
};

bool IsLeftBetter(const Support& left, const Support& right) {
    return left.num_inliers > right.num_inliers ||
           (left.num_inliers == right.num_inliers && left.residual_sum < right.residual_sum);
}

bool IsValidIntrinsics(const EpnpCameraIntrinsics& intrinsics) {
    return std::isfinite(intrinsics.fx) && std::isfinite(intrinsics.fy) &&
           std::isfinite(intrinsics.skew) && std::isfinite(intrinsics.cx) &&
           std::isfinite(intrinsics.cy) && intrinsics.fx > 0.0 && intrinsics.fy > 0.0;
}

bool IsValidOptions(const AbsolutePoseLORansacOptions& options) {
    return std::isfinite(options.max_error_px) && options.max_error_px > 0.0 &&
           std::isfinite(options.min_inlier_ratio) && options.min_inlier_ratio >= 0.0 &&
           options.min_inlier_ratio <= 1.0 && std::isfinite(options.confidence) &&
           options.confidence >= 0.0 && options.confidence <= 1.0 &&
           std::isfinite(options.dynamic_trial_multiplier) &&
           options.dynamic_trial_multiplier >= 0.0 && options.min_num_trials >= 0 &&
           options.max_num_trials >= 1 && options.max_num_trials <= kMaximumAndroidTrials &&
           options.min_num_trials <= options.max_num_trials && options.random_seed >= -1 &&
           options.max_local_trials >= 0 &&
           options.max_local_trials <= kMaximumAndroidLocalTrials;
}

Support Score(const Eigen::Matrix3d& rotation,
              const Eigen::Vector3d& translation,
              const std::vector<Eigen::Vector2d>& image_points,
              const std::vector<Eigen::Vector3d>& world_points,
              const EpnpCameraIntrinsics& intrinsics,
              double max_residual) {
    Support support;
    support.residual_sum = 0.0;
    support.residuals.resize(image_points.size(), std::numeric_limits<double>::max());
    const double epsilon = std::numeric_limits<double>::epsilon();
    for (size_t i = 0; i < image_points.size(); ++i) {
        const Eigen::Vector3d point_in_camera = rotation * world_points[i] + translation;
        double residual = std::numeric_limits<double>::max();
        if (point_in_camera.allFinite() && point_in_camera.z() > epsilon) {
            const double x = point_in_camera.x() / point_in_camera.z();
            const double y = point_in_camera.y() / point_in_camera.z();
            const Eigen::Vector2d projected(intrinsics.fx * x + intrinsics.skew * y + intrinsics.cx,
                                            intrinsics.fy * y + intrinsics.cy);
            if (projected.allFinite()) {
                const double error_squared = (projected - image_points[i]).squaredNorm();
                if (std::isfinite(error_squared)) residual = error_squared;
            }
        }
        support.residuals[i] = residual;
        if (residual <= max_residual) {
            ++support.num_inliers;
            support.residual_sum += residual;
        }
    }
    return support;
}

void GatherInliers(const Support& support,
                   const std::vector<Eigen::Vector2d>& image_points,
                   const std::vector<Eigen::Vector3d>& world_points,
                   double max_residual,
                   std::vector<Eigen::Vector2d>* local_image_points,
                   std::vector<Eigen::Vector3d>* local_world_points) {
    local_image_points->clear();
    local_world_points->clear();
    local_image_points->reserve(support.num_inliers);
    local_world_points->reserve(support.num_inliers);
    for (size_t i = 0; i < support.residuals.size(); ++i) {
        if (support.residuals[i] <= max_residual &&
            support.residuals[i] >= 0.0) {
            local_image_points->push_back(image_points[i]);
            local_world_points->push_back(world_points[i]);
        }
    }
}

}  // namespace

size_t ComputeAbsolutePoseRansacNumTrials(size_t num_inliers,
                                          size_t num_samples,
                                          double confidence,
                                          double multiplier) {
    const double probability_failure = 1.0 - confidence;
    if (probability_failure <= 0.0) return std::numeric_limits<size_t>::max();

    double probability_inlier = 1.0;
    for (int i = 0; i < kMinimalSampleSize; ++i) {
        if (num_inliers <= static_cast<size_t>(i) || num_samples <= static_cast<size_t>(i)) {
            probability_inlier = 0.0;
            break;
        }
        probability_inlier *= static_cast<double>(num_inliers - static_cast<size_t>(i)) /
                              static_cast<double>(num_samples - static_cast<size_t>(i));
    }
    const double probability_outlier = 1.0 - probability_inlier;
    if (probability_outlier <= 0.0) return 1;
    if (probability_outlier == 1.0) return std::numeric_limits<size_t>::max();

    const double requested = std::ceil(std::log(probability_failure) /
                                       std::log(probability_outlier) * multiplier);
    if (!std::isfinite(requested) || requested >=
            static_cast<double>(std::numeric_limits<size_t>::max())) {
        return std::numeric_limits<size_t>::max();
    }
    return requested <= 0.0 ? 0 : static_cast<size_t>(requested);
}

bool EstimateAbsolutePoseLORansac(
        const std::vector<Eigen::Vector2d>& image_points,
        const std::vector<Eigen::Vector3d>& world_points,
        const EpnpCameraIntrinsics& intrinsics,
        const AbsolutePoseLORansacOptions& options,
        AbsolutePoseLORansacResult* result) {
    if (result == nullptr) return false;
    *result = AbsolutePoseLORansacResult{};
    if (image_points.size() != world_points.size() ||
        image_points.size() < kMinimalSampleSize || !IsValidIntrinsics(intrinsics) ||
        !IsValidOptions(options)) {
        return false;
    }
    for (size_t i = 0; i < image_points.size(); ++i) {
        if (!image_points[i].allFinite() || !world_points[i].allFinite()) return false;
    }

    constexpr size_t kInitialSampleCount = 100000;
    const size_t initial_inliers = static_cast<size_t>(
            options.min_inlier_ratio * static_cast<double>(kInitialSampleCount));
    size_t max_num_trials = std::min(
            static_cast<size_t>(options.max_num_trials),
            ComputeAbsolutePoseRansacNumTrials(initial_inliers, kInitialSampleCount,
                                                options.confidence,
                                                options.dynamic_trial_multiplier));
    if (max_num_trials == 0) return false;
    size_t dynamic_max_num_trials = max_num_trials;
    const double max_residual = options.max_error_px * options.max_error_px;
    if (!std::isfinite(max_residual)) return false;

    std::mt19937 generator;
    if (options.random_seed >= 0) {
        generator.seed(static_cast<std::mt19937::result_type>(options.random_seed));
    } else {
        std::random_device random_device;
        std::seed_seq seed{random_device(), random_device(), random_device(), random_device()};
        generator.seed(seed);
    }
    std::vector<size_t> sample_indices(image_points.size());
    std::iota(sample_indices.begin(), sample_indices.end(), size_t{0});

    Support best_support;
    best_support.residual_sum = std::numeric_limits<double>::max();
    bool have_best_model = false;
    bool abort = false;
    std::vector<Eigen::Vector2d> sampled_image_points(kMinimalSampleSize);
    std::vector<Eigen::Vector3d> sampled_world_points(kMinimalSampleSize);
    std::vector<Eigen::Vector2d> local_image_points;
    std::vector<Eigen::Vector3d> local_world_points;

    for (size_t trial = 0; trial < max_num_trials; ++trial) {
        if (abort) break;

        // Pinned COLMAP RandomSampler: prefix Fisher–Yates over a persistent
        // [0..N) array, then use the first three indices without replacement.
        for (size_t i = 0; i < kMinimalSampleSize; ++i) {
            std::uniform_int_distribution<size_t> distribution(i, sample_indices.size() - 1);
            const size_t j = distribution(generator);
            std::swap(sample_indices[i], sample_indices[j]);
        }
        for (int i = 0; i < kMinimalSampleSize; ++i) {
            sampled_image_points[i] = image_points[sample_indices[static_cast<size_t>(i)]];
            sampled_world_points[i] = world_points[sample_indices[static_cast<size_t>(i)]];
        }

        std::vector<Eigen::Vector3d> sampled_rays(kMinimalSampleSize);
        bool valid_rays = true;
        for (int i = 0; i < kMinimalSampleSize; ++i) {
            const Eigen::Vector2d& image = sampled_image_points[static_cast<size_t>(i)];
            const double y = (image.y() - intrinsics.cy) / intrinsics.fy;
            const double x = (image.x() - intrinsics.cx - intrinsics.skew * y) / intrinsics.fx;
            Eigen::Vector3d ray(x, y, 1.0);
            const double norm = ray.norm();
            if (!ray.allFinite() || !std::isfinite(norm) || norm <= 0.0) {
                valid_rays = false;
                break;
            }
            sampled_rays[static_cast<size_t>(i)] = ray / norm;
        }
        if (!valid_rays) {
            result->num_trials = static_cast<int>(trial + 1);
            continue;
        }

        std::vector<P3PPoseEstimate> sample_models;
        if (!EstimateP3PPoseCandidates(sampled_rays, sampled_world_points, &sample_models)) {
            result->num_trials = static_cast<int>(trial + 1);
            continue;
        }

        for (const P3PPoseEstimate& sample_model : sample_models) {
            Support support = Score(sample_model.rotation, sample_model.translation,
                                    image_points, world_points, intrinsics, max_residual);
            if (IsLeftBetter(support, best_support)) {
                best_support = std::move(support);
                result->rotation = sample_model.rotation;
                result->translation = sample_model.translation;
                have_best_model = true;

                if (best_support.num_inliers > kMinimalSampleSize &&
                    best_support.num_inliers >= kLocalEstimatorMinimumSamples) {
                    for (int local_trial = 0; local_trial < options.max_local_trials; ++local_trial) {
                        GatherInliers(best_support, image_points, world_points, max_residual,
                                      &local_image_points, &local_world_points);
                        if (local_image_points.size() < kLocalEstimatorMinimumSamples) break;
                        ++result->num_local_trials;
                        const size_t previous_best_num_inliers = best_support.num_inliers;
                        EpnpPoseEstimate local_model;
                        if (EstimateEpnpPose(local_image_points, local_world_points,
                                             intrinsics, &local_model)) {
                            Support local_support = Score(local_model.rotation, local_model.translation,
                                                          image_points, world_points, intrinsics,
                                                          max_residual);
                            if (IsLeftBetter(local_support, best_support)) {
                                best_support = std::move(local_support);
                                result->rotation = local_model.rotation;
                                result->translation = local_model.translation;
                            }
                        }
                        if (best_support.num_inliers <= previous_best_num_inliers) break;
                    }
                }

                dynamic_max_num_trials = ComputeAbsolutePoseRansacNumTrials(
                        best_support.num_inliers, image_points.size(), options.confidence,
                        options.dynamic_trial_multiplier);
            }
            // COLMAP checks its zero-based trial index after every sampled model.
            if (trial >= dynamic_max_num_trials &&
                trial >= static_cast<size_t>(options.min_num_trials)) {
                abort = true;
                break;
            }
        }
        result->num_trials = static_cast<int>(trial + 1);
    }

    if (!have_best_model || best_support.num_inliers < kMinimalSampleSize) {
        result->inlier_indices.clear();
        return false;
    }

    // Re-evaluate the winning model once to produce an input-order inlier mask.
    best_support = Score(result->rotation, result->translation, image_points,
                         world_points, intrinsics, max_residual);
    result->inlier_residual_sum = best_support.residual_sum;
    result->inlier_indices.reserve(best_support.num_inliers);
    for (size_t i = 0; i < best_support.residuals.size(); ++i) {
        if (best_support.residuals[i] <= max_residual) {
            result->inlier_indices.push_back(static_cast<int>(i));
        }
    }
    result->success = result->inlier_indices.size() >= kMinimalSampleSize;
    return result->success;
}

}  // namespace tracking
