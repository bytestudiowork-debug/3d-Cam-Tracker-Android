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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Test-only differential port of source-level COLMAP math/LO-RANSAC behavior
// at c5f9cefc87e5dd596b638e4cee0ff543c7d14755. Minimal E/F solvers and any
// production estimator integration are deliberately not included.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace tracking::colmap_initial_pair_test {

constexpr int kAndroidTrialLimit = 10000;
constexpr int kMaxLocalOptimizationTrials = 10;

struct RansacOptions {
    double max_error = 4.0;
    double min_inlier_ratio = 0.25;
    double confidence = 0.999;
    double dyn_num_trials_multiplier = 3.0;
    int min_num_trials = 100;
    int max_num_trials = 10000;
    // Fixed seeds are intentional in this test-only harness.
    std::uint32_t random_seed = 20261118;
};

struct InlierSupport {
    std::size_t num_inliers = 0;
    double residual_sum = std::numeric_limits<double>::max();
};

InlierSupport EvaluateInlierSupport(const std::vector<double>& residuals,
                                    double max_residual);
bool IsLeftSupportBetter(const InlierSupport& left,
                         const InlierSupport& right);
std::size_t ComputeNumTrials(std::size_t num_inliers,
                             std::size_t num_samples,
                             std::size_t min_num_samples,
                             double confidence,
                             double num_trials_multiplier);

double SquaredSampsonError(const Eigen::Vector3d& ray1,
                           const Eigen::Vector3d& ray2,
                           const Eigen::Matrix3d& matrix);

struct RelativePoseHypothesis {
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    std::size_t positive_depth_count = 0;
};

struct EssentialPoseHypotheses {
    std::vector<RelativePoseHypothesis> candidates;
    std::size_t selected_index = 0;
};

std::optional<EssentialPoseHypotheses> RecoverEssentialPoseHypotheses(
        const Eigen::Matrix3d& essential,
        const std::vector<Eigen::Vector3d>& rays1,
        const std::vector<Eigen::Vector3d>& rays2);

template <typename Model>
struct LORansacReport {
    bool success = false;
    std::size_t num_trials = 0;
    InlierSupport support;
    std::vector<char> inlier_mask;
    std::optional<Model> model;
    bool best_model_is_local = false;
    std::size_t local_optimization_trials = 0;
    std::size_t accepted_local_refits = 0;
};

// Source-faithful COLMAP-style LO-RANSAC control layer. The estimator kernels
// are injected callbacks so this first port can be validated independently;
// source minimal-solvers are a separately bounded follow-up component.
// `residual` returns squared residuals. `estimate` and `local_estimate` receive
// observation indices and append every model candidate they estimate.
template <typename Model,
          typename EstimateFn,
          typename LocalEstimateFn,
          typename ResidualFn>
LORansacReport<Model> RunLORansac(std::size_t num_samples,
                                 std::size_t min_num_samples,
                                 std::size_t local_min_num_samples,
                                 const RansacOptions& options,
                                 EstimateFn estimate,
                                 LocalEstimateFn local_estimate,
                                 ResidualFn residual) {
    if (num_samples > std::numeric_limits<std::uint32_t>::max() ||
        min_num_samples == 0 || local_min_num_samples == 0 ||
        !std::isfinite(options.max_error) || options.max_error <= 0.0 ||
        !std::isfinite(options.min_inlier_ratio) ||
        options.min_inlier_ratio < 0.0 || options.min_inlier_ratio > 1.0 ||
        !std::isfinite(options.confidence) || options.confidence < 0.0 ||
        options.confidence > 1.0 ||
        !std::isfinite(options.dyn_num_trials_multiplier) ||
        options.dyn_num_trials_multiplier < 0.0 ||
        options.min_num_trials < 0 || options.max_num_trials < 1 ||
        options.min_num_trials > options.max_num_trials ||
        options.max_num_trials > kAndroidTrialLimit) {
        throw std::invalid_argument("invalid or unbounded COLMAP LO-RANSAC options");
    }

    LORansacReport<Model> report;
    if (num_samples < min_num_samples) {
        return report;
    }

    const double max_residual = options.max_error * options.max_error;
    const std::size_t prior_inliers = static_cast<std::size_t>(
            options.min_inlier_ratio * 100000.0);
    const std::size_t prior_trial_limit = ComputeNumTrials(
            prior_inliers, 100000, min_num_samples, options.confidence,
            options.dyn_num_trials_multiplier);
    const std::size_t max_num_trials = std::min<std::size_t>(
            options.max_num_trials, prior_trial_limit);
    std::size_t dyn_max_num_trials = max_num_trials;
    const std::size_t min_num_trials =
            static_cast<std::size_t>(options.min_num_trials);

    std::vector<std::size_t> sample_indices(num_samples);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    std::mt19937 prng(options.random_seed);
    const std::uint32_t last_index =
            static_cast<std::uint32_t>(num_samples - 1);
    auto sample_minimal_set = [&]() {
        // COLMAP RandomSampler::Sample + Shuffle: partial Fisher-Yates shuffle.
        for (std::size_t i = 0; i < min_num_samples; ++i) {
            std::uniform_int_distribution<std::uint32_t> distribution(
                    static_cast<std::uint32_t>(i), last_index);
            const std::uint32_t j = distribution(prng);
            std::swap(sample_indices[i], sample_indices[j]);
        }
        return std::vector<std::size_t>(sample_indices.begin(),
                                        sample_indices.begin() + min_num_samples);
    };

    std::vector<double> residuals;
    std::optional<Model> best_model;
    InlierSupport best_support;
    bool best_model_is_local = false;
    bool abort = false;
    for (report.num_trials = 0; report.num_trials < max_num_trials;
         ++report.num_trials) {
        if (abort) {
            // Preserve COLMAP's report counter behavior at adaptive termination.
            report.num_trials += 1;
            break;
        }

        std::vector<Model> sample_models;
        estimate(sample_minimal_set(), &sample_models);
        for (const Model& sample_model : sample_models) {
            residuals.resize(num_samples);
            for (std::size_t i = 0; i < num_samples; ++i) {
                residuals[i] = residual(sample_model, i);
            }
            const InlierSupport support =
                    EvaluateInlierSupport(residuals, max_residual);
            if (IsLeftSupportBetter(support, best_support)) {
                best_support = support;
                best_model = sample_model;
                best_model_is_local = false;

                if (support.num_inliers > min_num_samples &&
                    support.num_inliers >= local_min_num_samples) {
                    for (int local_trial = 0;
                         local_trial < kMaxLocalOptimizationTrials;
                         ++local_trial) {
                        ++report.local_optimization_trials;
                        std::vector<std::size_t> inlier_indices;
                        inlier_indices.reserve(support.num_inliers);
                        for (std::size_t i = 0; i < residuals.size(); ++i) {
                            if (residuals[i] <= max_residual) {
                                inlier_indices.push_back(i);
                            }
                        }
                        std::vector<Model> local_models;
                        local_estimate(inlier_indices, &local_models);
                        const std::size_t previous_best_num_inliers =
                                best_support.num_inliers;
                        bool local_model_improved = false;
                        std::vector<double> best_local_residuals;
                        for (const Model& local_model : local_models) {
                            std::vector<double> candidate_residuals(num_samples);
                            for (std::size_t i = 0; i < num_samples; ++i) {
                                candidate_residuals[i] = residual(local_model, i);
                            }
                            const InlierSupport local_support =
                                    EvaluateInlierSupport(candidate_residuals,
                                                          max_residual);
                            if (IsLeftSupportBetter(local_support, best_support)) {
                                best_support = local_support;
                                best_model = local_model;
                                best_model_is_local = true;
                                best_local_residuals.swap(candidate_residuals);
                                local_model_improved = true;
                                ++report.accepted_local_refits;
                            }
                        }
                        if (best_support.num_inliers <=
                            previous_best_num_inliers) {
                            break;
                        }
                        if (!local_model_improved) {
                            break;
                        }
                        residuals.swap(best_local_residuals);
                    }
                }
                dyn_max_num_trials = ComputeNumTrials(
                        best_support.num_inliers, num_samples, min_num_samples,
                        options.confidence,
                        options.dyn_num_trials_multiplier);
            }
            if (report.num_trials >= dyn_max_num_trials &&
                report.num_trials >= min_num_trials) {
                abort = true;
                break;
            }
        }
    }

    if (!best_model.has_value()) {
        return report;
    }
    report.support = best_support;
    report.model = *best_model;
    report.best_model_is_local = best_model_is_local;
    if (report.support.num_inliers < min_num_samples) {
        return report;
    }

    residuals.resize(num_samples);
    for (std::size_t i = 0; i < num_samples; ++i) {
        residuals[i] = residual(*report.model, i);
    }
    report.inlier_mask.resize(num_samples);
    for (std::size_t i = 0; i < num_samples; ++i) {
        report.inlier_mask[i] = residuals[i] <= max_residual;
    }
    report.success = true;
    return report;
}

}  // namespace tracking::colmap_initial_pair_test
