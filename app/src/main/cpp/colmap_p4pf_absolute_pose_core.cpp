#include "colmap_p4pf_absolute_pose_core.h"

#include <PoseLib/solvers/p4pf.h>
#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <utility>

namespace tracking {
namespace {

constexpr size_t kMinimalSampleSize = 4;
constexpr size_t kReferencePopulation = 100000;
constexpr int kAndroidMaximumTrials = 10000;
constexpr double kMinimumPositiveDepth = std::numeric_limits<double>::epsilon();

struct Model {
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    double focal = 0.0;
};

struct Support {
    size_t num_inliers = 0;
    double residual_sum = std::numeric_limits<double>::max();
    std::vector<double> residuals;
};

bool IsFinite(const Eigen::Vector2d& value) {
    return value.allFinite();
}

bool IsFinite(const Eigen::Vector3d& value) {
    return value.allFinite();
}

bool IsBetter(const Support& candidate, const Support& current) {
    return candidate.num_inliers > current.num_inliers ||
           (candidate.num_inliers == current.num_inliers &&
            candidate.residual_sum < current.residual_sum);
}

bool IsValidInput(const std::vector<Eigen::Vector2d>& image_points,
                  const std::vector<Eigen::Vector3d>& world_points,
                  const P4pfCameraIntrinsics& intrinsics,
                  const P4pfAbsolutePoseOptions& options) {
    if (image_points.size() != world_points.size() ||
        image_points.size() < kMinimalSampleSize ||
        image_points.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
        !std::isfinite(intrinsics.fx) || intrinsics.fx <= 0.0 ||
        !std::isfinite(intrinsics.fy) || intrinsics.fy <= 0.0 ||
        !std::isfinite(intrinsics.skew) || std::abs(intrinsics.skew) > 1e-9 ||
        !std::isfinite(intrinsics.cx) || !std::isfinite(intrinsics.cy) ||
        !std::isfinite(options.max_error_px) || options.max_error_px <= 0.0 ||
        !std::isfinite(options.min_inlier_ratio) || options.min_inlier_ratio < 0.0 ||
        options.min_inlier_ratio > 1.0 || !std::isfinite(options.confidence) ||
        options.confidence < 0.0 || options.confidence > 1.0 ||
        !std::isfinite(options.dynamic_trial_multiplier) ||
        options.dynamic_trial_multiplier <= 0.0 || options.min_num_trials < 0 ||
        options.max_num_trials < options.min_num_trials ||
        options.max_num_trials <= 0 || options.max_num_trials > kAndroidMaximumTrials ||
        options.random_seed < -1 || options.max_refinement_iterations <= 0 ||
        !std::isfinite(options.refinement_loss_scale) ||
        options.refinement_loss_scale <= 0.0 ||
        !std::isfinite(options.refinement_gradient_tolerance) ||
        options.refinement_gradient_tolerance <= 0.0) {
        return false;
    }
    for (size_t index = 0; index < image_points.size(); ++index) {
        if (!IsFinite(image_points[index]) || !IsFinite(world_points[index])) return false;
    }
    return true;
}

bool MakeModel(const poselib::CameraPose& pose,
               double focal,
               Model* model) {
    if (model == nullptr || !std::isfinite(focal) || focal <= 0.0 ||
        !pose.q.allFinite() || !pose.t.allFinite()) {
        return false;
    }
    Eigen::Quaterniond rotation(pose.q[0], pose.q[1], pose.q[2], pose.q[3]);
    const double norm = rotation.norm();
    if (!std::isfinite(norm) || norm <= 1e-12) return false;
    rotation.normalize();
    model->rotation = rotation;
    model->translation = pose.t;
    model->focal = focal;
    return true;
}

Support MeasureSupport(const Model& model,
                       const std::vector<Eigen::Vector2d>& centered_image_points,
                       const std::vector<Eigen::Vector3d>& world_points,
                       double maximum_squared_error) {
    Support support;
    support.residual_sum = 0.0;
    support.residuals.resize(centered_image_points.size(),
                             std::numeric_limits<double>::infinity());
    const Eigen::Matrix3d rotation = model.rotation.toRotationMatrix();
    for (size_t index = 0; index < world_points.size(); ++index) {
        const Eigen::Vector3d point_camera =
                rotation * world_points[index] + model.translation;
        if (!point_camera.allFinite() || point_camera.z() <= kMinimumPositiveDepth) {
            continue;
        }
        const double predicted_x = model.focal * point_camera.x() / point_camera.z();
        const double predicted_y = model.focal * point_camera.y() / point_camera.z();
        const double dx = predicted_x - centered_image_points[index].x();
        const double dy = predicted_y - centered_image_points[index].y();
        const double residual = dx * dx + dy * dy;
        if (!std::isfinite(residual)) continue;
        support.residuals[index] = residual;
        if (residual <= maximum_squared_error) {
            ++support.num_inliers;
            support.residual_sum += residual;
        }
    }
    return support;
}

struct ReprojectionResidual {
    ReprojectionResidual(const Eigen::Vector3d& world_point,
                         const Eigen::Vector2d& centered_observation)
            : world_point_(world_point),
              observed_x_(centered_observation.x()),
              observed_y_(centered_observation.y()) {}

    template <typename T>
    bool operator()(const T* const quaternion,
                    const T* const translation,
                    const T* const focal_lengths,
                    T* residuals) const {
        const T point[3] = {T(world_point_.x()), T(world_point_.y()), T(world_point_.z())};
        T point_camera[3];
        ceres::QuaternionRotatePoint(quaternion, point, point_camera);
        point_camera[0] += translation[0];
        point_camera[1] += translation[1];
        point_camera[2] += translation[2];
        const T safe_z = point_camera[2] > T(kMinimumPositiveDepth)
                                 ? point_camera[2]
                                 : T(kMinimumPositiveDepth);
        residuals[0] = focal_lengths[0] * point_camera[0] / safe_z - T(observed_x_);
        residuals[1] = focal_lengths[1] * point_camera[1] / safe_z - T(observed_y_);
        return true;
    }

    const Eigen::Vector3d world_point_;
    const double observed_x_;
    const double observed_y_;
};

bool RefinePoseAndFocal(const Model& initial_model,
                        const std::vector<int>& inlier_indices,
                        const std::vector<Eigen::Vector2d>& centered_image_points,
                        const std::vector<Eigen::Vector3d>& world_points,
                        const P4pfAbsolutePoseOptions& options,
                        Model* refined_model,
                        double* refined_fx,
                        double* refined_fy,
                        double* initial_cost,
                        double* final_cost,
                        int* iterations) {
    if (refined_model == nullptr || refined_fx == nullptr || refined_fy == nullptr ||
        initial_cost == nullptr || final_cost == nullptr || iterations == nullptr ||
        inlier_indices.size() < kMinimalSampleSize) {
        return false;
    }

    std::array<double, 4> quaternion = {
            initial_model.rotation.w(), initial_model.rotation.x(),
            initial_model.rotation.y(), initial_model.rotation.z()};
    std::array<double, 3> translation = {
            initial_model.translation.x(), initial_model.translation.y(),
            initial_model.translation.z()};
    std::array<double, 2> focal_lengths = {initial_model.focal, initial_model.focal};

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    std::unique_ptr<ceres::LossFunction> loss(
            new ceres::HuberLoss(options.refinement_loss_scale));
    problem.AddParameterBlock(quaternion.data(), 4, new ceres::QuaternionManifold());
    for (const int index : inlier_indices) {
        if (index < 0 || static_cast<size_t>(index) >= world_points.size()) return false;
        auto* cost = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 4, 3, 2>(
                new ReprojectionResidual(world_points[static_cast<size_t>(index)],
                                         centered_image_points[static_cast<size_t>(index)]));
        problem.AddResidualBlock(cost, loss.get(), quaternion.data(), translation.data(),
                                 focal_lengths.data());
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations = options.max_refinement_iterations;
    solver_options.gradient_tolerance = options.refinement_gradient_tolerance;
    solver_options.linear_solver_type = ceres::DENSE_QR;
    solver_options.num_threads = 1;
    solver_options.minimizer_progress_to_stdout = false;
    solver_options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable() ||
        !std::isfinite(summary.initial_cost) || !std::isfinite(summary.final_cost) ||
        !std::all_of(quaternion.begin(), quaternion.end(), [](double value) {
            return std::isfinite(value);
        }) ||
        !std::all_of(translation.begin(), translation.end(), [](double value) {
            return std::isfinite(value);
        }) ||
        !std::all_of(focal_lengths.begin(), focal_lengths.end(), [](double value) {
            return std::isfinite(value) && value > 0.0;
        })) {
        return false;
    }

    Eigen::Quaterniond refined_rotation(quaternion[0], quaternion[1],
                                        quaternion[2], quaternion[3]);
    const double rotation_norm = refined_rotation.norm();
    if (!std::isfinite(rotation_norm) || rotation_norm <= 1e-12) return false;
    refined_rotation.normalize();
    refined_model->rotation = refined_rotation;
    refined_model->translation = Eigen::Vector3d(translation[0], translation[1], translation[2]);
    refined_model->focal = 0.5 * (focal_lengths[0] + focal_lengths[1]);
    *refined_fx = focal_lengths[0];
    *refined_fy = focal_lengths[1];
    *initial_cost = summary.initial_cost;
    *final_cost = summary.final_cost;
    *iterations = static_cast<int>(summary.iterations.size());
    return refined_model->translation.allFinite() && std::isfinite(refined_model->focal) &&
           refined_model->focal > 0.0;
}

}  // namespace

size_t ComputeP4pfRansacNumTrials(size_t num_inliers,
                                  size_t num_samples,
                                  double confidence,
                                  double multiplier) {
    if (!std::isfinite(confidence) || confidence < 0.0 || confidence > 1.0 ||
        !std::isfinite(multiplier) || multiplier <= 0.0 ||
        num_samples < kMinimalSampleSize) {
        return std::numeric_limits<size_t>::max();
    }
    const double probability_failure = 1.0 - confidence;
    if (probability_failure <= 0.0) return std::numeric_limits<size_t>::max();

    double probability_inlier = 1.0;
    for (size_t index = 0; index < kMinimalSampleSize; ++index) {
        if (num_inliers <= index) {
            probability_inlier = 0.0;
            break;
        }
        probability_inlier *= static_cast<double>(num_inliers - index) /
                              static_cast<double>(num_samples - index);
    }
    const double probability_outlier = 1.0 - probability_inlier;
    if (probability_outlier <= 0.0) return 1;
    if (probability_outlier >= 1.0) return std::numeric_limits<size_t>::max();
    const double trials = std::ceil(std::log(probability_failure) /
                                    std::log(probability_outlier) * multiplier);
    if (!std::isfinite(trials) || trials <= 0.0 ||
        trials >= static_cast<double>(std::numeric_limits<size_t>::max())) {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(trials);
}

bool EstimateAbsolutePoseP4PF(const std::vector<Eigen::Vector2d>& image_points,
                              const std::vector<Eigen::Vector3d>& world_points,
                              const P4pfCameraIntrinsics& initial_intrinsics,
                              const P4pfAbsolutePoseOptions& options,
                              P4pfAbsolutePoseResult* result) {
    if (result == nullptr ||
        !IsValidInput(image_points, world_points, initial_intrinsics, options)) {
        return false;
    }
    *result = P4pfAbsolutePoseResult();

    std::vector<Eigen::Vector2d> centered_image_points(image_points.size());
    for (size_t index = 0; index < image_points.size(); ++index) {
        centered_image_points[index] = image_points[index] -
                Eigen::Vector2d(initial_intrinsics.cx, initial_intrinsics.cy);
    }

    const size_t assumed_inliers = static_cast<size_t>(
            options.min_inlier_ratio * static_cast<double>(kReferencePopulation));
    const size_t initial_max_trials = ComputeP4pfRansacNumTrials(
            assumed_inliers, kReferencePopulation, options.confidence,
            options.dynamic_trial_multiplier);
    size_t max_trials = std::min(static_cast<size_t>(options.max_num_trials), initial_max_trials);
    if (max_trials == 0) return false;
    size_t dynamic_max_trials = max_trials;

    std::mt19937 generator;
    if (options.random_seed >= 0) {
        generator.seed(static_cast<std::mt19937::result_type>(options.random_seed));
    } else {
        std::random_device random_device;
        std::seed_seq seed{random_device(), random_device(), random_device(), random_device()};
        generator.seed(seed);
    }
    // COLMAP RandomSampler initializes [0..N) once, then applies a prefix
    // Fisher–Yates shuffle for each sample. Keep the permutation across trials.
    std::vector<size_t> sample_indices(image_points.size());
    std::iota(sample_indices.begin(), sample_indices.end(), size_t{0});

    Model best_model;
    Support best_support;
    best_support.residual_sum = std::numeric_limits<double>::max();
    const double maximum_squared_error = options.max_error_px * options.max_error_px;
    bool have_model = false;

    for (size_t trial = 0; trial < max_trials; ++trial) {
        const uint32_t last_index = static_cast<uint32_t>(sample_indices.size() - 1);
        for (size_t index = 0; index < kMinimalSampleSize; ++index) {
            std::uniform_int_distribution<uint32_t> distribution(
                    static_cast<uint32_t>(index), last_index);
            const uint32_t selected = distribution(generator);
            std::swap(sample_indices[index], sample_indices[selected]);
        }

        std::vector<Eigen::Vector2d> sample_image_points(kMinimalSampleSize);
        std::vector<Eigen::Vector3d> sample_world_points(kMinimalSampleSize);
        for (size_t index = 0; index < kMinimalSampleSize; ++index) {
            sample_image_points[index] = centered_image_points[sample_indices[index]];
            sample_world_points[index] = world_points[sample_indices[index]];
        }

        std::vector<poselib::CameraPose> sample_poses;
        std::vector<double> sample_focals;
        poselib::p4pf(sample_image_points, sample_world_points,
                      &sample_poses, &sample_focals, true);
        const size_t model_count = std::min(sample_poses.size(), sample_focals.size());
        for (size_t model_index = 0; model_index < model_count; ++model_index) {
            Model candidate;
            if (!MakeModel(sample_poses[model_index], sample_focals[model_index], &candidate)) {
                continue;
            }
            Support support = MeasureSupport(candidate, centered_image_points,
                                             world_points, maximum_squared_error);
            if (IsBetter(support, best_support)) {
                best_support = std::move(support);
                best_model = candidate;
                have_model = true;
                dynamic_max_trials = ComputeP4pfRansacNumTrials(
                        best_support.num_inliers, image_points.size(), options.confidence,
                        options.dynamic_trial_multiplier);
            }
            // Pinned COLMAP RANSAC compares its zero-based trial index with the
            // adaptive bound and also honors min_num_trials.
            if (trial >= dynamic_max_trials &&
                trial >= static_cast<size_t>(options.min_num_trials)) {
                break;
            }
        }
        result->num_trials = static_cast<int>(trial + 1);
        if (trial >= dynamic_max_trials &&
            trial >= static_cast<size_t>(options.min_num_trials)) {
            break;
        }
    }

    if (!have_model || best_support.num_inliers < kMinimalSampleSize) return false;
    result->initial_focal_length = best_model.focal;
    result->inlier_indices.reserve(best_support.num_inliers);
    for (size_t index = 0; index < best_support.residuals.size(); ++index) {
        if (best_support.residuals[index] <= maximum_squared_error) {
            result->inlier_indices.push_back(static_cast<int>(index));
        }
    }

    Model refined_model;
    if (!RefinePoseAndFocal(best_model, result->inlier_indices, centered_image_points,
                            world_points, options, &refined_model, &result->refined_fx,
                            &result->refined_fy, &result->initial_cost,
                            &result->final_cost, &result->refinement_iterations)) {
        result->inlier_indices.clear();
        return false;
    }
    result->rotation = refined_model.rotation.toRotationMatrix();
    result->translation = refined_model.translation;
    result->success = result->rotation.allFinite() && result->translation.allFinite() &&
                      std::isfinite(result->refined_fx) && result->refined_fx > 0.0 &&
                      std::isfinite(result->refined_fy) && result->refined_fy > 0.0;
    return result->success;
}

}  // namespace tracking
