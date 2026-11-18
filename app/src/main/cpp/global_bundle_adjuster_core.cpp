#include "global_bundle_adjuster_core.h"
#include "optimization_robust_policy.h"

#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tracking {
namespace {

struct ReprojectionResidual {
    ReprojectionResidual(double observed_x,
                         double observed_y,
                         double skew)
            : observed_x(observed_x),
              observed_y(observed_y),
              skew(skew) {}

    template <typename T>
    bool operator()(const T* angle_axis,
                    const T* camera_center,
                    const T* point,
                    const T* focal_lengths,
                    const T* principal_point,
                    T* residuals) const {
        const T delta[3] = {
                point[0] - camera_center[0],
                point[1] - camera_center[1],
                point[2] - camera_center[2]};
        T camera_point[3];
        ceres::AngleAxisRotatePoint(angle_axis, delta, camera_point);
        const T safe_z = camera_point[2] > T(1e-8)
                ? camera_point[2]
                : T(1e-8);
        residuals[0] = focal_lengths[0] * camera_point[0] / safe_z +
                       T(skew) * camera_point[1] / safe_z +
                       principal_point[0] - T(observed_x);
        residuals[1] = focal_lengths[1] * camera_point[1] / safe_z +
                       principal_point[1] - T(observed_y);
        return true;
    }

    double observed_x;
    double observed_y;
    double skew;
};

struct PrincipalPointPrior {
    PrincipalPointPrior(double initial_x,
                        double initial_y,
                        double sigma_x,
                        double sigma_y)
            : initial_x(initial_x), initial_y(initial_y),
              sigma_x(sigma_x), sigma_y(sigma_y) {}

    template <typename T>
    bool operator()(const T* principal_point, T* residuals) const {
        residuals[0] = (principal_point[0] - T(initial_x)) / T(sigma_x);
        residuals[1] = (principal_point[1] - T(initial_y)) / T(sigma_y);
        return true;
    }

    double initial_x;
    double initial_y;
    double sigma_x;
    double sigma_y;
};

bool AllFinite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
}

class DisjointSet {
public:
    explicit DisjointSet(size_t size) : parent_(size), rank_(size, 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int Find(int value) {
        if (parent_[value] != value) parent_[value] = Find(parent_[value]);
        return parent_[value];
    }

    void Unite(int first, int second) {
        int a = Find(first);
        int b = Find(second);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }

private:
    std::vector<int> parent_;
    std::vector<int> rank_;
};

class ProgressCallback final : public ceres::IterationCallback {
public:
    ProgressCallback(const std::function<bool(int, int)>& callback,
                     int total_iterations)
            : callback_(callback), total_iterations_(total_iterations) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
        if (callback_ && !callback_(static_cast<int>(summary.iteration) + 1,
                                    total_iterations_)) {
            aborted_ = true;
            return ceres::SOLVER_ABORT;
        }
        return ceres::SOLVER_CONTINUE;
    }

    bool aborted() const { return aborted_; }

private:
    const std::function<bool(int, int)>& callback_;
    int total_iterations_;
    bool aborted_ = false;
};

void ApplyRotation(const double* rotation, const double* point, double* camera_point) {
    camera_point[0] = rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2];
    camera_point[1] = rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2];
    camera_point[2] = rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2];
}

}  // namespace

bool IsValidRotationMatrix(const double* r) {
    if (r == nullptr) return false;
    for (int i = 0; i < 9; ++i) if (!std::isfinite(r[i])) return false;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double value = 0.0;
            for (int k = 0; k < 3; ++k) value += r[row * 3 + k] * r[col * 3 + k];
            const double expected = row == col ? 1.0 : 0.0;
            if (std::abs(value - expected) > 1e-3) return false;
        }
    }
    const double determinant =
            r[0] * (r[4] * r[8] - r[5] * r[7]) -
            r[1] * (r[3] * r[8] - r[5] * r[6]) +
            r[2] * (r[3] * r[7] - r[4] * r[6]);
    return std::isfinite(determinant) && std::abs(determinant - 1.0) <= 1e-3;
}

bool RunGlobalBundleAdjustment(const std::vector<double>& initial_camera_centers,
                               const std::vector<double>& rotations,
                               const std::vector<double>& initial_points,
                               const std::vector<int>& frame_indices,
                               const std::vector<int>& track_indices,
                               const std::vector<double>& image_points,
                               const std::vector<double>& intrinsics,
                               int max_iterations,
                               int num_threads,
                               GlobalBundleAdjustmentResult* result,
                               const std::function<bool(int, int)>& progress_callback,
                               bool refine_principal_point,
                               int image_width,
                               int image_height,
                               bool optimize_rotations,
                               bool optimize_translation,
                               bool optimize_focal_length,
                               const std::vector<int>* point_optimization_mask,
                               ReprojectionLossType reprojection_loss_type) {
    if (result == nullptr) return false;
    *result = GlobalBundleAdjustmentResult{};
    if (initial_camera_centers.empty() ||
        initial_camera_centers.size() % 3 != 0 || initial_points.empty() ||
        initial_points.size() % 3 != 0 || frame_indices.size() < 2 ||
        track_indices.size() != frame_indices.size() ||
        frame_indices.size() > std::numeric_limits<size_t>::max() / 2 ||
        image_points.size() != frame_indices.size() * 2 || intrinsics.size() != 5 ||
        max_iterations <= 0 || num_threads <= 0 ||
        rotations.size() != (initial_camera_centers.size() / 3) * 9 ||
        !AllFinite(initial_camera_centers) || !AllFinite(rotations) ||
        !AllFinite(initial_points) || !AllFinite(image_points) ||
        !AllFinite(intrinsics) || intrinsics[0] < 1e-3 || intrinsics[3] < 1e-3 ||
        intrinsics[0] > 1e8 || intrinsics[3] > 1e8 ||
        image_width < 0 || image_height < 0 ||
        (reprojection_loss_type != ReprojectionLossType::HUBER &&
         reprojection_loss_type != ReprojectionLossType::SOFT_L1 &&
         reprojection_loss_type != ReprojectionLossType::TRIVIAL) ||
        (refine_principal_point && (image_width <= 0 || image_height <= 0))) {
        return false;
    }

    const int frame_count = static_cast<int>(initial_camera_centers.size() / 3);
    const int point_count = static_cast<int>(initial_points.size() / 3);
    if (frame_count < 2 || point_count < 1) return false;
    if (point_optimization_mask != nullptr &&
        (point_optimization_mask->size() != static_cast<size_t>(point_count) ||
         std::any_of(point_optimization_mask->begin(), point_optimization_mask->end(),
                     [](int value) { return value != 0 && value != 1; }))) {
        return false;
    }
    for (int frame = 0; frame < frame_count; ++frame) {
        if (!IsValidRotationMatrix(rotations.data() + frame * 9)) return false;
    }

    DisjointSet components(frame_count);
    std::vector<int> first_frame_for_track(point_count, -1);
    std::vector<int> observations_per_track(point_count, 0);
    std::vector<std::unordered_set<int>> observed_frames_by_track(point_count);
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        const int frame = frame_indices[i];
        const int track = track_indices[i];
        if (frame < 0 || frame >= frame_count || track < 0 || track >= point_count) return false;
        if (!observed_frames_by_track[track].insert(frame).second) return false;
        if (first_frame_for_track[track] < 0) first_frame_for_track[track] = frame;
        else components.Unite(first_frame_for_track[track], frame);
        ++observations_per_track[track];
    }
    const int graph_root = components.Find(0);
    for (int frame = 0; frame < frame_count; ++frame) {
        if (components.Find(frame) != graph_root) return false;
    }
    for (int count : observations_per_track) if (count < 3) return false;

    std::vector<double> centers = initial_camera_centers;
    std::vector<double> points = initial_points;
    const double origin[3] = {centers[0], centers[1], centers[2]};
    int scale_anchor = -1;
    double maximum_distance = 0.0;
    for (int frame = 1; frame < frame_count; ++frame) {
        const double dx = centers[frame * 3] - origin[0];
        const double dy = centers[frame * 3 + 1] - origin[1];
        const double dz = centers[frame * 3 + 2] - origin[2];
        const double distance = std::hypot(dx, dy, dz);
        if (!std::isfinite(distance)) return false;
        if (distance > maximum_distance) {
            maximum_distance = distance;
            scale_anchor = frame;
        }
    }
    if (scale_anchor < 0 || maximum_distance <= 0.0) return false;
    const double world_scale = maximum_distance;
    for (size_t i = 0; i < centers.size(); i += 3) {
        for (int axis = 0; axis < 3; ++axis) centers[i + axis] = (centers[i + axis] - origin[axis]) / world_scale;
    }
    for (size_t i = 0; i < points.size(); i += 3) {
        for (int axis = 0; axis < 3; ++axis) points[i + axis] = (points[i + axis] - origin[axis]) / world_scale;
    }
    if (!AllFinite(centers) || !AllFinite(points)) return false;

    std::vector<double> angle_axes(static_cast<size_t>(frame_count) * 3);
    for (int frame = 0; frame < frame_count; ++frame) {
        ceres::RotationMatrixToAngleAxis(
                ceres::RowMajorAdapter3x3(rotations.data() + frame * 9),
                angle_axes.data() + frame * 3);
    }
    std::vector<double> focal_lengths = {intrinsics[0], intrinsics[3]};
    std::vector<double> principal_point = {intrinsics[2], intrinsics[4]};

    ceres::Problem problem;
    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        const int frame = frame_indices[i];
        const int track = track_indices[i];
        double* angle_axis = angle_axes.data() + frame * 3;
        double* center = centers.data() + frame * 3;
        double* point = points.data() + track * 3;
        auto* cost = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 3, 3, 3, 2, 2>(
                new ReprojectionResidual(image_points[i * 2],
                                         image_points[i * 2 + 1], intrinsics[1]));
        ceres::LossFunction* loss = nullptr;
        if (reprojection_loss_type == ReprojectionLossType::HUBER) {
            loss = new ceres::HuberLoss(kGlobalBundleHuberDeltaPx);
        } else if (reprojection_loss_type == ReprojectionLossType::SOFT_L1) {
            loss = new ceres::SoftLOneLoss(1.0);
        }
        problem.AddResidualBlock(cost, loss,
                                 angle_axis, center, point, focal_lengths.data(),
                                 principal_point.data());
        if (problem.HasParameterBlock(point)) ordering->AddElementToGroup(point, 0);
        if (problem.HasParameterBlock(angle_axis)) ordering->AddElementToGroup(angle_axis, 1);
        if (problem.HasParameterBlock(center)) ordering->AddElementToGroup(center, 1);
    }
    ordering->AddElementToGroup(focal_lengths.data(), 1);
    ordering->AddElementToGroup(principal_point.data(), 1);

    if (point_optimization_mask != nullptr) {
        for (int track = 0; track < point_count; ++track) {
            if ((*point_optimization_mask)[track] == 0) {
                problem.SetParameterBlockConstant(points.data() + track * 3);
            }
        }
    }

    for (int frame = 0; frame < frame_count; ++frame) {
        if (!optimize_rotations || frame == 0) {
            problem.SetParameterBlockConstant(angle_axes.data() + frame * 3);
        }
        if (!optimize_translation || frame == 0) {
            problem.SetParameterBlockConstant(centers.data() + frame * 3);
        }
    }
    if (optimize_translation) {
        problem.SetManifold(centers.data() + scale_anchor * 3,
                            new ceres::SphereManifold<3>());
    }
    if (optimize_focal_length) {
        for (int focal = 0; focal < 2; ++focal) {
            const double initial_focal = focal_lengths[focal];
            problem.SetParameterLowerBound(focal_lengths.data(), focal,
                                           std::max(1e-3, initial_focal * 0.25));
            problem.SetParameterUpperBound(focal_lengths.data(), focal,
                                           initial_focal * 4.0);
        }
    } else {
        problem.SetParameterBlockConstant(focal_lengths.data());
    }
    if (refine_principal_point) {
        const double bound_x = 0.05 * static_cast<double>(image_width);
        const double bound_y = 0.05 * static_cast<double>(image_height);
        const double sigma_x = 0.02 * static_cast<double>(image_width);
        const double sigma_y = 0.02 * static_cast<double>(image_height);
        problem.SetParameterLowerBound(principal_point.data(), 0,
                                       intrinsics[2] - bound_x);
        problem.SetParameterUpperBound(principal_point.data(), 0,
                                       intrinsics[2] + bound_x);
        problem.SetParameterLowerBound(principal_point.data(), 1,
                                       intrinsics[4] - bound_y);
        problem.SetParameterUpperBound(principal_point.data(), 1,
                                       intrinsics[4] + bound_y);
        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<PrincipalPointPrior, 2, 2>(
                        new PrincipalPointPrior(intrinsics[2], intrinsics[4],
                                                sigma_x, sigma_y)),
                nullptr, principal_point.data());
    } else {
        problem.SetParameterBlockConstant(principal_point.data());
    }

    ceres::Solver::Options options;
    if (frame_count <= 100) options.linear_solver_type = ceres::DENSE_SCHUR;
    else {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.preconditioner_type = ceres::JACOBI;
    }
    options.max_num_iterations = std::min(max_iterations,
                                          kGlobalBundleAdjustmentMaxIterations);
    options.num_threads = std::min(num_threads, 4);
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_ordering = ordering;

    ProgressCallback iteration_callback(
            progress_callback,
            std::min(max_iterations, kGlobalBundleAdjustmentMaxIterations));
    if (progress_callback) options.callbacks.push_back(&iteration_callback);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (iteration_callback.aborted()) return false;
    const double allowed_cost_increase = 1e-9 * std::max(1.0, summary.initial_cost);
    if (!summary.IsSolutionUsable() || !std::isfinite(summary.initial_cost) ||
        !std::isfinite(summary.final_cost) ||
        summary.final_cost > summary.initial_cost + allowed_cost_increase ||
        !AllFinite(centers) || !AllFinite(points) || !AllFinite(angle_axes) ||
        !AllFinite(focal_lengths) || !AllFinite(principal_point)) {
        return false;
    }

    std::vector<double> solved_rotations(static_cast<size_t>(frame_count) * 9);
    for (int frame = 0; frame < frame_count; ++frame) {
        ceres::AngleAxisToRotationMatrix(
                angle_axes.data() + frame * 3,
                ceres::RowMajorAdapter3x3(solved_rotations.data() + frame * 9));
        if (!IsValidRotationMatrix(solved_rotations.data() + frame * 9)) return false;
    }

    size_t positive_depth_count = 0;
    std::vector<int> positive_by_track(point_count, 0);
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        const int frame = frame_indices[i];
        const int track = track_indices[i];
        const double delta[3] = {
                points[track * 3] - centers[frame * 3],
                points[track * 3 + 1] - centers[frame * 3 + 1],
                points[track * 3 + 2] - centers[frame * 3 + 2]};
        double camera_point[3];
        ApplyRotation(solved_rotations.data() + frame * 9, delta, camera_point);
        if (camera_point[2] > 1e-8) {
            ++positive_depth_count;
            ++positive_by_track[track];
        }
    }
    const double positive_depth_ratio =
            static_cast<double>(positive_depth_count) / static_cast<double>(frame_indices.size());
    if (positive_depth_ratio < 0.9) return false;
    for (int track = 0; track < point_count; ++track) {
        if (observations_per_track[track] > 0 && positive_by_track[track] < 2) return false;
    }

    for (size_t i = 0; i < centers.size(); i += 3) {
        for (int axis = 0; axis < 3; ++axis) centers[i + axis] = origin[axis] + world_scale * centers[i + axis];
    }
    for (size_t i = 0; i < points.size(); i += 3) {
        for (int axis = 0; axis < 3; ++axis) points[i + axis] = origin[axis] + world_scale * points[i + axis];
    }
    if (!AllFinite(centers) || !AllFinite(points)) return false;

    result->camera_centers = std::move(centers);
    result->rotations = std::move(solved_rotations);
    result->points = std::move(points);
    result->intrinsics = {focal_lengths[0], intrinsics[1], principal_point[0],
                          focal_lengths[1], principal_point[1]};
    result->initial_cost = summary.initial_cost;
    result->final_cost = summary.final_cost;
    result->positive_depth_ratio = positive_depth_ratio;
    result->iterations = static_cast<int>(summary.iterations.size());
    return true;
}

}  // namespace tracking
