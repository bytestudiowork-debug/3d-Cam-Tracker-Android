#include "global_bundle_adjuster_core.h"

#include <ceres/ceres.h>
#include <ceres/manifold.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace tracking {
namespace {

struct ReprojectionResidual {
    ReprojectionResidual(const double* rotation,
                         double observed_x,
                         double observed_y,
                         const double* intrinsics)
            : observed_x(observed_x),
              observed_y(observed_y),
              fx(intrinsics[0]),
              skew(intrinsics[1]),
              cx(intrinsics[2]),
              fy(intrinsics[3]),
              cy(intrinsics[4]) {
        std::copy(rotation, rotation + 9, this->rotation);
    }

    template <typename T>
    bool operator()(const T* camera_center, const T* point, T* residuals) const {
        const T dx = point[0] - camera_center[0];
        const T dy = point[1] - camera_center[1];
        const T dz = point[2] - camera_center[2];
        const T x = T(rotation[0]) * dx + T(rotation[1]) * dy + T(rotation[2]) * dz;
        const T y = T(rotation[3]) * dx + T(rotation[4]) * dy + T(rotation[5]) * dz;
        const T z = T(rotation[6]) * dx + T(rotation[7]) * dy + T(rotation[8]) * dz;
        const T safe_z = z > T(1e-8) ? z : T(1e-8);
        residuals[0] = T(fx) * x / safe_z + T(skew) * y / safe_z + T(cx) - T(observed_x);
        residuals[1] = T(fy) * y / safe_z + T(cy) - T(observed_y);
        return true;
    }

    double rotation[9];
    double observed_x;
    double observed_y;
    double fx;
    double skew;
    double cx;
    double fy;
    double cy;
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
    ProgressCallback(const std::function<bool(int, int)>& callback, int total_iterations)
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
                               const std::function<bool(int, int)>& progress_callback) {
    if (result == nullptr || initial_camera_centers.empty() ||
        initial_camera_centers.size() % 3 != 0 || initial_points.empty() ||
        initial_points.size() % 3 != 0 || frame_indices.size() < 2 ||
        track_indices.size() != frame_indices.size() ||
        image_points.size() != frame_indices.size() * 2 || intrinsics.size() != 5 ||
        max_iterations <= 0 || num_threads <= 0 ||
        rotations.size() != (initial_camera_centers.size() / 3) * 9 ||
        !AllFinite(initial_camera_centers) || !AllFinite(rotations) ||
        !AllFinite(initial_points) || !AllFinite(image_points) ||
        !AllFinite(intrinsics) || intrinsics[0] <= 0.0 || intrinsics[3] <= 0.0) {
        return false;
    }

    const int frame_count = static_cast<int>(initial_camera_centers.size() / 3);
    const int point_count = static_cast<int>(initial_points.size() / 3);
    if (frame_count < 2 || point_count < 1) return false;
    for (int frame = 0; frame < frame_count; ++frame) {
        if (!IsValidRotationMatrix(rotations.data() + frame * 9)) return false;
    }

    DisjointSet components(frame_count);
    std::vector<int> first_frame_for_track(point_count, -1);
    std::vector<int> observations_per_track(point_count, 0);
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        const int frame = frame_indices[i];
        const int track = track_indices[i];
        if (frame < 0 || frame >= frame_count || track < 0 || track >= point_count) return false;
        if (first_frame_for_track[track] < 0) first_frame_for_track[track] = frame;
        else components.Unite(first_frame_for_track[track], frame);
        ++observations_per_track[track];
    }
    const int graph_root = components.Find(0);
    for (int frame = 0; frame < frame_count; ++frame) {
        if (components.Find(frame) != graph_root) return false;
    }
    for (int count : observations_per_track) if (count == 1) return false;

    std::vector<double> centers = initial_camera_centers;
    std::vector<double> points = initial_points;
    const double origin[3] = {centers[0], centers[1], centers[2]};
    int scale_anchor = -1;
    double maximum_distance_squared = 0.0;
    for (int frame = 1; frame < frame_count; ++frame) {
        const double dx = centers[frame * 3] - origin[0];
        const double dy = centers[frame * 3 + 1] - origin[1];
        const double dz = centers[frame * 3 + 2] - origin[2];
        const double distance_squared = dx * dx + dy * dy + dz * dz;
        if (distance_squared > maximum_distance_squared) {
            maximum_distance_squared = distance_squared;
            scale_anchor = frame;
        }
    }
    if (scale_anchor < 0 || maximum_distance_squared < 1e-16) return false;
    const double world_scale = std::sqrt(maximum_distance_squared);
    for (size_t i = 0; i < centers.size(); i += 3) {
        for (int axis = 0; axis < 3; ++axis) centers[i + axis] = (centers[i + axis] - origin[axis]) / world_scale;
    }
    for (size_t i = 0; i < points.size(); i += 3) {
        for (int axis = 0; axis < 3; ++axis) points[i + axis] = (points[i + axis] - origin[axis]) / world_scale;
    }

    ceres::Problem problem;
    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        const int frame = frame_indices[i];
        const int track = track_indices[i];
        double* center = centers.data() + frame * 3;
        double* point = points.data() + track * 3;
        const double* rotation = rotations.data() + frame * 9;
        auto* cost = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 3, 3>(
                new ReprojectionResidual(rotation, image_points[i * 2],
                                         image_points[i * 2 + 1], intrinsics.data()));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), center, point);
        if (problem.HasParameterBlock(point)) ordering->AddElementToGroup(point, 0);
        if (problem.HasParameterBlock(center)) ordering->AddElementToGroup(center, 1);
    }

    problem.SetParameterBlockConstant(centers.data());
    problem.SetManifold(centers.data() + scale_anchor * 3, new ceres::SphereManifold<3>());

    ceres::Solver::Options options;
    if (frame_count <= 100) options.linear_solver_type = ceres::DENSE_SCHUR;
    else {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.preconditioner_type = ceres::JACOBI;
    }
    options.max_num_iterations = std::min(max_iterations, 60);
    options.num_threads = std::min(num_threads, 4);
    options.function_tolerance = 1e-6;
    options.gradient_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_ordering = ordering;

    ProgressCallback iteration_callback(
            progress_callback, std::min(max_iterations, 60));
    if (progress_callback) options.callbacks.push_back(&iteration_callback);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (iteration_callback.aborted()) return false;
    const double allowed_cost_increase = 1e-9 * std::max(1.0, summary.initial_cost);
    if (!summary.IsSolutionUsable() || !std::isfinite(summary.initial_cost) ||
        !std::isfinite(summary.final_cost) ||
        summary.final_cost > summary.initial_cost + allowed_cost_increase ||
        !AllFinite(centers) || !AllFinite(points)) {
        return false;
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
        ApplyRotation(rotations.data() + frame * 9, delta, camera_point);
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
    result->points = std::move(points);
    result->initial_cost = summary.initial_cost;
    result->final_cost = summary.final_cost;
    result->positive_depth_ratio = positive_depth_ratio;
    result->iterations = static_cast<int>(summary.iterations.size());
    return true;
}

}  // namespace tracking
