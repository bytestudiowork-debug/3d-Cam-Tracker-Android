#include "global_positioning_core.h"

#include "positioning_direction_residual.h"

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

namespace tracking {
namespace {

bool AllFinite(const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
}

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

}  // namespace

bool RunGlobalPositioning(const std::vector<double>& initial_frame_centers,
                          const std::vector<double>& initial_points,
                          const std::vector<int>& frame_indices,
                          const std::vector<int>& track_indices,
                          const std::vector<double>& directions,
                          const std::vector<double>& initial_scales,
                          const std::vector<double>& loss_scales,
                          int max_iterations,
                          int num_threads,
                          GlobalPositioningResult* result,
                          const std::function<bool(int, int)>& progress_callback) {
    if (result == nullptr || initial_frame_centers.empty() ||
        initial_frame_centers.size() % 3 != 0 || initial_points.empty() ||
        initial_points.size() % 3 != 0 || frame_indices.empty() ||
        frame_indices.size() > std::numeric_limits<size_t>::max() / 3 ||
        track_indices.size() != frame_indices.size() ||
        directions.size() != frame_indices.size() * 3 ||
        initial_scales.size() != frame_indices.size() ||
        loss_scales.size() != frame_indices.size() || max_iterations <= 0 ||
        num_threads <= 0 ||
        initial_frame_centers.size() / 3 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        initial_points.size() / 3 > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        !AllFinite(initial_frame_centers) || !AllFinite(initial_points) ||
        !AllFinite(directions) || !AllFinite(initial_scales) ||
        !AllFinite(loss_scales)) {
        return false;
    }

    const int frame_count = static_cast<int>(initial_frame_centers.size() / 3);
    const int point_count = static_cast<int>(initial_points.size() / 3);
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        if (frame_indices[i] < 0 || frame_indices[i] >= frame_count ||
            track_indices[i] < 0 || track_indices[i] >= point_count ||
            loss_scales[i] <= 0.0) {
            return false;
        }
    }

    std::vector<double> frame_centers = initial_frame_centers;
    std::vector<double> points = initial_points;
    std::vector<double> scales = initial_scales;
    for (double& scale : scales) scale = std::max(1e-5, scale);

    ceres::Problem problem;
    for (size_t i = 0; i < frame_indices.size(); ++i) {
        ceres::CostFunction* cost =
                PairwiseDirectionResidual::Create(directions.data() + i * 3);
        ceres::LossFunction* loss = new ceres::ScaledLoss(
                new ceres::HuberLoss(0.1), loss_scales[i],
                ceres::TAKE_OWNERSHIP);
        double* camera = frame_centers.data() + frame_indices[i] * 3;
        double* point = points.data() + track_indices[i] * 3;
        double* scale = scales.data() + i;
        problem.AddResidualBlock(cost, loss, camera, point, scale);
        problem.SetParameterLowerBound(scale, 0, 1e-5);
    }

    // BATA observes only point-to-camera differences, so the scene's common
    // translation is unobservable. Preserve the initialized origin by fixing
    // the camera in the first active observation.
    problem.SetParameterBlockConstant(
            frame_centers.data() + frame_indices.front() * 3);
    // GLOMAP's ONLY_POINTS gauge handling fixes the first active observation scale.
    problem.SetParameterBlockConstant(scales.data());

    auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
    for (size_t i = 0; i < scales.size(); ++i) {
        double* scale = scales.data() + i;
        if (problem.HasParameterBlock(scale)) ordering->AddElementToGroup(scale, 0);
    }
    for (int i = 0; i < point_count; ++i) {
        double* point = points.data() + i * 3;
        if (problem.HasParameterBlock(point)) ordering->AddElementToGroup(point, 1);
    }
    for (int i = 0; i < frame_count; ++i) {
        double* center = frame_centers.data() + i * 3;
        if (problem.HasParameterBlock(center)) ordering->AddElementToGroup(center, 2);
    }

    const int iteration_limit = std::min(max_iterations, 100);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    // SPARSE_SCHUR performs a direct sparse Schur solve; iterative-Schur
    // preconditioners (including CLUSTER_TRIDIAGONAL) do not apply here.
    // Keep the upstream objective, robust losses, scale gauge and
    // parameter ordering unchanged.
    options.max_num_iterations = iteration_limit;
    options.num_threads = std::min(num_threads, 4);
    options.function_tolerance = 1e-5;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_ordering = ordering;

    ProgressCallback iteration_callback(progress_callback, iteration_limit);
    if (progress_callback) options.callbacks.push_back(&iteration_callback);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (iteration_callback.aborted() || !summary.IsSolutionUsable() ||
        !std::isfinite(summary.initial_cost) || !std::isfinite(summary.final_cost) ||
        !AllFinite(frame_centers) || !AllFinite(points) || !AllFinite(scales)) {
        return false;
    }

    result->frame_centers = std::move(frame_centers);
    result->points = std::move(points);
    result->scales = std::move(scales);
    result->initial_cost = summary.initial_cost;
    result->final_cost = summary.final_cost;
    result->iterations = static_cast<int>(summary.iterations.size());
    result->solver_report = summary.FullReport();
    return true;
}

}  // namespace tracking
