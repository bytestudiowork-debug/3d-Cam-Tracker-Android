#include "camera_pose_refiner_core.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>

namespace tracking {
namespace {

struct PoseReprojectionResidual {
    PoseReprojectionResidual(const double* world_point,
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
        std::copy(world_point, world_point + 3, this->world_point);
    }

    template <typename T>
    bool operator()(const T* angle_axis, const T* translation, T* residuals) const {
        const T point[3] = {T(world_point[0]), T(world_point[1]), T(world_point[2])};
        T rotated[3];
        ceres::AngleAxisRotatePoint(angle_axis, point, rotated);
        const T x = rotated[0] + translation[0];
        const T y = rotated[1] + translation[1];
        const T z = rotated[2] + translation[2];
        const T safe_z = z > T(1e-8) ? z : T(1e-8);
        residuals[0] = T(fx) * x / safe_z + T(skew) * y / safe_z + T(cx) - T(observed_x);
        residuals[1] = T(fy) * y / safe_z + T(cy) - T(observed_y);
        return true;
    }

    double world_point[3];
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

}  // namespace

bool RunPoseRefinement(const std::vector<double>& object_points,
                       const std::vector<double>& image_points,
                       const std::vector<double>& intrinsics,
                       const std::vector<double>& initial_pose,
                       double huber_scale_px,
                       int max_iterations,
                       double gradient_tolerance,
                       PoseRefinementResult* result) {
    if (result == nullptr || object_points.size() < 12 || object_points.size() % 3 != 0 ||
        image_points.size() != (object_points.size() / 3) * 2 || intrinsics.size() != 5 ||
        initial_pose.size() != 6 || max_iterations <= 0 ||
        !std::isfinite(huber_scale_px) || huber_scale_px <= 0.0 ||
        !std::isfinite(gradient_tolerance) || gradient_tolerance <= 0.0 ||
        !AllFinite(object_points) || !AllFinite(image_points) ||
        !AllFinite(intrinsics) || !AllFinite(initial_pose) ||
        intrinsics[0] <= 0.0 || intrinsics[3] <= 0.0) {
        return false;
    }

    std::vector<double> angle_axis(initial_pose.begin(), initial_pose.begin() + 3);
    std::vector<double> translation(initial_pose.begin() + 3, initial_pose.end());
    ceres::Problem problem;
    const size_t point_count = object_points.size() / 3;
    for (size_t index = 0; index < point_count; ++index) {
        const double* world_point = object_points.data() + index * 3;
        auto* cost = new ceres::AutoDiffCostFunction<PoseReprojectionResidual, 2, 3, 3>(
                new PoseReprojectionResidual(world_point, image_points[index * 2],
                                             image_points[index * 2 + 1], intrinsics.data()));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(huber_scale_px),
                                 angle_axis.data(), translation.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = max_iterations;
    options.num_threads = 1;
    options.function_tolerance = 1e-8;
    options.gradient_tolerance = gradient_tolerance;
    options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    const double allowed_cost_increase = 1e-9 * std::max(1.0, summary.initial_cost);
    if (!summary.IsSolutionUsable() || !std::isfinite(summary.initial_cost) ||
        !std::isfinite(summary.final_cost) ||
        summary.final_cost > summary.initial_cost + allowed_cost_increase ||
        !AllFinite(angle_axis) || !AllFinite(translation)) {
        return false;
    }

    result->initial_cost = summary.initial_cost;
    result->final_cost = summary.final_cost;
    result->iterations = static_cast<int>(summary.iterations.size());
    std::copy(angle_axis.begin(), angle_axis.end(), result->pose);
    std::copy(translation.begin(), translation.end(), result->pose + 3);
    return true;
}

}  // namespace tracking
