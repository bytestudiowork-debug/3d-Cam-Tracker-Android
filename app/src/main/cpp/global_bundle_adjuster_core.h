#pragma once

#include <functional>
#include <vector>

namespace tracking {

struct GlobalBundleAdjustmentResult {
    std::vector<double> camera_centers;
    std::vector<double> rotations;
    std::vector<double> points;
    std::vector<double> intrinsics;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    double positive_depth_ratio = 0.0;
    int iterations = 0;
};

bool IsValidRotationMatrix(const double* rotation);

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
                               const std::function<bool(int, int)>& progress_callback = {});

}  // namespace tracking
