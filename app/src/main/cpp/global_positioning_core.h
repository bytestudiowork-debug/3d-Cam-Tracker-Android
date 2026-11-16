#pragma once

#include <functional>
#include <string>
#include <vector>

namespace tracking {

struct GlobalPositioningResult {
    std::vector<double> frame_centers;
    std::vector<double> points;
    std::vector<double> scales;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int iterations = 0;
    std::string solver_report;
};

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
                          const std::function<bool(int, int)>& progress_callback = {});

}  // namespace tracking
