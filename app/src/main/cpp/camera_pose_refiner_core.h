#pragma once

#include <vector>

namespace tracking {

struct PoseRefinementResult {
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int iterations = 0;
    double pose[6] = {};
};

bool RunPoseRefinement(const std::vector<double>& object_points,
                       const std::vector<double>& image_points,
                       const std::vector<double>& intrinsics,
                       const std::vector<double>& initial_pose,
                       double huber_scale_px,
                       int max_iterations,
                       double gradient_tolerance,
                       PoseRefinementResult* result);

}  // namespace tracking
