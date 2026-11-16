// Copyright (c) 2024, ETH Zurich.
// Source-derived view-graph focal calibration from GLOMAP commit
// 0edb1b8435e0f9a594318908b81a31f078a51bf7.
// Fetzer residual code adapted from glomap/estimators/cost_function.h; that
// source section identifies the DMAP assets by Philipp Lindenberger.
#pragma once

#include <cstdint>
#include <vector>

namespace tracking {

struct ViewGraphCalibrationResult {
    double focal_length = 0.0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int iterations = 0;
    std::vector<double> pair_error_norms;
    std::vector<std::uint8_t> pair_is_valid;
};

// fundamental_matrices contains row-major 3x3 matrices concatenated in pair
// order. A shared scalar focal is estimated for one physical camera, matching
// GLOMAP's same-camera Fetzer residual. Principal point and skew are held fixed
// outside this solver.
bool RunViewGraphFocalCalibration(
        const std::vector<double>& fundamental_matrices,
        double principal_x,
        double principal_y,
        double initial_focal_length,
        int max_num_iterations,
        int num_threads,
        ViewGraphCalibrationResult* result);

}  // namespace tracking
