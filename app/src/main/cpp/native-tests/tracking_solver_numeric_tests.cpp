#include "camera_pose_refiner_core.h"
#include "global_bundle_adjuster_core.h"
#include "least_absolute_deviations_core.h"
#include "optimization_robust_policy.h"
#include "global_positioning_core.h"
#include "positioning_direction_residual.h"

#include <ceres/loss_function.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<double> RotationFromVector(const double* angle_axis) {
    const double theta = std::sqrt(angle_axis[0] * angle_axis[0] +
                                   angle_axis[1] * angle_axis[1] +
                                   angle_axis[2] * angle_axis[2]);
    if (theta < 1e-12) return {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double x = angle_axis[0] / theta;
    const double y = angle_axis[1] / theta;
    const double z = angle_axis[2] / theta;
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const double t = 1.0 - c;
    return {
        t*x*x+c, t*x*y-s*z, t*x*z+s*y,
        t*x*y+s*z, t*y*y+c, t*y*z-s*x,
        t*x*z-s*y, t*y*z+s*x, t*z*z+c
    };
}

std::vector<double> RotationFromVector(double x, double y, double z) {
    const double values[3] = {x, y, z};
    return RotationFromVector(values);
}

void Project(const double* center,
             const double* rotation,
             const double* point,
             const double* intrinsics,
             double* image_point) {
    const double delta[3] = {point[0] - center[0], point[1] - center[1], point[2] - center[2]};
    const double x = rotation[0] * delta[0] + rotation[1] * delta[1] + rotation[2] * delta[2];
    const double y = rotation[3] * delta[0] + rotation[4] * delta[1] + rotation[5] * delta[2];
    const double z = rotation[6] * delta[0] + rotation[7] * delta[1] + rotation[8] * delta[2];
    image_point[0] = intrinsics[0] * x / z + intrinsics[1] * y / z + intrinsics[2];
    image_point[1] = intrinsics[3] * y / z + intrinsics[4];
}

double RotationError(const double* first, const double* second) {
    double relative_trace = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            relative_trace += first[row * 3 + column] * second[row * 3 + column];
        }
    }
    const double cosine = std::clamp((relative_trace - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(cosine);
}

double Distance3(const double* first, const double* second) {
    const double dx = first[0] - second[0];
    const double dy = first[1] - second[1];
    const double dz = first[2] - second[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void TestSparseLadReferenceRecoveryWithOutlier() {
    Eigen::SparseMatrix<double> matrix(5, 2);
    const std::vector<Eigen::Triplet<double>> entries = {
            {0, 0, 1.0}, {1, 0, 1.0}, {2, 1, 1.0},
            {3, 1, 1.0}, {4, 0, 1.0}, {4, 1, 1.0}};
    matrix.setFromTriplets(entries.begin(), entries.end());

    Eigen::VectorXd observations(5);
    observations << 1.0, 10.0, 2.0, 2.0, 3.0;
    Eigen::VectorXd solution = Eigen::VectorXd::Zero(2);
    tracking::LeastAbsoluteDeviationsOptions options;
    options.max_num_iterations = 1000;
    options.absolute_tolerance = 1e-8;
    options.relative_tolerance = 1e-8;

    Require(tracking::SolveLeastAbsoluteDeviations(
                    options, matrix, observations, &solution),
            "sparse LAD solver rejected a well-conditioned reference system");
    Require(std::abs(solution[0] - 1.0) < 1e-4 &&
                    std::abs(solution[1] - 2.0) < 1e-4,
            "sparse LAD solver failed the unique outlier-resistant reference solution");
}

void TestSparseLadReturnsFiniteLastIterateAtIterationLimit() {
    Eigen::SparseMatrix<double> matrix(5, 2);
    const std::vector<Eigen::Triplet<double>> entries = {
            {0, 0, 1.0}, {1, 0, 1.0}, {2, 1, 1.0},
            {3, 1, 1.0}, {4, 0, 1.0}, {4, 1, 1.0}};
    matrix.setFromTriplets(entries.begin(), entries.end());
    Eigen::VectorXd observations(5);
    observations << 1.0, 10.0, 2.0, 2.0, 3.0;
    Eigen::VectorXd solution = Eigen::VectorXd::Zero(2);
    tracking::LeastAbsoluteDeviationsOptions options;
    options.max_num_iterations = 1;
    options.absolute_tolerance = 0.0;
    options.relative_tolerance = 0.0;

    Require(tracking::SolveLeastAbsoluteDeviations(
                    options, matrix, observations, &solution),
            "sparse LAD solver changed its finite-last-iterate contract");
    Require(std::abs(solution[0] - 4.375) < 1e-12 &&
                    std::abs(solution[1] - 0.875) < 1e-12,
            "sparse LAD solver changed the first ADMM update at the iteration limit");
}

void TestSparseLadExplicitAnchorRow() {
    Eigen::SparseMatrix<double> matrix(4, 3);
    const std::vector<Eigen::Triplet<double>> entries = {
            {0, 0, -1.0}, {0, 1, 1.0},
            {1, 1, -1.0}, {1, 2, 1.0},
            {2, 0, -1.0}, {2, 2, 1.0},
            {3, 0, 1.0}};
    matrix.setFromTriplets(entries.begin(), entries.end());

    Eigen::VectorXd observations(4);
    observations << 0.1, -0.5, -0.4, 0.02;
    Eigen::VectorXd solution = Eigen::VectorXd::Zero(3);
    tracking::LeastAbsoluteDeviationsOptions options;
    options.max_num_iterations = 1000;
    options.absolute_tolerance = 1e-8;
    options.relative_tolerance = 1e-8;

    Require(tracking::SolveLeastAbsoluteDeviations(
                    options, matrix, observations, &solution),
            "sparse LAD solver rejected a full-column anchored incidence system");
    Require(std::abs(solution[0] - 0.02) < 1e-8 &&
                    std::abs(solution[1] - 0.12) < 1e-8 &&
                    std::abs(solution[2] + 0.38) < 1e-8,
            "sparse LAD solver failed the explicit anchor-row reference solution");
}

void TestPoseRefinement() {
    const std::vector<double> intrinsics = {800.0, 0.0, 640.0, 820.0, 360.0};
    const double truth[6] = {0.04, -0.06, 0.025, 0.15, -0.08, 0.3};
    std::vector<double> points;
    std::vector<double> observations;
    for (int i = 0; i < 72; ++i) {
        const double point[3] = {
                (static_cast<double>(i % 9) - 4.0) * 0.24,
                (static_cast<double>(i / 9) - 3.5) * 0.19,
                4.3 + 0.13 * (i % 11)};
        points.insert(points.end(), point, point + 3);
        double rotated[3];
        ceres::AngleAxisRotatePoint(truth, point, rotated);
        const double z = rotated[2] + truth[5];
        double image[2] = {
                intrinsics[0] * (rotated[0] + truth[3]) / z + intrinsics[2],
                intrinsics[3] * (rotated[1] + truth[4]) / z + intrinsics[4]};
        if (i % 19 == 0) {
            image[0] += 28.0;
            image[1] -= 23.0;
        }
        observations.insert(observations.end(), image, image + 2);
    }

    std::vector<double> initial(truth, truth + 6);
    initial[0] += 0.025;
    initial[1] -= 0.018;
    initial[2] += 0.02;
    initial[3] += 0.12;
    initial[4] -= 0.09;
    initial[5] += 0.07;
    tracking::PoseRefinementResult result;
    Require(tracking::RunPoseRefinement(points, observations, intrinsics, initial,
                                       1.0, 100, 1e-10, &result),
            "pose refinement did not produce a usable result");
    Require(result.final_cost < result.initial_cost * 0.1,
            "pose refinement did not substantially reduce robust reprojection cost");
    double rotation_error = 0.0;
    double translation_error_squared = 0.0;
    for (int i = 0; i < 3; ++i) {
        rotation_error += (result.pose[i] - truth[i]) * (result.pose[i] - truth[i]);
        const double delta = result.pose[i + 3] - truth[i + 3];
        translation_error_squared += delta * delta;
    }
    Require(std::sqrt(rotation_error) < 0.004,
            "pose refinement rotation did not recover the synthetic truth");
    Require(std::sqrt(translation_error_squared) < 0.025,
            "pose refinement translation did not recover the synthetic truth");
}

void TestGlobalHuberLossPolicy() {
    static_assert(tracking::kGlobalBundleHuberDeltaPx == 1.0);
    const ceres::HuberLoss loss(tracking::kGlobalBundleHuberDeltaPx);
    double rho[3] = {};
    loss.Evaluate(4.0, rho);
    Require(std::abs(rho[0] - 3.0) < 1e-12 &&
                    std::abs(rho[1] - 0.5) < 1e-12 &&
                    std::abs(rho[2] + 0.0625) < 1e-12,
            "global BA Huber objective changed from its one-pixel scale");
}

void TestPairwiseDirectionResidualReferenceValues() {
    const double translation_observation[3] = {0.25, -0.5, 1.0};
    double position1[3] = {1.0, 2.0, 3.0};
    double position2[3] = {2.0, 0.0, 5.0};
    double scale[1] = {0.5};
    const double* parameters[3] = {position1, position2, scale};
    double residuals[3] = {};
    double position1_jacobian[9] = {};
    double position2_jacobian[9] = {};
    double scale_jacobian[3] = {};
    double* jacobians[3] = {
            position1_jacobian, position2_jacobian, scale_jacobian};
    std::unique_ptr<ceres::CostFunction> cost(
            tracking::PairwiseDirectionResidual::Create(translation_observation));

    Require(cost->Evaluate(parameters, residuals, jacobians),
            "direction residual AutoDiff evaluation failed");
    const double expected_residuals[3] = {-0.25, 0.5, 0.0};
    const double expected_position1_jacobian[9] = {
            0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5};
    const double expected_position2_jacobian[9] = {
            -0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5};
    const double expected_scale_jacobian[3] = {-1.0, 2.0, -2.0};
    for (int i = 0; i < 3; ++i) {
        Require(std::abs(residuals[i] - expected_residuals[i]) < 1e-12,
                "direction residual differs from its fixed reference output");
        Require(std::abs(scale_jacobian[i] - expected_scale_jacobian[i]) < 1e-12,
                "direction residual scale Jacobian differs from its reference");
    }
    for (int i = 0; i < 9; ++i) {
        Require(std::abs(position1_jacobian[i] -
                         expected_position1_jacobian[i]) < 1e-12,
                "direction residual first-position Jacobian differs from its reference");
        Require(std::abs(position2_jacobian[i] -
                         expected_position2_jacobian[i]) < 1e-12,
                "direction residual second-position Jacobian differs from its reference");
    }
}

void TestBundleAdjustmentJointRecoveryOutliersAndGauge() {
    const std::vector<double> truth_intrinsics = {700.0, 0.0, 640.0, 710.0, 360.0};
    const std::vector<double> initial_intrinsics = {665.0, 0.0, 640.0, 675.0, 360.0};
    std::vector<double> truth_centers = {
            0.0, 0.0, 0.0,
            0.85, 0.18, 0.08,
            1.7, 0.48, 0.22,
            2.55, 0.75, 0.35};
    std::vector<double> truth_rotations;
    std::vector<double> initial_rotations;
    for (int frame = 0; frame < 4; ++frame) {
        const double angle_axis[3] = {0.01 * frame, 0.025 * frame, -0.015 * frame};
        const auto truth_rotation = RotationFromVector(angle_axis);
        truth_rotations.insert(truth_rotations.end(), truth_rotation.begin(), truth_rotation.end());
        const auto initial_rotation = frame == 0
                ? truth_rotation
                : RotationFromVector(angle_axis[0] + 0.012,
                                     angle_axis[1] - 0.009,
                                     angle_axis[2] + 0.007);
        initial_rotations.insert(initial_rotations.end(), initial_rotation.begin(), initial_rotation.end());
    }

    std::vector<double> truth_points;
    for (int i = 0; i < 40; ++i) {
        truth_points.push_back((static_cast<double>(i % 8) - 3.5) * 0.34);
        truth_points.push_back((static_cast<double>(i / 8) - 2.0) * 0.28);
        truth_points.push_back(4.8 + 0.17 * (i % 7));
    }

    std::vector<int> frame_indices;
    std::vector<int> track_indices;
    std::vector<double> image_points;
    for (int track = 0; track < 40; ++track) {
        for (int frame = 0; frame < 4; ++frame) {
            double image[2];
            Project(truth_centers.data() + frame * 3,
                    truth_rotations.data() + frame * 9,
                    truth_points.data() + track * 3,
                    truth_intrinsics.data(), image);
            frame_indices.push_back(frame);
            track_indices.push_back(track);
            image_points.insert(image_points.end(), image, image + 2);
        }
    }

    std::vector<double> initial_centers = truth_centers;
    initial_centers[3] += 0.11;
    initial_centers[4] -= 0.07;
    initial_centers[5] += 0.05;
    initial_centers[6] -= 0.08;
    initial_centers[7] += 0.06;
    initial_centers[8] -= 0.03;
    const double baseline_rotation = 0.04;
    initial_centers[9] = truth_centers[9] * std::cos(baseline_rotation) -
                         truth_centers[10] * std::sin(baseline_rotation);
    initial_centers[10] = truth_centers[9] * std::sin(baseline_rotation) +
                          truth_centers[10] * std::cos(baseline_rotation);
    std::vector<double> initial_points = truth_points;
    for (int i = 0; i < 40; ++i) {
        initial_points[i * 3] += 0.035 * std::sin(0.7 * i);
        initial_points[i * 3 + 1] += 0.03 * std::cos(0.4 * i);
        initial_points[i * 3 + 2] += 0.055 * std::sin(0.3 * i);
    }

    tracking::GlobalBundleAdjustmentResult result;
    Require(tracking::RunGlobalBundleAdjustment(
                    initial_centers, initial_rotations, initial_points, frame_indices,
                    track_indices, image_points, initial_intrinsics, 60, 1, &result),
            "global bundle adjustment did not produce a usable result");
    Require(result.final_cost < result.initial_cost * 0.1,
            "joint bundle adjustment did not substantially reduce robust reprojection cost");
    Require(result.positive_depth_ratio == 1.0,
            "synthetic reconstruction failed the positive-depth quality gate");
    Require(result.rotations.size() == truth_rotations.size() && result.intrinsics.size() == 5,
            "joint bundle adjustment did not return camera rotations and calibration");

    Require(Distance3(result.camera_centers.data(), initial_centers.data()) < 1e-12,
            "reference camera center moved despite the translation gauge");
    Require(RotationError(result.rotations.data(), initial_rotations.data()) < 1e-10,
            "reference camera rotation moved despite the rotation gauge");
    const double initial_anchor_length = Distance3(initial_centers.data() + 9,
                                                   initial_centers.data());
    const double final_anchor_length = Distance3(result.camera_centers.data() + 9,
                                                 result.camera_centers.data());
    Require(std::abs(initial_anchor_length - final_anchor_length) < 1e-8,
            "baseline norm changed despite the scale gauge");

    for (int frame = 0; frame < 4; ++frame) {
        double error_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double error = result.camera_centers[frame * 3 + axis] - truth_centers[frame * 3 + axis];
            error_squared += error * error;
        }
        const double center_error = std::sqrt(error_squared);
        Require(center_error < 0.08,
                "bundle adjustment camera center did not recover frame " +
                        std::to_string(frame) + "; error=" + std::to_string(center_error));
        const double rotation_error = RotationError(result.rotations.data() + frame * 9,
                                                    truth_rotations.data() + frame * 9);
        Require(rotation_error < 0.012,
                "bundle adjustment rotation did not recover frame " + std::to_string(frame) +
                        "; error=" + std::to_string(rotation_error));
    }
    double mean_point_error = 0.0;
    for (int track = 0; track < 40; ++track) {
        double error_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double error = result.points[track * 3 + axis] - truth_points[track * 3 + axis];
            error_squared += error * error;
        }
        mean_point_error += std::sqrt(error_squared);
    }
    mean_point_error /= 40.0;
    Require(mean_point_error < 0.08,
            "joint bundle adjustment points did not recover clean synthetic truth");
    Require(std::abs(result.intrinsics[0] - truth_intrinsics[0]) < 8.0 &&
                    std::abs(result.intrinsics[3] - truth_intrinsics[3]) < 8.0,
            "joint bundle adjustment focal lengths did not recover synthetic truth");
    Require(result.intrinsics[1] == initial_intrinsics[1] &&
                    result.intrinsics[2] == initial_intrinsics[2] &&
                    result.intrinsics[4] == initial_intrinsics[4],
            "focal-only bundle adjustment changed fixed skew or principal point");

    std::vector<double> outlier_image_points = image_points;
    const size_t outlier_observation = 3;
    outlier_image_points[outlier_observation * 2] += 45.0;
    outlier_image_points[outlier_observation * 2 + 1] -= 32.0;
    tracking::GlobalBundleAdjustmentResult robust_result;
    Require(tracking::RunGlobalBundleAdjustment(
                    truth_centers, truth_rotations, truth_points, frame_indices,
                    track_indices, outlier_image_points, truth_intrinsics,
                    60, 1, &robust_result),
            "robust bundle adjustment rejected a solvable scene with one gross observation outlier");
    Require(robust_result.final_cost < robust_result.initial_cost &&
                    robust_result.positive_depth_ratio == 1.0,
            "robust outlier solve did not reduce its objective or preserve cheirality");
    Require(Distance3(robust_result.camera_centers.data(), truth_centers.data()) < 1e-12 &&
                    RotationError(robust_result.rotations.data(), truth_rotations.data()) < 1e-10,
            "robust outlier solve changed the fixed reference pose");
    Require(std::abs(Distance3(robust_result.camera_centers.data() + 9,
                               robust_result.camera_centers.data()) - initial_anchor_length) < 1e-8,
            "robust outlier solve violated the baseline scale gauge");
    double robust_mean_point_error = 0.0;
    for (int frame = 0; frame < 4; ++frame) {
        const double center_error = Distance3(robust_result.camera_centers.data() + frame * 3,
                                              truth_centers.data() + frame * 3);
        Require(center_error < 0.03,
                "robust outlier solve moved frame " + std::to_string(frame) +
                        " center; error=" + std::to_string(center_error));
        const double rotation_error = RotationError(robust_result.rotations.data() + frame * 9,
                                                    truth_rotations.data() + frame * 9);
        Require(rotation_error < 0.01,
                "robust outlier solve moved frame " + std::to_string(frame) +
                        " rotation; error=" + std::to_string(rotation_error));
    }
    for (int track = 0; track < 40; ++track) {
        robust_mean_point_error += Distance3(robust_result.points.data() + track * 3,
                                             truth_points.data() + track * 3);
    }
    robust_mean_point_error /= 40.0;
    const double outlier_track_error = Distance3(robust_result.points.data(), truth_points.data());
    Require(robust_mean_point_error < 0.08,
            "robust outlier solve moved landmarks away from truth; mean=" +
                    std::to_string(robust_mean_point_error) + "; outlier_track=" +
                    std::to_string(outlier_track_error));
    double clean_track_mean_error = 0.0;
    for (int track = 1; track < 40; ++track) {
        clean_track_mean_error += Distance3(robust_result.points.data() + track * 3,
                                            truth_points.data() + track * 3);
    }
    clean_track_mean_error /= 39.0;
    Require(outlier_track_error < 0.1 && clean_track_mean_error < 0.05,
            "Huber outlier recovery exceeded its bounds; outlier_track=" +
                    std::to_string(outlier_track_error) + "; clean_mean=" +
                    std::to_string(clean_track_mean_error));
    Require(robust_result.intrinsics[0] > 0.25 * truth_intrinsics[0] &&
                    robust_result.intrinsics[0] < 4.0 * truth_intrinsics[0] &&
                    robust_result.intrinsics[3] > 0.25 * truth_intrinsics[3] &&
                    robust_result.intrinsics[3] < 4.0 * truth_intrinsics[3],
            "robust outlier solve produced an unbounded focal length");
}

void TestObservablePrincipalPointRefinement() {
    const std::vector<double> truth_intrinsics = {710.0, 2.5, 640.0, 705.0, 360.0};
    const std::vector<double> initial_intrinsics = {710.0, 2.5, 620.0, 705.0, 375.0};
    constexpr int frame_count = 8;
    constexpr int track_count = 160;

    std::vector<double> truth_centers;
    std::vector<double> truth_rotations;
    std::vector<double> initial_rotations;
    for (int frame = 0; frame < frame_count; ++frame) {
        const double phase = 2.0 * 3.14159265358979323846 * frame / frame_count;
        truth_centers.insert(truth_centers.end(), {
                1.8 * (1.0 - std::cos(phase)),
                1.1 * std::sin(phase),
                0.3 * std::sin(2.0 * phase)});
        const double angle_axis[3] = {
                0.12 * std::sin(phase),
                0.17 * std::sin(2.0 * phase),
                -0.10 * std::sin(phase)};
        const auto truth_rotation = RotationFromVector(angle_axis);
        truth_rotations.insert(truth_rotations.end(), truth_rotation.begin(), truth_rotation.end());
        const auto initial_rotation = frame == 0
                ? truth_rotation
                : RotationFromVector(angle_axis[0] + 0.003,
                                     angle_axis[1] - 0.002,
                                     angle_axis[2] + 0.002);
        initial_rotations.insert(initial_rotations.end(),
                                 initial_rotation.begin(), initial_rotation.end());
    }

    std::vector<double> truth_points;
    for (int track = 0; track < track_count; ++track) {
        const int column = track % 16;
        const int row = track / 16;
        truth_points.insert(truth_points.end(), {
                (column - 7.5) * 0.45,
                (row - 4.5) * 0.55,
                5.5 + 0.2 * (track % 17)});
    }

    std::vector<int> frame_indices;
    std::vector<int> track_indices;
    std::vector<double> image_points;
    for (int track = 0; track < track_count; ++track) {
        for (int frame = 0; frame < frame_count; ++frame) {
            double image[2];
            Project(truth_centers.data() + frame * 3,
                    truth_rotations.data() + frame * 9,
                    truth_points.data() + track * 3,
                    truth_intrinsics.data(), image);
            image[0] += 0.04 * std::sin(0.17 * track + 0.31 * frame);
            image[1] += 0.04 * std::cos(0.13 * track - 0.27 * frame);
            if (track % 37 == 0 && frame == 3) {
                image[0] += 45.0;
                image[1] -= 32.0;
            }
            frame_indices.push_back(frame);
            track_indices.push_back(track);
            image_points.insert(image_points.end(), image, image + 2);
        }
    }

    std::vector<double> initial_centers = truth_centers;
    std::vector<double> initial_points = truth_points;
    for (int frame = 1; frame < frame_count; ++frame) {
        initial_centers[frame * 3] += 0.025 * std::sin(frame);
        initial_centers[frame * 3 + 1] -= 0.02 * std::cos(0.7 * frame);
        initial_centers[frame * 3 + 2] += 0.015 * std::sin(0.4 * frame);
    }
    for (int track = 0; track < track_count; ++track) {
        initial_points[track * 3] += 0.012 * std::sin(0.3 * track);
        initial_points[track * 3 + 1] += 0.01 * std::cos(0.2 * track);
        initial_points[track * 3 + 2] += 0.02 * std::sin(0.11 * track);
    }

    tracking::GlobalBundleAdjustmentResult result;
    Require(tracking::RunGlobalBundleAdjustment(
                    initial_centers, initial_rotations, initial_points, frame_indices,
                    track_indices, image_points, initial_intrinsics, 60, 1,
                    &result, {}, true, 1280, 720),
            "observable principal-point refinement did not produce a usable result");
    Require(result.final_cost < result.initial_cost * 0.1 &&
                    result.positive_depth_ratio > 0.99,
            "principal-point refinement did not reduce robust cost with positive depth");
    Require(std::abs(result.intrinsics[2] - truth_intrinsics[2]) < 2.0 &&
                    std::abs(result.intrinsics[4] - truth_intrinsics[4]) < 2.0,
            "principal point did not recover in the broad-coverage, multi-view scene; cx=" +
                    std::to_string(result.intrinsics[2]) + "; cy=" +
                    std::to_string(result.intrinsics[4]));
    Require(result.intrinsics[1] == initial_intrinsics[1],
            "principal-point refinement changed the fixed skew parameter");
    Require(Distance3(result.camera_centers.data(), initial_centers.data()) < 1e-12 &&
                    RotationError(result.rotations.data(), initial_rotations.data()) < 1e-10,
            "principal-point refinement moved the fixed reference pose");
}

void TestBundleAdjustmentRejectsDisconnectedGraph() {
    const std::vector<double> centers = {
            0, 0, 0, 1, 0, 0,
            0, 2, 0, 1, 2, 0};
    const std::vector<double> identity = {
            1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<double> rotations;
    for (int frame = 0; frame < 4; ++frame) {
        rotations.insert(rotations.end(), identity.begin(), identity.end());
    }
    const std::vector<double> points = {0, 0, 5, 0, 0, 6};
    const std::vector<int> frame_indices = {0, 1, 1, 0, 2, 3, 3, 2};
    const std::vector<int> track_indices = {0, 0, 0, 0, 1, 1, 1, 1};
    const std::vector<double> image_points(16, 640.0);
    const std::vector<double> intrinsics = {700, 0, 640, 700, 360};
    tracking::GlobalBundleAdjustmentResult result;
    result.camera_centers = {42.0};
    result.rotations = {43.0};
    result.points = {44.0};
    result.intrinsics = {45.0};
    result.initial_cost = 46.0;
    result.final_cost = 47.0;
    result.positive_depth_ratio = 0.75;
    result.iterations = 48;
    Require(!tracking::RunGlobalBundleAdjustment(
                    centers, rotations, points, frame_indices, track_indices,
                    image_points, intrinsics, 30, 1, &result),
            "disconnected camera-track graph was accepted");
    Require(result.camera_centers.empty() && result.rotations.empty() &&
                    result.points.empty() && result.intrinsics.empty(),
            "failed bundle adjustment retained stale result vectors");
    Require(result.initial_cost == 0.0 && result.final_cost == 0.0 &&
                    result.positive_depth_ratio == 0.0 && result.iterations == 0,
            "failed bundle adjustment retained stale result metrics");
}

void TestBundleAdjustmentRejectsRepeatedFrameObservation() {
    const std::vector<double> centers = {
            0, 0, 0, 1, 0, 0, 0, 1, 0};
    const std::vector<double> identity = {
            1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<double> rotations;
    for (int frame = 0; frame < 3; ++frame) {
        rotations.insert(rotations.end(), identity.begin(), identity.end());
    }
    const std::vector<double> points = {0, 0, 5, 0, 0, 6};
    const std::vector<int> frame_indices = {0, 0, 1, 0, 1, 2};
    const std::vector<int> track_indices = {0, 0, 0, 1, 1, 1};
    const std::vector<double> image_points(12, 640.0);
    const std::vector<double> intrinsics = {700, 0, 640, 700, 360};
    tracking::GlobalBundleAdjustmentResult result;
    Require(!tracking::RunGlobalBundleAdjustment(
                    centers, rotations, points, frame_indices, track_indices,
                    image_points, intrinsics, 30, 1, &result),
            "bundle adjustment accepted duplicate measurements from one frame");
}

void TestBundleAdjustmentHandlesLargeFiniteWorldScale() {
    const std::vector<double> centers = {
            0.0, 0.0, 0.0, 1e200, 0.0, 0.0, 2e200, 0.0, 0.0};
    const std::vector<double> identity = {
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> rotations;
    for (int frame = 0; frame < 3; ++frame) {
        rotations.insert(rotations.end(), identity.begin(), identity.end());
    }
    const std::vector<double> points = {1e200, 0.0, 5e200};
    const std::vector<int> frame_indices = {0, 1, 2};
    const std::vector<int> track_indices = {0, 0, 0};
    const std::vector<double> image_points = {
            780.0, 360.0, 640.0, 360.0, 500.0, 360.0};
    const std::vector<double> intrinsics = {700.0, 0.0, 640.0, 700.0, 360.0};
    tracking::GlobalBundleAdjustmentResult result;

    Require(tracking::RunGlobalBundleAdjustment(
                    centers, rotations, points, frame_indices, track_indices,
                    image_points, intrinsics, 30, 1, &result),
            "bundle adjustment rejected a valid finite large-world scene");
    Require(result.positive_depth_ratio == 1.0,
            "large-world bundle adjustment lost positive camera depth");
    const auto all_finite = [](const std::vector<double>& values) {
        return std::all_of(values.begin(), values.end(),
                           [](double value) { return std::isfinite(value); });
    };
    Require(all_finite(result.camera_centers) && all_finite(result.rotations) &&
                    all_finite(result.points) && all_finite(result.intrinsics),
            "large-world bundle adjustment returned non-finite scene values");
    Require(std::abs(result.camera_centers[3] / 1e200 - 1.0) < 1e-8 &&
                    std::abs(result.camera_centers[6] / 1e200 - 2.0) < 1e-8 &&
                    std::abs(result.points[2] / 1e200 - 5.0) < 1e-8,
            "large-world normalization changed the recovered scene geometry");
}

void TestBundleAdjustmentHandlesSmallFiniteWorldScale() {
    const std::vector<double> centers = {
            0.0, 0.0, 0.0, 1e-12, 0.0, 0.0, 2e-12, 0.0, 0.0};
    const std::vector<double> identity = {
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> rotations;
    for (int frame = 0; frame < 3; ++frame) {
        rotations.insert(rotations.end(), identity.begin(), identity.end());
    }
    const std::vector<double> points = {1e-12, 0.0, 5e-12};
    const std::vector<int> frame_indices = {0, 1, 2};
    const std::vector<int> track_indices = {0, 0, 0};
    const std::vector<double> image_points = {
            780.0, 360.0, 640.0, 360.0, 500.0, 360.0};
    const std::vector<double> intrinsics = {700.0, 0.0, 640.0, 700.0, 360.0};
    tracking::GlobalBundleAdjustmentResult result;

    Require(tracking::RunGlobalBundleAdjustment(
                    centers, rotations, points, frame_indices, track_indices,
                    image_points, intrinsics, 30, 1, &result),
            "bundle adjustment rejected a valid finite small-world scene");
    Require(result.positive_depth_ratio == 1.0,
            "small-world bundle adjustment lost positive camera depth");
    Require(std::abs(result.camera_centers[3] / 1e-12 - 1.0) < 1e-8 &&
                    std::abs(result.camera_centers[6] / 1e-12 - 2.0) < 1e-8 &&
                    std::abs(result.points[2] / 1e-12 - 5.0) < 1e-8,
            "small-world normalization changed the recovered scene geometry");
}

void TestGlobalPositioningCeresRecovery() {
    const std::vector<double> truth_centers = {
            0.0, 0.0, 0.0,
            0.18, 0.02, 0.01,
            0.37, 0.07, 0.02};
    std::vector<double> truth_points;
    for (int track = 0; track < 36; ++track) {
        truth_points.insert(truth_points.end(), {
                (static_cast<double>(track % 9) - 4.0) * 0.075,
                (static_cast<double>(track / 9) - 1.5) * 0.075,
                1.2 + 0.015 * (track % 5)});
    }

    const double world_offset[3] = {1.5, -2.0, 0.7};
    std::vector<double> initial_centers = truth_centers;
    for (size_t i = 0; i < initial_centers.size(); i += 3) {
        initial_centers[i] += world_offset[0];
        initial_centers[i + 1] += world_offset[1];
        initial_centers[i + 2] += world_offset[2];
    }
    initial_centers[3] += 0.012;
    initial_centers[4] -= 0.009;
    initial_centers[8] += 0.015;

    std::vector<double> initial_points = truth_points;
    for (int track = 0; track < 36; ++track) {
        initial_points[track * 3] += world_offset[0] + 0.006 * std::sin(0.4 * track);
        initial_points[track * 3 + 1] += world_offset[1] + 0.005 * std::cos(0.3 * track);
        initial_points[track * 3 + 2] += world_offset[2] + 0.008 * std::sin(0.2 * track);
    }

    std::vector<int> frame_indices;
    std::vector<int> track_indices;
    std::vector<double> directions;
    std::vector<double> initial_scales;
    std::vector<double> loss_scales;
    const int observation_frame_order[] = {1, 0, 2};
    for (int track = 0; track < 36; ++track) {
        for (const int frame : observation_frame_order) {
            const double dx = truth_points[track * 3] - truth_centers[frame * 3];
            const double dy = truth_points[track * 3 + 1] - truth_centers[frame * 3 + 1];
            const double dz = truth_points[track * 3 + 2] - truth_centers[frame * 3 + 2];
            directions.insert(directions.end(), {dx / dz, dy / dz, 1.0});
            frame_indices.push_back(frame);
            track_indices.push_back(track);
            initial_scales.push_back(frame_indices.size() == 1 ? 1.0 / dz : 1.0);
            loss_scales.push_back(frame == 1 ? 0.5 : 1.0);
        }
    }

    tracking::GlobalPositioningResult result;
    int progress_calls = 0;
    Require(tracking::RunGlobalPositioning(
                    initial_centers, initial_points, frame_indices, track_indices,
                    directions, initial_scales, loss_scales, 100, 1, &result,
                    [&progress_calls](int, int) {
                        ++progress_calls;
                        return true;
                    }),
            "GLOMAP point-to-camera Ceres solve did not produce a usable result");
    Require(result.final_cost < result.initial_cost * 0.05,
            "GLOMAP positioning did not substantially reduce its robust BATA cost");
    Require(result.frame_centers.size() == truth_centers.size() &&
                    result.points.size() == truth_points.size() &&
                    result.scales.size() == frame_indices.size(),
            "GLOMAP positioning returned malformed camera, point, or scale arrays");
    Require(frame_indices.front() == 1,
            "GLOMAP gauge fixture did not select the intended first active frame");
    Require(progress_calls > 0 && result.iterations > 0,
            "GLOMAP positioning did not report native solver progress");
    for (int axis = 0; axis < 3; ++axis) {
        Require(std::abs(result.frame_centers[3 + axis] - initial_centers[3 + axis]) < 1e-12,
                "GLOMAP positioning moved the first active camera's translation gauge");
    }
    Require(std::abs(result.scales.front() - initial_scales.front()) < 1e-12,
            "GLOMAP positioning changed the first fixed observation scale");
    Require(std::all_of(result.scales.begin(), result.scales.end(),
                        [](double scale) { return std::isfinite(scale) && scale >= 1e-5; }),
            "GLOMAP positioning violated the observation-scale lower bound");

    for (int frame = 1; frame < 3; ++frame) {
        double recovered_relative[3];
        double expected_relative[3];
        for (int axis = 0; axis < 3; ++axis) {
            recovered_relative[axis] = result.frame_centers[frame * 3 + axis] -
                                       result.frame_centers[axis];
            expected_relative[axis] = truth_centers[frame * 3 + axis] -
                                      truth_centers[axis];
        }
        Require(Distance3(recovered_relative, expected_relative) < 0.02,
                "GLOMAP positioning camera centers did not recover the synthetic baseline");
    }

    double mean_relative_point_error = 0.0;
    for (int track = 0; track < 36; ++track) {
        double recovered_relative[3];
        double expected_relative[3];
        for (int axis = 0; axis < 3; ++axis) {
            recovered_relative[axis] = result.points[track * 3 + axis] -
                                       result.frame_centers[axis];
            expected_relative[axis] = truth_points[track * 3 + axis] -
                                      truth_centers[axis];
        }
        mean_relative_point_error += Distance3(recovered_relative, expected_relative);
    }
    mean_relative_point_error /= 36.0;
    Require(mean_relative_point_error < 0.02,
            "GLOMAP positioning points did not recover the synthetic scene geometry");
}

void TestGlobalPositioningRejectsDisconnectedTrackGraph() {
    const std::vector<double> centers = {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::vector<double> points = {
            0.0, 0.0, 5.0, 0.0, 0.0, 5.0};
    const std::vector<int> frame_indices = {0, 1, 2, 3, 4, 5};
    const std::vector<int> track_indices = {0, 0, 0, 1, 1, 1};
    const std::vector<double> directions = {
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0};
    const std::vector<double> scales(6, 0.2);
    const std::vector<double> loss_scales(6, 1.0);
    tracking::GlobalPositioningResult result;

    Require(!tracking::RunGlobalPositioning(
                    centers, points, frame_indices, track_indices, directions,
                    scales, loss_scales, 20, 1, &result),
            "global positioning accepted two disconnected camera-track components");
}

void TestGlobalPositioningRejectsRepeatedFrameObservation() {
    const std::vector<double> centers = {
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    const std::vector<double> points = {0.0, 0.0, 5.0};
    const std::vector<int> frame_indices = {0, 1, 2, 0};
    const std::vector<int> track_indices = {0, 0, 0, 0};
    const std::vector<double> directions = {
            0.0, 0.0, 1.0, -0.2, 0.0, 1.0,
            -0.4, 0.0, 1.0, 0.0, 0.0, 1.0};
    const std::vector<double> scales(4, 0.2);
    const std::vector<double> loss_scales(4, 1.0);
    tracking::GlobalPositioningResult result;

    Require(!tracking::RunGlobalPositioning(
                    centers, points, frame_indices, track_indices, directions,
                    scales, loss_scales, 20, 1, &result),
            "global positioning accepted duplicate measurements from one frame");
}

void TestGlobalPositioningRejectsZeroDirection() {
    const std::vector<double> centers(9, 0.0);
    const std::vector<double> points(3, 0.0);
    const std::vector<int> frame_indices = {0, 1, 2};
    const std::vector<int> track_indices = {0, 0, 0};
    const std::vector<double> directions(9, 0.0);
    const std::vector<double> scales(3, 0.2);
    const std::vector<double> loss_scales(3, 1.0);
    tracking::GlobalPositioningResult result;

    Require(!tracking::RunGlobalPositioning(
                    centers, points, frame_indices, track_indices, directions,
                    scales, loss_scales, 20, 1, &result),
            "global positioning accepted zero-length viewing directions");
}

void TestMalformedRotationIsRejected() {
    double scaled_identity[9] = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    Require(!tracking::IsValidRotationMatrix(scaled_identity),
            "scaled identity was accepted as a camera rotation");
}

}  // namespace

int main() {
    try {
        TestPairwiseDirectionResidualReferenceValues();
        TestSparseLadReferenceRecoveryWithOutlier();
        TestSparseLadReturnsFiniteLastIterateAtIterationLimit();
        TestSparseLadExplicitAnchorRow();
        TestGlobalHuberLossPolicy();
        TestPoseRefinement();
        TestBundleAdjustmentJointRecoveryOutliersAndGauge();
        TestObservablePrincipalPointRefinement();
        TestBundleAdjustmentRejectsDisconnectedGraph();
        TestBundleAdjustmentRejectsRepeatedFrameObservation();
        TestBundleAdjustmentHandlesLargeFiniteWorldScale();
        TestBundleAdjustmentHandlesSmallFiniteWorldScale();
        TestGlobalPositioningCeresRecovery();
        TestGlobalPositioningRejectsDisconnectedTrackGraph();
        TestGlobalPositioningRejectsRepeatedFrameObservation();
        TestGlobalPositioningRejectsZeroDirection();
        TestMalformedRotationIsRejected();
        std::cout << "Native numerical regression tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
