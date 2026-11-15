#include "camera_pose_refiner_core.h"
#include "global_bundle_adjuster_core.h"

#include <ceres/rotation.h>

#include <algorithm>
#include <cmath>
#include <iostream>
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

void TestBundleAdjustmentRecovery() {
    const std::vector<double> intrinsics = {700.0, 0.0, 640.0, 710.0, 360.0};
    std::vector<double> truth_centers = {
            0.0, 0.0, 0.0,
            0.9, 0.05, 0.0,
            1.8, 0.12, 0.03,
            2.7, 0.20, 0.07};
    std::vector<double> rotations;
    for (int frame = 0; frame < 4; ++frame) {
        const auto rotation = RotationFromVector(0.003 * frame, 0.018 * frame, -0.009 * frame);
        rotations.insert(rotations.end(), rotation.begin(), rotation.end());
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
                    rotations.data() + frame * 9,
                    truth_points.data() + track * 3,
                    intrinsics.data(), image);
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
                    initial_centers, rotations, initial_points, frame_indices,
                    track_indices, image_points, intrinsics, 60, 1, &result),
            "global bundle adjustment did not produce a usable result");
    Require(result.final_cost < result.initial_cost * 0.05,
            "global bundle adjustment did not substantially reduce reprojection cost");
    Require(result.positive_depth_ratio == 1.0,
            "synthetic reconstruction failed the positive-depth quality gate");

    for (int frame = 0; frame < 4; ++frame) {
        double error_squared = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double error = result.camera_centers[frame * 3 + axis] - truth_centers[frame * 3 + axis];
            error_squared += error * error;
        }
        Require(std::sqrt(error_squared) < 0.035,
                "bundle adjustment camera center did not recover synthetic truth");
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
    Require(mean_point_error < 0.035,
            "bundle adjustment points did not recover synthetic truth");
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
    Require(!tracking::RunGlobalBundleAdjustment(
                    centers, rotations, points, frame_indices, track_indices,
                    image_points, intrinsics, 30, 1, &result),
            "disconnected camera-track graph was accepted");
}

void TestMalformedRotationIsRejected() {
    double scaled_identity[9] = {2, 0, 0, 0, 2, 0, 0, 0, 2};
    Require(!tracking::IsValidRotationMatrix(scaled_identity),
            "scaled identity was accepted as a camera rotation");
}

}  // namespace

int main() {
    try {
        TestPoseRefinement();
        TestBundleAdjustmentRecovery();
        TestBundleAdjustmentRejectsDisconnectedGraph();
        TestMalformedRotationIsRejected();
        std::cout << "Native numerical regression tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
