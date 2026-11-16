// Copyright (c) 2024, ETH Zurich.
// Source-derived from GLOMAP commit 0edb1b8435e0f9a594318908b81a31f078a51bf7:
// glomap/estimators/view_graph_calibration.cc and cost_function.h.
// The Fetzer focal residual section in cost_function.h identifies the DMAP
// assets by Philipp Lindenberger.
#include "view_graph_calibration_core.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <ceres/ceres.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace tracking {
namespace {

constexpr double kTwoViewErrorThreshold = 2.0;
constexpr double kCauchyLossScale = 1e-2;
constexpr double kMinimumFocal = 1e-3;
constexpr double kMinimumSingularValueRatio = 1e-12;

Eigen::Vector4d FetzerD(const Eigen::Vector3d& ai,
                        const Eigen::Vector3d& bi,
                        const Eigen::Vector3d& aj,
                        const Eigen::Vector3d& bj,
                        const int u,
                        const int v) {
    Eigen::Vector4d d;
    d.setZero();
    d(0) = ai(u) * aj(v) - ai(v) * aj(u);
    d(1) = ai(u) * bj(v) - ai(v) * bj(u);
    d(2) = bi(u) * aj(v) - bi(v) * aj(u);
    d(3) = bi(u) * bj(v) - bi(v) * bj(u);
    return d;
}

std::array<Eigen::Vector4d, 3> FetzerDs(const Eigen::Matrix3d& i1_G_i0) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            i1_G_i0, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Vector3d s = svd.singularValues();
    const Eigen::Vector3d v0 = svd.matrixV().col(0);
    const Eigen::Vector3d v1 = svd.matrixV().col(1);
    const Eigen::Vector3d u0 = svd.matrixU().col(0);
    const Eigen::Vector3d u1 = svd.matrixU().col(1);

    const Eigen::Vector3d ai(
            s(0) * s(0) * (v0(0) * v0(0) + v0(1) * v0(1)),
            s(0) * s(1) * (v0(0) * v1(0) + v0(1) * v1(1)),
            s(1) * s(1) * (v1(0) * v1(0) + v1(1) * v1(1)));
    const Eigen::Vector3d aj(
            u1(0) * u1(0) + u1(1) * u1(1),
            -(u0(0) * u1(0) + u0(1) * u1(1)),
            u0(0) * u0(0) + u0(1) * u0(1));
    const Eigen::Vector3d bi(
            s(0) * s(0) * v0(2) * v0(2),
            s(0) * s(1) * v0(2) * v1(2),
            s(1) * s(1) * v1(2) * v1(2));
    const Eigen::Vector3d bj(
            u1(2) * u1(2), -(u0(2) * u1(2)), u0(2) * u0(2));

    std::array<Eigen::Vector4d, 3> ds;
    ds[0] = FetzerD(ai, bi, aj, bj, 1, 0);
    ds[1] = FetzerD(ai, bi, aj, bj, 0, 2);
    ds[2] = FetzerD(ai, bi, aj, bj, 2, 1);
    return ds;
}

class FetzerFocalLengthSameCameraCost {
public:
    FetzerFocalLengthSameCameraCost(const Eigen::Matrix3d& fundamental,
                                   const Eigen::Vector2d& principal_point) {
        Eigen::Matrix3d k0 = Eigen::Matrix3d::Identity();
        k0(0, 2) = principal_point(0);
        k0(1, 2) = principal_point(1);
        const Eigen::Matrix3d k1 = k0;
        const Eigen::Matrix3d i1_g_i0 = k1.transpose() * fundamental * k0;
        const std::array<Eigen::Vector4d, 3> ds = FetzerDs(i1_g_i0);
        d01_ = ds[0];
        d12_ = ds[2];
    }

    template <typename T>
    bool operator()(const T* const focal_parameter, T* residuals) const {
        const Eigen::Vector<T, 4> d01 = d01_.cast<T>();
        const Eigen::Vector<T, 4> d12 = d12_.cast<T>();
        const T focal = focal_parameter[0];
        const T focal_j = focal_parameter[0];

        T di = focal_j * focal_j * d01(0) + d01(1);
        T dj = focal * focal * d12(0) + d12(2);
        di = di == T(0) ? T(1e-6) : di;
        dj = dj == T(0) ? T(1e-6) : dj;

        const T k0_01 = -(focal_j * focal_j * d01(2) + d01(3)) / di;
        const T k1_12 = -(focal * focal * d12(1) + d12(3)) / dj;
        residuals[0] = (focal * focal - k0_01) / (focal * focal);
        residuals[1] = (focal_j * focal_j - k1_12) / (focal_j * focal_j);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Matrix3d& fundamental,
                                       const Eigen::Vector2d& principal_point) {
        return new ceres::AutoDiffCostFunction<
                FetzerFocalLengthSameCameraCost, 2, 1>(
                new FetzerFocalLengthSameCameraCost(fundamental, principal_point));
    }

    void Evaluate(const double focal, double* residuals) const {
        (*this)(&focal, residuals);
    }

private:
    Eigen::Vector4d d01_;
    Eigen::Vector4d d12_;
};

bool ReadFundamental(const std::vector<double>& packed,
                     const size_t pair_index,
                     Eigen::Matrix3d* fundamental) {
    if (fundamental == nullptr) return false;
    const size_t offset = pair_index * 9;
    double max_abs = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double value = packed[offset + static_cast<size_t>(row * 3 + col)];
            if (!std::isfinite(value)) return false;
            (*fundamental)(row, col) = value;
            max_abs = std::max(max_abs, std::abs(value));
        }
    }
    if (!(max_abs > 0.0)) return false;
    const double norm = fundamental->norm();
    if (!std::isfinite(norm) || norm <= 0.0) return false;
    *fundamental /= norm;
    return fundamental->allFinite();
}

}  // namespace

bool RunViewGraphFocalCalibration(
        const std::vector<double>& fundamental_matrices,
        const double principal_x,
        const double principal_y,
        const double initial_focal_length,
        const int max_num_iterations,
        const int num_threads,
        ViewGraphCalibrationResult* result) {
    if (result == nullptr || fundamental_matrices.empty() ||
        fundamental_matrices.size() % 9 != 0 ||
        !std::isfinite(principal_x) || !std::isfinite(principal_y) ||
        !std::isfinite(initial_focal_length) || initial_focal_length <= 0.0 ||
        max_num_iterations <= 0 || num_threads <= 0) {
        return false;
    }

    const size_t pair_count = fundamental_matrices.size() / 9;
    std::vector<Eigen::Matrix3d> fundamentals;
    std::vector<size_t> source_indices;
    fundamentals.reserve(pair_count);
    source_indices.reserve(pair_count);
    for (size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
        Eigen::Matrix3d fundamental;
        if (!ReadFundamental(fundamental_matrices, pair_index, &fundamental)) {
            continue;
        }
        const Eigen::Vector3d singular_values =
                Eigen::JacobiSVD<Eigen::Matrix3d>(fundamental).singularValues();
        if (!singular_values.allFinite() || singular_values(0) <= 0.0 ||
            singular_values(1) <= singular_values(0) * kMinimumSingularValueRatio) {
            continue;
        }
        fundamentals.push_back(fundamental);
        source_indices.push_back(pair_index);
    }
    if (fundamentals.empty()) return false;

    const Eigen::Vector2d principal_point(principal_x, principal_y);
    std::vector<FetzerFocalLengthSameCameraCost> residual_models;
    residual_models.reserve(fundamentals.size());
    for (const Eigen::Matrix3d& fundamental : fundamentals) {
        residual_models.emplace_back(fundamental, principal_point);
    }

    double focal = initial_focal_length;
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::CauchyLoss loss(kCauchyLossScale);
    ceres::Problem problem(problem_options);
    for (size_t index = 0; index < residual_models.size(); ++index) {
        problem.AddResidualBlock(FetzerFocalLengthSameCameraCost::Create(
                                         fundamentals[index], principal_point),
                                 &loss, &focal);
    }
    problem.SetParameterLowerBound(&focal, 0, kMinimumFocal);

    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    solver_options.max_num_iterations = max_num_iterations;
    solver_options.num_threads = num_threads;
    solver_options.function_tolerance = 1e-5;
    solver_options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    if (!summary.IsSolutionUsable() || !std::isfinite(focal) || focal <= 0.0 ||
        !std::isfinite(summary.initial_cost) || !std::isfinite(summary.final_cost)) {
        return false;
    }

    ViewGraphCalibrationResult candidate;
    candidate.focal_length = focal;
    candidate.initial_cost = summary.initial_cost;
    candidate.final_cost = summary.final_cost;
    candidate.iterations = static_cast<int>(summary.iterations.size());
    candidate.pair_error_norms.assign(pair_count,
                                      std::numeric_limits<double>::infinity());
    candidate.pair_is_valid.assign(pair_count, 0);
    for (size_t local_index = 0; local_index < residual_models.size(); ++local_index) {
        double residuals[2] = {0.0, 0.0};
        residual_models[local_index].Evaluate(focal, residuals);
        if (!std::isfinite(residuals[0]) || !std::isfinite(residuals[1])) continue;
        const double error_norm = std::hypot(residuals[0], residuals[1]);
        const size_t source_index = source_indices[local_index];
        candidate.pair_error_norms[source_index] = error_norm;
        candidate.pair_is_valid[source_index] =
                error_norm <= kTwoViewErrorThreshold ? 1 : 0;
    }

    *result = std::move(candidate);
    return true;
}

}  // namespace tracking
