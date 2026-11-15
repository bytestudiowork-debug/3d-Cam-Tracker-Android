/*
Copyright (c), ETH Zurich and UNC Chapel Hill.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
      its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

#include "least_absolute_deviations_core.h"

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>

namespace tracking {
namespace {

Eigen::VectorXd Shrinkage(const Eigen::VectorXd& values, double threshold) {
    const Eigen::VectorXd plus = values.array() + threshold;
    const Eigen::VectorXd minus = values.array() - threshold;
    return plus.cwiseMin(0.0) + minus.cwiseMax(0.0);
}

}  // namespace

bool SolveLeastAbsoluteDeviations(const LeastAbsoluteDeviationsOptions& options,
                                  const Eigen::SparseMatrix<double>& matrix,
                                  const Eigen::VectorXd& observations,
                                  Eigen::VectorXd* solution) {
    if (solution == nullptr || matrix.rows() <= 0 || matrix.cols() <= 0 ||
        observations.size() != matrix.rows() || options.rho <= 0.0 ||
        options.alpha <= 0.0 || options.max_num_iterations <= 0 ||
        options.absolute_tolerance < 0.0 || options.relative_tolerance < 0.0 ||
        !std::isfinite(options.rho) || !std::isfinite(options.alpha) ||
        !std::isfinite(options.absolute_tolerance) ||
        !std::isfinite(options.relative_tolerance) || !observations.allFinite()) {
        return false;
    }
    for (Eigen::Index i = 0; i < matrix.nonZeros(); ++i) {
        if (!std::isfinite(matrix.valuePtr()[i])) return false;
    }

    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> linear_solver;
    linear_solver.compute(matrix.transpose() * matrix);
    if (linear_solver.info() != Eigen::Success) return false;

    Eigen::VectorXd z = Eigen::VectorXd::Zero(matrix.rows());
    Eigen::VectorXd z_old(matrix.rows());
    Eigen::VectorXd u = Eigen::VectorXd::Zero(matrix.rows());
    const double observations_norm = observations.norm();
    const double primal_absolute_threshold =
            std::sqrt(static_cast<double>(matrix.rows())) * options.absolute_tolerance;
    const double dual_absolute_threshold =
            std::sqrt(static_cast<double>(matrix.cols())) * options.absolute_tolerance;

    solution->resize(matrix.cols());
    for (int iteration = 0; iteration < options.max_num_iterations; ++iteration) {
        *solution = linear_solver.solve(
                matrix.transpose() * (observations + z - u));
        if (linear_solver.info() != Eigen::Success || !solution->allFinite()) {
            return false;
        }

        const Eigen::VectorXd matrix_times_solution = matrix * *solution;
        const Eigen::VectorXd relaxed_solution =
                options.alpha * matrix_times_solution +
                (1.0 - options.alpha) * (z + observations);

        z_old = z;
        z = Shrinkage(relaxed_solution - observations + u, 1.0 / options.rho);
        u += relaxed_solution - z - observations;

        const double primal_norm =
                (matrix_times_solution - z - observations).norm();
        const double dual_norm =
                (-options.rho * matrix.transpose() * (z - z_old)).norm();
        const double primal_tolerance =
                primal_absolute_threshold + options.relative_tolerance *
                    std::max(observations_norm,
                             std::max(matrix_times_solution.norm(), z.norm()));
        const double dual_tolerance =
                dual_absolute_threshold + options.relative_tolerance *
                    (options.rho * matrix.transpose() * u).norm();
        if (!std::isfinite(primal_norm) || !std::isfinite(dual_norm) ||
            !std::isfinite(primal_tolerance) || !std::isfinite(dual_tolerance)) {
            return false;
        }
        if (primal_norm < primal_tolerance && dual_norm < dual_tolerance) break;
    }

    return solution->allFinite();
}

}  // namespace tracking
