// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Production source adaptation from pinned revision c5f9cefc87e5dd596b638e4cee0ff543c7d14755:
// src/estimators/essential_matrix.cc, its polynomial coefficient headers, and
// src/math/polynomial.cc. Minimal model candidates are capped at degree ten.
#include "two_view_epipolar_estimator_core.h"
#include <cmath>
#include <stdexcept>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
namespace tracking::source_two_view {
namespace {
// These coefficient-trimming and companion-matrix root routines follow
// the polynomial implementation in src/math/polynomial.cc at the pinned revision.
Eigen::VectorXd RemoveLeadingZeros(const Eigen::VectorXd& coeffs) {
    Eigen::VectorXd::Index num_zeros = 0;
    for (; num_zeros < coeffs.size(); ++num_zeros) {
        if (coeffs(num_zeros) != 0) {
            break;
        }
    }
    return coeffs.tail(coeffs.size() - num_zeros);
}
Eigen::VectorXd RemoveTrailingZeros(const Eigen::VectorXd& coeffs) {
    Eigen::VectorXd::Index num_zeros = 0;
    for (; num_zeros < coeffs.size(); ++num_zeros) {
        if (coeffs(coeffs.size() - 1 - num_zeros) != 0) {
            break;
        }
    }
    return coeffs.head(coeffs.size() - num_zeros);
}
bool FindLinearPolynomialRoots(const Eigen::VectorXd& coeffs,
                               Eigen::VectorXd* real,
                               Eigen::VectorXd* imag) {
    if (coeffs.size() != 2) {
        throw std::invalid_argument("linear polynomial requires 2 coefficients");
    }
    if (coeffs(0) == 0) {
        return false;
    }
    if (real != nullptr) {
        real->resize(1);
        (*real)(0) = -coeffs(1) / coeffs(0);
    }
    if (imag != nullptr) {
        imag->resize(1);
        (*imag)(0) = 0;
    }
    return true;
}
bool FindQuadraticPolynomialRoots(const Eigen::VectorXd& coeffs,
                                  Eigen::VectorXd* real,
                                  Eigen::VectorXd* imag) {
    if (coeffs.size() != 3) {
        throw std::invalid_argument("quadratic polynomial requires 3 coefficients");
    }
    const double a = coeffs(0);
    if (a == 0) {
        return FindLinearPolynomialRoots(coeffs.tail(2), real, imag);
    }
    const double b = coeffs(1);
    const double c = coeffs(2);
    if (b == 0 && c == 0) {
        if (real != nullptr) {
            real->resize(1);
            (*real)(0) = 0;
        }
        if (imag != nullptr) {
            imag->resize(1);
            (*imag)(0) = 0;
        }
        return true;
    }
    const double d = b * b - 4 * a * c;
    if (d >= 0) {
        const double sqrt_d = std::sqrt(d);
        if (real != nullptr) {
            real->resize(2);
            if (b >= 0) {
                (*real)(0) = (-b - sqrt_d) / (2 * a);
                (*real)(1) = (2 * c) / (-b - sqrt_d);
            } else {
                (*real)(0) = (2 * c) / (-b + sqrt_d);
                (*real)(1) = (-b + sqrt_d) / (2 * a);
            }
        }
        if (imag != nullptr) {
            imag->resize(2);
            imag->setZero();
        }
    } else {
        if (real != nullptr) {
            real->resize(2);
            real->setConstant(-b / (2 * a));
        }
        if (imag != nullptr) {
            imag->resize(2);
            (*imag)(0) = std::sqrt(-d) / (2 * a);
            (*imag)(1) = -(*imag)(0);
        }
    }
    return true;
}
bool FindPolynomialRootsCompanionMatrix(const Eigen::VectorXd& coeffs_all,
                                        Eigen::VectorXd* real,
                                        Eigen::VectorXd* imag) {
    if (coeffs_all.size() < 2) {
        throw std::invalid_argument("polynomial root input needs 2 coefficients");
    }
    Eigen::VectorXd coeffs = RemoveLeadingZeros(coeffs_all);
    const int degree = static_cast<int>(coeffs.size()) - 1;
    if (degree <= 0) {
        return false;
    } else if (degree == 1) {
        return FindLinearPolynomialRoots(coeffs, real, imag);
    } else if (degree == 2) {
        return FindQuadraticPolynomialRoots(coeffs, real, imag);
    }
    // Remove coefficients where zero is a solution.
    coeffs = RemoveTrailingZeros(coeffs);
    if (coeffs.size() == 1) {
        if (real != nullptr) {
            real->resize(1);
            (*real)(0) = 0;
        }
        if (imag != nullptr) {
            imag->resize(1);
            (*imag)(0) = 0;
        }
        return true;
    }
    Eigen::MatrixXd companion(coeffs.size() - 1, coeffs.size() - 1);
    companion.setZero();
    for (Eigen::MatrixXd::Index i = 1; i < companion.rows(); ++i) {
        companion(i, i - 1) = 1;
    }
    companion.row(0) = -coeffs.tail(coeffs.size() - 1) / coeffs(0);
    Eigen::EigenSolver<Eigen::MatrixXd> solver(companion, false);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    const int effective_degree =
            coeffs.size() - 1 < degree ? static_cast<int>(coeffs.size())
                                       : static_cast<int>(coeffs.size() - 1);
    if (real != nullptr) {
        real->resize(effective_degree);
        real->head(coeffs.size() - 1) = solver.eigenvalues().real();
        if (effective_degree > coeffs.size() - 1) {
            (*real)(real->size() - 1) = 0;
        }
    }
    if (imag != nullptr) {
        imag->resize(effective_degree);
        imag->head(coeffs.size() - 1) = solver.eigenvalues().imag();
        if (effective_degree > coeffs.size() - 1) {
            (*imag)(imag->size() - 1) = 0;
        }
    }
    return true;
}
}  // namespace
void EstimateEssentialFivePoint(
        const std::vector<Eigen::Vector3d>& cam_rays1,
        const std::vector<Eigen::Vector3d>& cam_rays2,
        std::vector<Eigen::Matrix3d>* models) {
    if (cam_rays1.size() != cam_rays2.size()) {
        throw std::invalid_argument("five-point ray vectors must have equal length");
    }
    if (cam_rays1.size() < 5) {
        throw std::invalid_argument("five-point estimator requires at least 5 rays");
    }
    if (models == nullptr) {
        throw std::invalid_argument("five-point model output must not be null");
    }
    models->clear();
    Eigen::Matrix<double, Eigen::Dynamic, 9> Q(cam_rays1.size(), 9);
    for (std::size_t i = 0; i < cam_rays1.size(); ++i) {
        Q.row(i) << cam_rays2[i].x() * cam_rays1[i].transpose(),
                    cam_rays2[i].y() * cam_rays1[i].transpose(),
                    cam_rays2[i].z() * cam_rays1[i].transpose();
    }
    Eigen::Matrix<double, 9, 4> E;
    if (cam_rays1.size() == 5) {
        E = Q.transpose().fullPivHouseholderQr().matrixQ().rightCols<4>();
    } else {
        const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
                Q, Eigen::ComputeFullV);
        E = svd.matrixV().rightCols<4>();
    }
    Eigen::Matrix<double, 10, 20> A;
#include "essential_matrix_poly.h"
    const Eigen::Matrix<double, 10, 10> AA =
            A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));
    Eigen::Matrix<double, 13, 3> B;
    for (std::size_t i = 0; i < 3; ++i) {
        B(0, i) = 0;
        B(4, i) = 0;
        B(8, i) = 0;
        B.block<3, 1>(1, i) = AA.block<1, 3>(i * 2 + 4, 0);
        B.block<3, 1>(5, i) = AA.block<1, 3>(i * 2 + 4, 3);
        B.block<4, 1>(9, i) = AA.block<1, 4>(i * 2 + 4, 6);
        B.block<3, 1>(0, i) -= AA.block<1, 3>(i * 2 + 5, 0);
        B.block<3, 1>(4, i) -= AA.block<1, 3>(i * 2 + 5, 3);
        B.block<4, 1>(8, i) -= AA.block<1, 4>(i * 2 + 5, 6);
    }
    Eigen::Matrix<double, 11, 1> coeffs;
#include "essential_matrix_coeffs.h"
    Eigen::VectorXd roots_real;
    Eigen::VectorXd roots_imag;
    if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
        return;
    }
    const int num_roots = static_cast<int>(roots_real.size());
    models->reserve(num_roots);
    for (int i = 0; i < num_roots; ++i) {
        constexpr double kMaxRootImag = 1e-10;
        if (std::abs(roots_imag(i)) > kMaxRootImag) {
            continue;
        }
        const double z1 = roots_real(i);
        const double z2 = z1 * z1;
        const double z3 = z2 * z1;
        const double z4 = z3 * z1;
        Eigen::Matrix3d Bz;
        for (int j = 0; j < 3; ++j) {
            Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
            Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
            Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 + B(12, j);
        }
        const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
        const Eigen::Vector3d X = svd.matrixV().rightCols<1>();
        constexpr double kMaxX3 = 1e-10;
        if (std::abs(X(2)) < kMaxX3) {
            continue;
        }
        const Eigen::Matrix<double, 9, 1> e =
                (E.col(0) * (X(0) / X(2)) + E.col(1) * (X(1) / X(2)) +
                 E.col(2) * z1 + E.col(3))
                        .normalized();
        models->push_back(
                Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
                        e.data()));
    }
}
}  // namespace tracking::source_two_view
