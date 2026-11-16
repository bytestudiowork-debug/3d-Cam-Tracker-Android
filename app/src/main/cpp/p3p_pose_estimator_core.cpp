// Copyright (c) 2020, Viktor Larsson. All rights reserved.
// Adapted from PoseLib PoseLib/solvers/p3p.cc, p3p_common.h, and the
// solve_cubic_single_real routine in PoseLib/misc/univariate.cc at commit
// f119951fca625133112acde48daffa5f20eba451 (the revision pinned by COLMAP
// c5f9cefc87e5dd596b638e4cee0ff543c7d14755). The P3P equations, branch order,
// cubic/quadratic root policy, five depth-refinement steps, and pose recovery
// follow those sources. Finite-input and degeneracy checks are Android safety
// guards. See app/src/main/assets/licenses/poselib-p3p-license.txt.

#include "p3p_pose_estimator_core.h"

#include <Eigen/Geometry>
#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace tracking {
namespace {

bool SolveCubicSingleReal(double c2, double c1, double c0, double& root) {
    const double a = c1 - c2 * c2 / 3.0;
    double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double discriminant = b * b / 4.0 + a * a * a / 27.0;
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(discriminant)) {
        return false;
    }
    if (discriminant != 0.0) {
        if (discriminant > 0.0) {
            discriminant = std::sqrt(discriminant);
            b *= -0.5;
            root = std::cbrt(b + discriminant) + std::cbrt(b - discriminant) - c2 / 3.0;
            return std::isfinite(root);
        }
        const double angle_argument = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
        if (!std::isfinite(angle_argument)) return false;
        root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(angle_argument) / 3.0) - c2 / 3.0;
    } else {
        root = -c2 / 3.0 + (a != 0.0 ? (3.0 * b / a) : 0.0);
    }
    return std::isfinite(root);
}

bool Root2Real(double b, double c, double& r1, double& r2) {
    constexpr double kThreshold = -1.0e-12;
    const double discriminant = b * b - 4.0 * c;
    if (!std::isfinite(discriminant)) return false;
    if (discriminant < kThreshold) {
        r1 = r2 = -0.5 * b;
        return false;
    }
    if (discriminant > kThreshold && discriminant < 0.0) {
        r1 = -0.5 * b;
        r2 = -2.0;
        return true;
    }
    const double y = std::sqrt(discriminant);
    if (b < 0.0) {
        r1 = 0.5 * (-b + y);
        r2 = 0.5 * (-b - y);
    } else {
        const double d1 = -b + y;
        const double d2 = -b - y;
        r1 = d1 != 0.0 ? 2.0 * c / d1 : 0.0;
        r2 = d2 != 0.0 ? 2.0 * c / d2 : 0.0;
    }
    return std::isfinite(r1) && std::isfinite(r2);
}

bool ComputePq(Eigen::Matrix3d C, std::array<Eigen::Vector3d, 2>* pq) {
    if (pq == nullptr || !C.allFinite()) return false;
    Eigen::Matrix3d C_adj;
    C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
    C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
    C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
    C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
    C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
    C_adj(1, 0) = C_adj(0, 1);
    C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
    C_adj(2, 0) = C_adj(0, 2);
    C_adj(2, 1) = C_adj(1, 2);
    if (!C_adj.allFinite()) return false;

    Eigen::Vector3d v;
    if (C_adj(0, 0) > C_adj(1, 1)) {
        if (C_adj(0, 0) > C_adj(2, 2)) {
            if (C_adj(0, 0) <= 0.0) return false;
            v = C_adj.col(0) / std::sqrt(C_adj(0, 0));
        } else {
            if (C_adj(2, 2) <= 0.0) return false;
            v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
        }
    } else if (C_adj(1, 1) > C_adj(2, 2)) {
        if (C_adj(1, 1) <= 0.0) return false;
        v = C_adj.col(1) / std::sqrt(C_adj(1, 1));
    } else {
        if (C_adj(2, 2) <= 0.0) return false;
        v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
    }
    if (!v.allFinite()) return false;

    C(0, 1) -= v(2);
    C(0, 2) += v(1);
    C(1, 2) -= v(0);
    C(1, 0) += v(2);
    C(2, 0) -= v(1);
    C(2, 1) += v(0);
    (*pq)[0] = C.col(0);
    (*pq)[1] = C.row(0).transpose();
    return (*pq)[0].allFinite() && (*pq)[1].allFinite();
}

void RefineLambda(double& lambda1,
                  double& lambda2,
                  double& lambda3,
                  double a12,
                  double a13,
                  double a23,
                  double b12,
                  double b13,
                  double b23) {
    for (int iter = 0; iter < 5; ++iter) {
        const double r1 = lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 +
                          lambda2 * lambda2 - a12;
        const double r2 = lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 +
                          lambda3 * lambda3 - a13;
        const double r3 = lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 +
                          lambda3 * lambda3 - a23;
        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-10) return;
        const double x11 = lambda1 - lambda2 * b12;
        const double x12 = lambda2 - lambda1 * b12;
        const double x21 = lambda1 - lambda3 * b13;
        const double x23 = lambda3 - lambda1 * b13;
        const double x32 = lambda2 - lambda3 * b23;
        const double x33 = lambda3 - lambda2 * b23;
        const double denominator = x11 * x23 * x32 + x12 * x21 * x33;
        if (!std::isfinite(denominator) || std::abs(denominator) <=
                std::numeric_limits<double>::min()) return;
        const double detJ = 0.5 / denominator;
        lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
        if (!std::isfinite(lambda1) || !std::isfinite(lambda2) || !std::isfinite(lambda3)) return;
    }
}

bool AddPose(const Eigen::Vector3d& ray0,
             const Eigen::Vector3d& world0,
             const Eigen::Vector3d& ray1,
             const Eigen::Vector3d& world1,
             const Eigen::Vector3d& ray2,
             const Eigen::Vector3d& world2,
             const Eigen::Matrix3d& world_basis_inverse,
             double d0,
             double d1,
             double d2,
             std::vector<P3PPoseEstimate>* candidates) {
    if (!(d0 > 0.0 && d1 > 0.0 && d2 > 0.0) ||
        !std::isfinite(d0) || !std::isfinite(d1) || !std::isfinite(d2)) {
        return false;
    }
    const Eigen::Vector3d v1 = d0 * ray0 - d1 * ray1;
    const Eigen::Vector3d v2 = d0 * ray0 - d2 * ray2;
    Eigen::Matrix3d camera_basis;
    camera_basis << v1, v2, v1.cross(v2);
    const Eigen::Matrix3d raw_rotation = camera_basis * world_basis_inverse;
    if (!raw_rotation.allFinite() || raw_rotation.determinant() <= 0.0) return false;

    // PoseLib stores each solution in a unit quaternion and returns Rt(); this
    // equivalent round-trip keeps the same world-to-camera rotation convention.
    Eigen::Quaterniond quaternion(raw_rotation);
    const double quaternion_norm = quaternion.norm();
    if (!std::isfinite(quaternion_norm) || quaternion_norm <= 0.0) return false;
    quaternion.normalize();

    P3PPoseEstimate pose;
    pose.rotation = quaternion.toRotationMatrix();
    pose.translation = d0 * ray0 - pose.rotation * world0;
    if (!pose.rotation.allFinite() || !pose.translation.allFinite() ||
        pose.rotation.determinant() <= 0.0) {
        return false;
    }
    candidates->push_back(std::move(pose));
    return true;
}

}  // namespace

bool EstimateP3PPoseCandidates(const std::vector<Eigen::Vector3d>& camera_rays,
                                const std::vector<Eigen::Vector3d>& world_points,
                                std::vector<P3PPoseEstimate>* candidates) {
    if (candidates == nullptr) return false;
    candidates->clear();
    if (camera_rays.size() != 3 || world_points.size() != 3) return false;

    std::array<Eigen::Vector3d, 3> X = {world_points[0], world_points[1], world_points[2]};
    std::array<Eigen::Vector3d, 3> x = {camera_rays[0], camera_rays[1], camera_rays[2]};
    for (int i = 0; i < 3; ++i) {
        if (!X[i].allFinite() || !x[i].allFinite()) return false;
        const double norm = x[i].norm();
        if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::min()) return false;
        x[i] /= norm;
    }

    Eigen::Vector3d X01 = X[0] - X[1];
    Eigen::Vector3d X02 = X[0] - X[2];
    Eigen::Vector3d X12 = X[1] - X[2];
    double a01 = X01.squaredNorm();
    double a02 = X02.squaredNorm();
    double a12 = X12.squaredNorm();
    if (!std::isfinite(a01) || !std::isfinite(a02) || !std::isfinite(a12)) return false;

    // The pinned solver switches so that the BC baseline is the largest.
    if (a01 > a02) {
        if (a01 > a12) {
            std::swap(x[0], x[2]);
            std::swap(X[0], X[2]);
            std::swap(a01, a12);
            X01 = -X12;
            X02 = -X02;
        }
    } else if (a02 > a12) {
        std::swap(x[0], x[1]);
        std::swap(X[0], X[1]);
        std::swap(a02, a12);
        X01 = -X01;
        X02 = X12;
    }
    if (!(a12 > std::numeric_limits<double>::min())) return false;

    Eigen::Matrix3d world_basis;
    world_basis << X01, X02, X01.cross(X02);
    const double basis_det = world_basis.determinant();
    if (!world_basis.allFinite() || !std::isfinite(basis_det) ||
        std::abs(basis_det) <= std::numeric_limits<double>::min()) {
        return false;
    }
    const Eigen::Matrix3d world_basis_inverse = world_basis.inverse();
    if (!world_basis_inverse.allFinite()) return false;

    const double a12_inverse = 1.0 / a12;
    const double a = a01 * a12_inverse;
    const double b = a02 * a12_inverse;
    const double m01 = x[0].dot(x[1]);
    const double m02 = x[0].dot(x[2]);
    const double m12 = x[1].dot(x[2]);
    const double m12sq = -m12 * m12 + 1.0;
    const double m02sq = -1.0 + m02 * m02;
    const double m01sq = -1.0 + m01 * m01;
    const double ab = a * b;
    const double bsq = b * b;
    const double asq = a * a;
    const double m013 = -2.0 + 2.0 * m01 * m02 * m12;
    const double bsqm12sq = bsq * m12sq;
    const double asqm12sq = asq * m12sq;
    const double abm12sq = 2.0 * ab * m12sq;
    const double k3_denominator = bsqm12sq + b * m02sq;
    if (!std::isfinite(k3_denominator) ||
        std::abs(k3_denominator) <= std::numeric_limits<double>::min()) return false;
    const double k3_inverse = 1.0 / k3_denominator;
    const double k2 = k3_inverse * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
    const double k1 = k3_inverse * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
    const double k0 = k3_inverse * (asqm12sq + a * m01sq);
    double s = 0.0;
    const bool single_real_root = SolveCubicSingleReal(k2, k1, k0, s);
    if (!std::isfinite(s)) return false;

    Eigen::Matrix3d C;
    C(0, 0) = -a + s * (1.0 - b);
    C(0, 1) = -m02 * s;
    C(0, 2) = a * m12 + b * m12 * s;
    C(1, 0) = C(0, 1);
    C(1, 1) = s + 1.0;
    C(1, 2) = -m01;
    C(2, 0) = C(0, 2);
    C(2, 1) = C(1, 2);
    C(2, 2) = -a - b * s + 1.0;
    std::array<Eigen::Vector3d, 2> pq;
    if (!ComputePq(C, &pq)) return false;

    int num_solutions = 0;
    for (int branch = 0; branch < 2; ++branch) {
        const double p0 = pq[branch](0);
        const double p1 = pq[branch](1);
        const double p2 = pq[branch](2);
        if (!std::isfinite(p0) || !std::isfinite(p1) || !std::isfinite(p2)) continue;
        const bool switch_12 = std::abs(p0) <= std::abs(p1);
        if (switch_12) {
            if (std::abs(p1) <= std::numeric_limits<double>::min()) continue;
            const double w0 = -p0 / p1;
            const double w1 = -p2 / p1;
            const double ca_denominator = w1 * w1 - b;
            if (!std::isfinite(ca_denominator) ||
                std::abs(ca_denominator) <= std::numeric_limits<double>::min()) continue;
            const double ca = 1.0 / ca_denominator;
            const double cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
            const double cc = (w0 * w0 - 2.0 * m02 * w0 - b + 1.0) * ca;
            double taus[2] = {};
            if (!Root2Real(cb, cc, taus[0], taus[1])) continue;
            for (double tau : taus) {
                if (!(tau > 0.0)) continue;
                const double depth_denominator = tau * (tau - 2.0 * m12) + 1.0;
                if (!(depth_denominator > 0.0) || !std::isfinite(depth_denominator)) continue;
                double d2 = std::sqrt(a12 / depth_denominator);
                double d1 = tau * d2;
                double d0 = w0 * d2 + w1 * d1;
                if (!(d0 >= 0.0)) continue;
                RefineLambda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
                if (AddPose(x[0], X[0], x[1], X[1], x[2], X[2], world_basis_inverse,
                            d0, d1, d2, candidates)) {
                    ++num_solutions;
                }
            }
        } else {
            if (std::abs(p0) <= std::numeric_limits<double>::min()) continue;
            const double w0 = -p1 / p0;
            const double w1 = -p2 / p0;
            const double ca_denominator = -a * w1 * w1 + 2.0 * a * m12 * w1 - a + 1.0;
            if (!std::isfinite(ca_denominator) ||
                std::abs(ca_denominator) <= std::numeric_limits<double>::min()) continue;
            const double ca = 1.0 / ca_denominator;
            const double cb = 2.0 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
            const double cc = (1.0 - a * w0 * w0) * ca;
            double taus[2] = {};
            if (!Root2Real(cb, cc, taus[0], taus[1])) continue;
            for (double tau : taus) {
                if (!(tau > 0.0)) continue;
                const double depth_denominator = tau * (tau - 2.0 * m01) + 1.0;
                if (!(depth_denominator > 0.0) || !std::isfinite(depth_denominator)) continue;
                double d0 = std::sqrt(a01 / depth_denominator);
                double d1 = tau * d0;
                double d2 = w0 * d0 + w1 * d1;
                if (!(d2 >= 0.0)) continue;
                RefineLambda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
                if (AddPose(x[0], X[0], x[1], X[1], x[2], X[2], world_basis_inverse,
                            d0, d1, d2, candidates)) {
                    ++num_solutions;
                }
            }
        }
        if (num_solutions > 0 && single_real_root) break;
    }
    return !candidates->empty();
}

}  // namespace tracking
