// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace trackstudio_upstream {

// Source-equivalent multi-view bearing DLT implementation.
inline Eigen::Matrix4d TriangulationDltTerm(
    const Eigen::Matrix<double, 3, 4>& cam_from_world,
    const Eigen::Vector3d& cam_ray) {
    const Eigen::Matrix<double, 3, 4> term =
        cam_from_world - cam_ray * cam_ray.transpose() * cam_from_world;
    return term.transpose() * term;
}

inline bool SolveTriangulationDlt(const Eigen::Matrix4d& matrix,
                                  Eigen::Vector3d* xyz) {
    if (xyz == nullptr) return false;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(matrix);
    if (solver.info() != Eigen::Success ||
        solver.eigenvectors()(3, 0) == 0.0) {
        return false;
    }
    *xyz = solver.eigenvectors().col(0).hnormalized();
    return xyz->allFinite();
}

inline bool TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix<double, 3, 4>>& cams_from_world,
    const std::vector<Eigen::Vector3d>& cam_rays,
    Eigen::Vector3d* xyz) {
    if (xyz == nullptr || cams_from_world.size() != cam_rays.size() ||
        cams_from_world.size() < 2) {
        return false;
    }
    Eigen::Matrix4d matrix = Eigen::Matrix4d::Zero();
    for (size_t i = 0; i < cam_rays.size(); ++i) {
        if (!cams_from_world[i].allFinite() || !cam_rays[i].allFinite() ||
            cam_rays[i].squaredNorm() <= 0.0) {
            return false;
        }
        matrix += TriangulationDltTerm(cams_from_world[i], cam_rays[i]);
    }
    return SolveTriangulationDlt(matrix, xyz);
}

}  // namespace trackstudio_upstream
