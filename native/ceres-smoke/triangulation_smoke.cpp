#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Core>

#include "upstream_triangulation_core.h"

int main() {
    const Eigen::Vector3d expected(0.2, -0.1, 4.0);
    const std::vector<Eigen::Vector3d> centers = {
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.8, 0.0),
        Eigen::Vector3d(-0.6, -0.4, 0.0)
    };

    std::vector<Eigen::Matrix<double, 3, 4>> projections;
    std::vector<Eigen::Vector3d> rays;
    for (const auto& center : centers) {
        Eigen::Matrix<double, 3, 4> projection;
        projection.setZero();
        projection(0, 0) = 1.0;
        projection(1, 1) = 1.0;
        projection(2, 2) = 1.0;
        projection(0, 3) = -center.x();
        projection(1, 3) = -center.y();
        projection(2, 3) = -center.z();
        projections.push_back(projection);
        rays.push_back((expected - center).normalized());
    }

    Eigen::Vector3d actual;
    if (!trackstudio_upstream::TriangulateMultiViewPoint(
            projections, rays, &actual)) {
        std::cerr << "Multi-view triangulation returned failure.\n";
        return 1;
    }
    const double error = (actual - expected).norm();
    if (!std::isfinite(error) || error > 1e-8) {
        std::cerr << "Multi-view triangulation error: " << error << "\n";
        return 2;
    }

    std::cout << "Multi-view triangulation smoke passed; error="
              << error << "\n";
    return 0;
}
