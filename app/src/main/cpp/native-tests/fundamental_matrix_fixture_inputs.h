#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
namespace tracking::fundamental_solver_test_fixtures {
struct Correspondences {
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector2d> points2;
};
enum class Geometry {
    kGeneral,
    kPlanar,
    kCollinear,
};
inline Eigen::Vector3d ScenePoint(const std::size_t index,
                                 const std::size_t count,
                                 const Geometry geometry) {
    if (geometry == Geometry::kCollinear) {
        const double t = count <= 1
                ? 0.0
                : -0.9 + 1.8 * static_cast<double>(index) /
                                  static_cast<double>(count - 1);
        return {0.4 + 0.9 * t, -0.3 + 0.4 * t, 5.0 + 0.6 * t};
    }
    const double i = static_cast<double>(index);
    const double x = 0.72 * std::sin(0.71 * i + 0.13) +
                     0.19 * std::cos(1.31 * i + 0.27) + 0.055 * i;
    const double y = 0.51 * std::cos(0.53 * i + 0.41) -
                     0.12 * std::sin(1.17 * i + 0.09) + 0.018 * i * i;
    if (geometry == Geometry::kPlanar) {
        return {x, y, 5.2 + 0.22 * x - 0.17 * y};
    }
    return {x, y, 5.0 + 0.31 * std::sin(0.37 * i + 0.2) + 0.08 * i};
}
inline Correspondences MakeCorrespondences(const std::size_t count,
                                            const Geometry geometry,
                                            const double noise_pixels = 0.0) {
    Correspondences result;
    result.points1.reserve(count);
    result.points2.reserve(count);
    const Eigen::Matrix3d rotation = Eigen::AngleAxisd(
            0.12, Eigen::Vector3d(0.1, 0.6, 0.2).normalized())
            .toRotationMatrix();
    const Eigen::Vector3d translation(0.72, -0.15, 0.1);
    for (std::size_t index = 0; index < count; ++index) {
        const Eigen::Vector3d point1 = ScenePoint(index, count, geometry);
        const Eigen::Vector3d point2 = rotation * point1 + translation;
        Eigen::Vector2d image1(710.0 * point1.x() / point1.z() + 320.0,
                               690.0 * point1.y() / point1.z() + 240.0);
        Eigen::Vector2d image2(690.0 * point2.x() / point2.z() + 310.0,
                               730.0 * point2.y() / point2.z() + 250.0);
        const double i = static_cast<double>(index);
        image2.x() += noise_pixels * std::sin(0.91 * i + 0.3);
        image2.y() += noise_pixels * std::cos(1.27 * i + 0.1);
        result.points1.push_back(image1);
        result.points2.push_back(image2);
    }
    return result;
}
}  // namespace tracking::fundamental_solver_test_fixtures
