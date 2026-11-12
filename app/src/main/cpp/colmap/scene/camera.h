// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cmath>
#include <optional>
#include <Eigen/Core>
namespace colmap {
struct Camera {
 double fx=1.0,fy=1.0,skew=0.0,cx=0.0,cy=0.0;
 int width=0;
 bool IsPerspective() const{return true;} bool IsSpherical() const{return false;}
 std::optional<Eigen::Vector3d> CamRayFromImg(const Eigen::Vector2d& p) const {
  if(!std::isfinite(fx)||!std::isfinite(fy)||fx<=0||fy<=0)return std::nullopt;
  const double y=(p.y()-cy)/fy,x=(p.x()-cx-skew*y)/fx;Eigen::Vector3d ray(x,y,1.0);
  const double n=ray.norm();if(!std::isfinite(n)||n<=1e-12)return std::nullopt;return ray/n;
 }
 std::optional<Eigen::Vector2d> ImgFromCam(const Eigen::Vector3d& p,bool check=true) const {
  if(!p.allFinite()||(check&&p.z()<=1e-12)||std::abs(p.z())<=1e-12)return std::nullopt;
  const double x=p.x()/p.z(),y=p.y()/p.z();return Eigen::Vector2d(fx*x+skew*y+cx,fy*y+cy);
 }
};
}
