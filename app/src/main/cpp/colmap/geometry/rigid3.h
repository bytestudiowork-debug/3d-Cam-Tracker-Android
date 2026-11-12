// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "colmap/util/types.h"
#include <Eigen/Geometry>
namespace colmap {
struct Rigid3d {
 Eigen::Quaterniond rotation_; Eigen::Vector3d translation_;
 Rigid3d():rotation_(Eigen::Quaterniond::Identity()),translation_(Eigen::Vector3d::Zero()){}
 Rigid3d(const Eigen::Quaterniond& r,const Eigen::Vector3d& t):rotation_(r),translation_(t){}
 Eigen::Matrix3x4d ToMatrix() const {Eigen::Matrix3x4d m;m.leftCols<3>()=rotation_.toRotationMatrix();m.col(3)=translation_;return m;}
 Eigen::Vector3d TgtOriginInSrc() const{return rotation_.inverse()*-translation_;}
};
inline Eigen::Vector3d operator*(const Rigid3d& t,const Eigen::Vector3d& x){return t.rotation_*x+t.translation_;}
}