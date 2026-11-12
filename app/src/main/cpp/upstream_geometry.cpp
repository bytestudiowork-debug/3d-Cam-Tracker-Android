// SPDX-License-Identifier: BSD-3-Clause
#include "colmap/geometry/triangulation.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
namespace colmap {
bool TriangulatePoint(const Eigen::Matrix3x4d& p1,const Eigen::Matrix3x4d& p2,const Eigen::Vector2d& x1,const Eigen::Vector2d& x2,Eigen::Vector3d* xyz){
 THROW_CHECK_NOTNULL(xyz);Eigen::Matrix4d A;
 A.row(0)=x1(0)*p1.row(2)-p1.row(0);A.row(1)=x1(1)*p1.row(2)-p1.row(1);
 A.row(2)=x2(0)*p2.row(2)-p2.row(0);A.row(3)=x2(1)*p2.row(2)-p2.row(1);
 const Eigen::JacobiSVD<Eigen::Matrix4d> svd(A,Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3,4,0)
 if(svd.info()!=Eigen::Success)return false;
#endif
 if(svd.matrixV()(3,3)==0)return false;*xyz=svd.matrixV().col(3).hnormalized();return true;
}
bool TriangulatePoint(const Eigen::Matrix3x4d& p1,const Eigen::Matrix3x4d& p2,const Eigen::Vector3d& b1,const Eigen::Vector3d& b2,Eigen::Vector3d* xyz){
 THROW_CHECK_NOTNULL(xyz);Eigen::Matrix<double,6,4>A;
 A.topRows<3>()=p1-b1*(b1.transpose()*p1);A.bottomRows<3>()=p2-b2*(b2.transpose()*p2);
 const Eigen::JacobiSVD<Eigen::Matrix<double,6,4>> svd(A,Eigen::ComputeFullV);
#if EIGEN_VERSION_AT_LEAST(3,4,0)
 if(svd.info()!=Eigen::Success)return false;
#endif
 if(svd.matrixV()(3,3)==0)return false;*xyz=svd.matrixV().col(3).hnormalized();return true;
}
namespace {
inline Eigen::Matrix4d TriangulationDltTerm(const Eigen::Matrix3x4d& p,const Eigen::Vector3d& b){
 const Eigen::Matrix3x4d term=p-b*b.transpose()*p;return term.transpose()*term;
}
inline bool SolveTriangulationDlt(const Eigen::Matrix4d& A,Eigen::Vector3d* xyz){
 const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(A);
 if(solver.info()!=Eigen::Success||solver.eigenvectors()(3,0)==0)return false;
 *xyz=solver.eigenvectors().col(0).hnormalized();return true;
}
}
bool TriangulateMultiViewPoint(const span<const Eigen::Matrix3x4d>& ps,const span<const Eigen::Vector2d>& xs,Eigen::Vector3d* xyz){
 THROW_CHECK_EQ(ps.size(),xs.size());THROW_CHECK_NOTNULL(xyz);Eigen::Matrix4d A=Eigen::Matrix4d::Zero();
 for(size_t i=0;i<xs.size();++i)A+=TriangulationDltTerm(ps[i],xs[i].homogeneous().normalized());
 return SolveTriangulationDlt(A,xyz);
}
bool TriangulateMultiViewPoint(const span<const Eigen::Matrix3x4d>& ps,const span<const Eigen::Vector3d>& rays,Eigen::Vector3d* xyz){
 THROW_CHECK_EQ(ps.size(),rays.size());THROW_CHECK_NOTNULL(xyz);Eigen::Matrix4d A=Eigen::Matrix4d::Zero();
 for(size_t i=0;i<rays.size();++i)A+=TriangulationDltTerm(ps[i],rays[i]);
 return SolveTriangulationDlt(A,xyz);
}
double CalculateTriangulationAngle(const Eigen::Vector3d& c1,const Eigen::Vector3d& c2,const Eigen::Vector3d& x){
 const double a=CalculateAngleBetweenVectors(x-c1,x-c2);return std::min(a,static_cast<double>(EIGEN_PI)-a);
}
double CalculateAngleBetweenVectors(const Eigen::Vector3d& a,const Eigen::Vector3d& b){
 const double n1=a.squaredNorm(),n2=b.squaredNorm(),prod=n1*n2;
 if(std::isnormal(prod)&&std::min(n1,n2)>=std::numeric_limits<double>::min()){
  const double cosine=a.dot(b)/std::sqrt(prod);constexpr double threshold=1.0-1e-8;
  if(std::abs(cosine)<threshold)return std::acos(cosine);
 }
 const double s1=a.cwiseAbs().maxCoeff(),s2=b.cwiseAbs().maxCoeff();if(s1==0||s2==0)return 0;
 const Eigen::Vector3d x=a/s1,y=b/s2;return std::atan2(x.cross(y).stableNorm(),x.dot(y));
}
}