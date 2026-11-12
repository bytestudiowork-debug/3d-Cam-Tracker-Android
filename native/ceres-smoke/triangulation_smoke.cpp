#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "colmap/estimators/triangulation.h"
int main(){
 colmap::Camera camera{800,800,0,640,360};Eigen::Vector3d expected(0.2,-0.1,4.0);
 std::vector<Eigen::Vector3d> centers={{0,0,0},{1,0,0},{0,0.8,0},{-0.6,-0.4,0},{0.5,-0.7,0}};
 std::vector<Eigen::Vector2d> points;std::vector<colmap::Rigid3d> poses;std::vector<colmap::Camera const*> cameras(centers.size(),&camera);
 for(const auto& c:centers){poses.emplace_back(Eigen::Quaterniond::Identity(),-c);auto pixel=camera.ImgFromCam(expected-c);if(!pixel)return 1;points.push_back(*pixel);}
 points.back()+=Eigen::Vector2d(80,-60);
 colmap::EstimateTriangulationOptions options;options.min_tri_angle=colmap::DegToRad(1.0);options.residual_type=colmap::TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
 options.ransac_options.max_error=2.0;options.ransac_options.min_inlier_ratio=0.02;options.ransac_options.max_num_trials=1000;options.ransac_options.num_threads=1;
 std::vector<char> mask;Eigen::Vector3d actual;
 if(!colmap::EstimateTriangulation(options,points,poses,cameras,&mask,&actual)){std::cerr<<"Robust triangulation failed\n";return 2;}
 const double error=(actual-expected).norm();const size_t inliers=std::count(mask.begin(),mask.end(),char(1));
 if(!std::isfinite(error)||error>1e-5||inliers!=4){std::cerr<<"Mismatch "<<error<<" inliers "<<inliers<<"\n";return 3;}
 std::cout<<"Robust triangulation passed; error="<<error<<" inliers="<<inliers<<"\n";return 0;
}