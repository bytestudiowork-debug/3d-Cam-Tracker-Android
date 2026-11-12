// SPDX-License-Identifier: BSD-3-Clause
#include <jni.h>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "colmap/estimators/triangulation.h"
namespace { jdoubleArray EmptyResult(JNIEnv* e){return e->NewDoubleArray(0);} }
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_UpstreamTriangulation_nativeEstimateRobust(
 JNIEnv* env,jobject,jdoubleArray pointArray,jdoubleArray poseArray,jdoubleArray kArray,
 jdouble minAngleDeg,jdouble maxErrorPx,jint maxTrials){
 if(!pointArray||!poseArray||!kArray||!std::isfinite(minAngleDeg)||!std::isfinite(maxErrorPx)||maxErrorPx<=0||maxTrials<1)return EmptyResult(env);
 const jsize plen=env->GetArrayLength(pointArray),tlen=env->GetArrayLength(poseArray);
 if(plen<4||plen%2||env->GetArrayLength(kArray)!=5)return EmptyResult(env);
 const jsize n=plen/2;if(n<2||n>256||tlen!=n*12)return EmptyResult(env);
 std::vector<jdouble> pv(static_cast<size_t>(plen)),tv(static_cast<size_t>(tlen));jdouble k[5];
 env->GetDoubleArrayRegion(pointArray,0,plen,pv.data());env->GetDoubleArrayRegion(poseArray,0,tlen,tv.data());env->GetDoubleArrayRegion(kArray,0,5,k);
 if(env->ExceptionCheck())return EmptyResult(env);
 for(double v:pv)if(!std::isfinite(v))return EmptyResult(env);
 for(double v:tv)if(!std::isfinite(v))return EmptyResult(env);
 for(double v:k)if(!std::isfinite(v))return EmptyResult(env);
 if(k[0]<=1e-12||k[1]<=1e-12)return EmptyResult(env);
 colmap::Camera camera{k[0],k[1],k[2],k[3],k[4]};
 std::vector<Eigen::Vector2d> points(static_cast<size_t>(n));
 std::vector<colmap::Rigid3d> poses;poses.reserve(static_cast<size_t>(n));
 std::vector<colmap::Camera const*> cameras(static_cast<size_t>(n),&camera);
 for(jsize i=0;i<n;++i){
  points[static_cast<size_t>(i)]=Eigen::Vector2d(pv[static_cast<size_t>(i*2)],pv[static_cast<size_t>(i*2+1)]);
  Eigen::Matrix3x4d p;const size_t o=static_cast<size_t>(i)*12;
  for(int r=0;r<3;++r)for(int c=0;c<4;++c)p(r,c)=tv[o+static_cast<size_t>(r*4+c)];
  poses.emplace_back(Eigen::Quaterniond(p.leftCols<3>()),p.col(3));
 }
 colmap::EstimateTriangulationOptions options;
 options.min_tri_angle=minAngleDeg*0.017453292519943295;
 options.residual_type=colmap::TriangulationEstimator::ResidualType::REPROJECTION_ERROR;
 options.ransac_options.max_error=maxErrorPx;
 options.ransac_options.max_num_trials=static_cast<size_t>(maxTrials);
 options.ransac_options.min_inlier_ratio=0.02;
 options.ransac_options.num_threads=1;
 std::vector<char> mask;Eigen::Vector3d xyz;bool ok=false;
 try{ok=colmap::EstimateTriangulation(options,points,poses,cameras,&mask,&xyz);}catch(...){return EmptyResult(env);}
 if(!ok||mask.size()!=static_cast<size_t>(n)||!xyz.allFinite())return EmptyResult(env);
 jdoubleArray result=env->NewDoubleArray(3+n);if(!result)return nullptr;
 std::vector<jdouble> out(static_cast<size_t>(3+n));out[0]=xyz.x();out[1]=xyz.y();out[2]=xyz.z();
 for(jsize i=0;i<n;++i)out[static_cast<size_t>(i+3)]=mask[static_cast<size_t>(i)]?1.0:0.0;
 env->SetDoubleArrayRegion(result,0,static_cast<jsize>(out.size()),out.data());return result;
}