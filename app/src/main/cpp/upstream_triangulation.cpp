// SPDX-License-Identifier: BSD-3-Clause
#include <jni.h>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "colmap/geometry/triangulation.h"
namespace { jdoubleArray EmptyResult(JNIEnv* e){return e->NewDoubleArray(0);} }
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_UpstreamTriangulation_nativeTriangulate(
 JNIEnv* env,jobject,jdoubleArray a,jdoubleArray b,jdouble x1,jdouble y1,jdouble x2,jdouble y2){
 if(!a||!b||env->GetArrayLength(a)!=12||env->GetArrayLength(b)!=12||
 !std::isfinite(x1)||!std::isfinite(y1)||!std::isfinite(x2)||!std::isfinite(y2))return EmptyResult(env);
 Eigen::Matrix3x4d p1,p2;env->GetDoubleArrayRegion(a,0,12,p1.data());env->GetDoubleArrayRegion(b,0,12,p2.data());
 if(env->ExceptionCheck()||!p1.allFinite()||!p2.allFinite())return EmptyResult(env);
 Eigen::Vector3d xyz;if(!colmap::TriangulatePoint(p1,p2,Eigen::Vector2d(x1,y1),Eigen::Vector2d(x2,y2),&xyz))return EmptyResult(env);
 jdoubleArray out=env->NewDoubleArray(3);if(out){const jdouble v[3]={xyz.x(),xyz.y(),xyz.z()};env->SetDoubleArrayRegion(out,0,3,v);}return out;
}
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_trackstudio_app_tracking_UpstreamTriangulation_nativeTriangulateMultiView(
 JNIEnv* env,jobject,jdoubleArray pa,jdoubleArray ra,jint n){
 if(!pa||!ra||n<2||n>100000||env->GetArrayLength(pa)!=n*12||env->GetArrayLength(ra)!=n*3)return EmptyResult(env);
 std::vector<Eigen::Matrix3x4d> ps(static_cast<size_t>(n));std::vector<Eigen::Vector3d> rays(static_cast<size_t>(n));
 for(jint i=0;i<n;++i){jdouble p[12],r[3];env->GetDoubleArrayRegion(pa,i*12,12,p);env->GetDoubleArrayRegion(ra,i*3,3,r);if(env->ExceptionCheck())return EmptyResult(env);
  for(int y=0;y<3;++y)for(int x=0;x<4;++x)ps[static_cast<size_t>(i)](y,x)=p[y*4+x];
  rays[static_cast<size_t>(i)]=Eigen::Vector3d(r[0],r[1],r[2]);
  if(!ps[static_cast<size_t>(i)].allFinite()||!rays[static_cast<size_t>(i)].allFinite()||rays[static_cast<size_t>(i)].squaredNorm()<=0)return EmptyResult(env);
 }
 colmap::span<const Eigen::Matrix3x4d> pspan(ps.data(),ps.size());colmap::span<const Eigen::Vector3d> rspan(rays.data(),rays.size());
 Eigen::Vector3d xyz;if(!colmap::TriangulateMultiViewPoint(pspan,rspan,&xyz))return EmptyResult(env);
 jdoubleArray out=env->NewDoubleArray(3);if(out){const jdouble v[3]={xyz.x(),xyz.y(),xyz.z()};env->SetDoubleArrayRegion(out,0,3,v);}return out;
}