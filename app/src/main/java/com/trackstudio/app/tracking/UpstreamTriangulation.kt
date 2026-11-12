package com.trackstudio.app.tracking
internal data class RobustTriangulationEstimate(val point: DoubleArray, val inlierMask: BooleanArray)
internal object UpstreamTriangulation {
 init { System.loadLibrary("trackstudio_reconstruction") }
 private external fun nativeTriangulateMultiView(projections:DoubleArray,cameraRays:DoubleArray,viewCount:Int):DoubleArray
 private external fun nativeTriangulate(firstProjection:DoubleArray,secondProjection:DoubleArray,firstX:Double,firstY:Double,secondX:Double,secondY:Double):DoubleArray
 private external fun nativeEstimateRobust(points:DoubleArray,poseMatrices:DoubleArray,intrinsics:DoubleArray,minimumAngleDegrees:Double,maximumReprojectionErrorPx:Double,maximumTrials:Int):DoubleArray
 fun estimateRobust(points:DoubleArray,poses:DoubleArray,k:DoubleArray,minAngle:Double,maxError:Double,maxTrials:Int=10000):RobustTriangulationEstimate?{
  if(points.size<4||points.size%2!=0)return null
  val n=points.size/2
  if(n<2||poses.size!=n*12||k.size!=5||!points.all(Double::isFinite)||!poses.all(Double::isFinite)||!k.all(Double::isFinite)||k[0]<=1e-12||k[1]<=1e-12||!minAngle.isFinite()||minAngle<0||!maxError.isFinite()||maxError<=0||maxTrials<1)return null
  val r=nativeEstimateRobust(points,poses,k,minAngle,maxError,maxTrials)
  if(r.size!=n+3||!r.all(Double::isFinite))return null
  return RobustTriangulationEstimate(r.copyOfRange(0,3),BooleanArray(n){r[it+3]>0.5})
 }
 fun triangulateMultiView(p:DoubleArray,r:DoubleArray,n:Int):DoubleArray?{
  if(n<2||p.size!=n*12||r.size!=n*3||!p.all(Double::isFinite)||!r.all(Double::isFinite))return null
  for(i in 0 until n){val o=i*3;val norm=r[o]*r[o]+r[o+1]*r[o+1]+r[o+2]*r[o+2];if(!norm.isFinite()||norm<=1e-12)return null}
  return nativeTriangulateMultiView(p,r,n).takeIf{it.size==3&&it.all(Double::isFinite)}
 }
 fun triangulate(a:DoubleArray,b:DoubleArray,x1:Double,y1:Double,x2:Double,y2:Double):DoubleArray?{
  if(a.size!=12||b.size!=12||!a.all(Double::isFinite)||!b.all(Double::isFinite)||!x1.isFinite()||!y1.isFinite()||!x2.isFinite()||!y2.isFinite())return null
  return nativeTriangulate(a,b,x1,y1,x2,y2).takeIf{it.size==3&&it.all(Double::isFinite)}
 }
}