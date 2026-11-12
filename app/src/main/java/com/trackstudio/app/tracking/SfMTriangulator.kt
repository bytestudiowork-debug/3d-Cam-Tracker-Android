package com.trackstudio.app.tracking
import org.opencv.core.Mat
import org.opencv.core.Point3
import kotlin.math.acos
import kotlin.math.sqrt
internal object SfMTriangulator {
 fun triangulate(featureTracks:List<CameraFeatureTrack>,poses:List<CameraPose>,cameraMatrix:Mat,
  maximumReprojectionErrorPx:Double=4.0,minimumParallaxDegrees:Double=1.0,
  minimumObservationInlierRatio:Double=0.70,onProgress:(Int,Int)->Unit={_,_->}):TriangulationResult{
  require(cameraMatrix.rows()==3&&cameraMatrix.cols()==3&&!cameraMatrix.empty()){ "Camera matrix must be a non-empty 3x3 matrix." }
  val poseByFrame=poses.associateBy{it.frameIndex}
  val fx=cameraMatrix.get(0,0)[0];val skew=cameraMatrix.get(0,1)[0];val cx=cameraMatrix.get(0,2)[0]
  val fy=cameraMatrix.get(1,1)[0];val cy=cameraMatrix.get(1,2)[0]
  val k=doubleArrayOf(fx,fy,skew,cx,cy)
  val landmarks=mutableListOf<CameraLandmark>()
  var rejectedShort=0;var rejectedMissingPose=0;var rejectedGeometry=0;var rejectedReprojection=0
  for((trackIndex,track) in featureTracks.withIndex()){
   if(trackIndex%16==0||trackIndex==featureTracks.lastIndex)onProgress(trackIndex+1,featureTracks.size)
   if(track.observations.size<2){rejectedShort++;continue}
   val valid=track.observations.mapNotNull{o->poseByFrame[o.frameIndex]?.takeIf{
    it.status!=CameraPoseStatus.HELD_LAST_VALID&&it.status!=CameraPoseStatus.UNREGISTERED
   }?.let{p->o to p}}
   if(valid.size<2){rejectedMissingPose++;continue}
   val candidates=sample(valid,32);var bestAngle=-1.0
   for(i in 0 until candidates.lastIndex)for(j in i+1 until candidates.size){
    val a=candidates[i];val b=candidates[j]
    if(centerDistanceSquared(a.second,b.second)<1e-10)continue
    bestAngle=maxOf(bestAngle,viewAngle(a.first,a.second,b.first,b.second,cameraMatrix))
   }
   if(bestAngle<minimumParallaxDegrees||bestAngle>=179.0){rejectedGeometry++;continue}
   val pointValues=DoubleArray(candidates.size*2);val poseValues=DoubleArray(candidates.size*12)
   candidates.forEachIndexed{index,(observation,pose)->
    pointValues[index*2]=observation.x;pointValues[index*2+1]=observation.y
    val offset=index*12
    for(row in 0..2){for(column in 0..2)poseValues[offset+row*4+column]=pose.rotation[row*3+column]
     poseValues[offset+row*4+3]=pose.translation[row]}
   }
   val robust=UpstreamTriangulation.estimateRobust(pointValues,poseValues,k,minimumParallaxDegrees,maximumReprojectionErrorPx)
   if(robust==null){rejectedGeometry++;continue}
   val sampleRatio=robust.inlierMask.count{it}.toDouble()/robust.inlierMask.size
   if(sampleRatio<minimumObservationInlierRatio){rejectedReprojection++;continue}
   val xyz=robust.point
   val errors=valid.mapNotNull{(observation,pose)->
    val p=transform(pose,xyz);if(p[2]<=1e-6)return@mapNotNull null
    val e=reprojectionError(cameraMatrix,p,observation);e.takeIf{it.isFinite()&&it<=maximumReprojectionErrorPx}
   }
   if(errors.size<2||errors.size.toDouble()/valid.size<minimumObservationInlierRatio){rejectedReprojection++;continue}
   val medianError=median(errors);if(!medianError.isFinite()||medianError>maximumReprojectionErrorPx){rejectedReprojection++;continue}
   landmarks+=CameraLandmark(track.id,Point3(xyz[0],xyz[1],xyz[2]),errors.size,errors.average(),bestAngle)
  }
  return TriangulationResult(landmarks,TriangulationStats(
   inputTracks=featureTracks.size,triangulatedLandmarks=landmarks.size,
   rejectedShortTracks=rejectedShort,rejectedMissingPoses=rejectedMissingPose,
   rejectedInvalidGeometry=rejectedGeometry,rejectedReprojection=rejectedReprojection,
   meanReprojectionErrorPx=if(landmarks.isEmpty())0.0 else landmarks.map{it.meanReprojectionErrorPx}.average(),
   meanTriangulationAngleDegrees=if(landmarks.isEmpty())0.0 else landmarks.map{it.triangulationAngleDegrees}.average()))
 }
 private fun sample(v:List<Pair<CameraFeatureObservation,CameraPose>>,m:Int)=if(v.size<=m)v else (0 until m).map{i->v[i*(v.size-1)/(m-1)]}
 private fun median(v:List<Double>):Double{val s=v.sorted();val m=s.size/2;return if(s.size%2==0)(s[m-1]+s[m])*0.5 else s[m]}
 private fun viewAngle(a:CameraFeatureObservation,pa:CameraPose,b:CameraFeatureObservation,pb:CameraPose,k:Mat):Double{
  val fx=k.get(0,0)[0];val sk=k.get(0,1)[0];val cx=k.get(0,2)[0];val fy=k.get(1,1)[0];val cy=k.get(1,2)[0]
  if(kotlin.math.abs(fx)<1e-12||kotlin.math.abs(fy)<1e-12)return 0.0
  fun ray(o:CameraFeatureObservation,p:CameraPose):DoubleArray{
   val y=(o.y-cy)/fy;val x=(o.x-cx-sk*y)/fx;val n=sqrt(x*x+y*y+1.0);val c=doubleArrayOf(x/n,y/n,1.0/n);val r=p.rotation
   val w=doubleArrayOf(r[0]*c[0]+r[3]*c[1]+r[6]*c[2],r[1]*c[0]+r[4]*c[1]+r[7]*c[2],r[2]*c[0]+r[5]*c[1]+r[8]*c[2])
   val z=sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]);return doubleArrayOf(w[0]/z,w[1]/z,w[2]/z)
  }
  val x=ray(a,pa);val y=ray(b,pb);return Math.toDegrees(acos((x[0]*y[0]+x[1]*y[1]+x[2]*y[2]).coerceIn(-1.0,1.0)))
 }
 private fun centerDistanceSquared(a:CameraPose,b:CameraPose):Double{
  fun c(p:CameraPose):DoubleArray{val r=p.rotation;val t=p.translation;return doubleArrayOf(-(r[0]*t[0]+r[3]*t[1]+r[6]*t[2]),-(r[1]*t[0]+r[4]*t[1]+r[7]*t[2]),-(r[2]*t[0]+r[5]*t[1]+r[8]*t[2]))}
  val x=c(a);val y=c(b);val dx=x[0]-y[0];val dy=x[1]-y[1];val dz=x[2]-y[2];return dx*dx+dy*dy+dz*dz
 }
 private fun transform(p:CameraPose,x:DoubleArray):DoubleArray{val r=p.rotation;val t=p.translation;return doubleArrayOf(r[0]*x[0]+r[1]*x[1]+r[2]*x[2]+t[0],r[3]*x[0]+r[4]*x[1]+r[5]*x[2]+t[1],r[6]*x[0]+r[7]*x[1]+r[8]*x[2]+t[2])}
 private fun reprojectionError(k:Mat,p:DoubleArray,o:CameraFeatureObservation):Double{
  val fx=k.get(0,0)[0];val sk=k.get(0,1)[0];val cx=k.get(0,2)[0];val fy=k.get(1,1)[0];val cy=k.get(1,2)[0]
  val u=(fx*p[0]+sk*p[1])/p[2]+cx;val v=fy*p[1]/p[2]+cy;val dx=u-o.x;val dy=v-o.y;return sqrt(dx*dx+dy*dy)
 }
}
internal data class TriangulationResult(val landmarks:List<CameraLandmark>,val stats:TriangulationStats)
data class CameraLandmark(val trackId:Int,val position:Point3,val observationCount:Int,val meanReprojectionErrorPx:Double,val triangulationAngleDegrees:Double=0.0)
data class TriangulationStats(val inputTracks:Int=0,val triangulatedLandmarks:Int=0,val rejectedShortTracks:Int=0,val rejectedMissingPoses:Int=0,val rejectedInvalidGeometry:Int=0,val rejectedReprojection:Int=0,val meanReprojectionErrorPx:Double=0.0,val meanTriangulationAngleDegrees:Double=0.0)
