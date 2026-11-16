package com.trackstudio.app.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class CameraCalibration(
    val imageWidth: Int,
    val imageHeight: Int,
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val skew: Double = 0.0,
    distortionCoefficients: List<Double> = emptyList()
) {
    val distortionCoefficients: List<Double> = distortionCoefficients.toList()
    val hasDistortion: Boolean
        get() = distortionCoefficients.any { it != 0.0 }

    init {
        require(imageWidth > 0 && imageHeight > 0)
        require(fx.isFinite() && fy.isFinite() && fx > 0.0 && fy > 0.0)
        require(cx.isFinite() && cy.isFinite() && skew.isFinite())
        require(this.distortionCoefficients.size in setOf(0, 4, 5, 8, 12, 14))
        require(this.distortionCoefficients.all { it.isFinite() })
    }

    internal fun cameraMatrixValues(): DoubleArray = doubleArrayOf(
        fx, skew, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0
    )

    fun projectRawPixel(cameraPoint: DoubleArray): DoubleArray =
        CameraDistortionModel.projectRawPixel(cameraPoint, this)
}

suspend fun ReconstructionEngine.solve(
    frameCount: Int,
    calibration: CameraCalibration,
    frameProvider: suspend (Int) -> Mat?,
    onProgress: suspend (CameraTrackingProgress) -> Unit = {}
): CameraTrackingResult {
    require(frameCount >= 3)
    check(OpenCvRuntime.ensureLoaded()) { "OpenCV could not be initialized on this device." }

    val cameraMatrix = Mat()
    val distortion = MatOfDouble()
    val skewNeedsRectification = calibration.hasDistortion && calibration.skew != 0.0
    val observationSkew = if (skewNeedsRectification) 0.0 else calibration.skew
    val observationIntrinsics = CameraIntrinsics(
        fx = calibration.fx,
        skew = observationSkew,
        cx = calibration.cx,
        fy = calibration.fy,
        cy = calibration.cy
    )
    var skewCompensatedMaps: Pair<Mat, Mat>? = null
    try {
        cameraMatrix.create(3, 3, CvType.CV_64F)
        val matrixValues = calibration.cameraMatrixValues()
        matrixValues[1] = observationSkew
        cameraMatrix.put(0, 0, *matrixValues)
        if (calibration.hasDistortion) {
            distortion.fromArray(*calibration.distortionCoefficients.toDoubleArray())
        }
        if (skewNeedsRectification) {
            skewCompensatedMaps = createSkewCompensatedUndistortionMaps(
                calibration, cameraMatrix, distortion
            )
        }
        val remap = skewCompensatedMaps

        val calibratedFrameProvider: suspend (Int) -> Mat? = { frameIndex ->
            val source = frameProvider(frameIndex)
            if (source == null) {
                null
            } else {
                val width = source.cols()
                val height = source.rows()
                if (width != calibration.imageWidth || height != calibration.imageHeight) {
                    source.release()
                    throw IllegalArgumentException(
                        "Frame ${frameIndex + 1} is ${width}x${height}, but calibration is " +
                            "${calibration.imageWidth}x${calibration.imageHeight}."
                    )
                }
                if (!calibration.hasDistortion) {
                    source
                } else {
                    val corrected = Mat()
                    try {
                        if (remap != null) {
                            Imgproc.remap(
                                source, corrected, remap.first, remap.second,
                                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0)
                            )
                        } else {
                            Calib3d.undistort(source, corrected, cameraMatrix, distortion, cameraMatrix)
                        }
                        corrected
                    } catch (failure: Throwable) {
                        corrected.release()
                        throw failure
                    } finally {
                        source.release()
                    }
                }
            }
        }

        return solve(frameCount, cameraMatrix, calibratedFrameProvider, onProgress).copy(
            inputCameraCalibration = calibration,
            observationCoordinateModel = CameraObservationCoordinateModel.CALIBRATED_PINHOLE_PIXELS,
            observationCameraIntrinsics = observationIntrinsics
        )
    } finally {
        skewCompensatedMaps?.first?.release()
        skewCompensatedMaps?.second?.release()
        distortion.release()
        cameraMatrix.release()
    }
}

internal fun createSkewCompensatedUndistortionMaps(
    calibration: CameraCalibration,
    zeroSkewCameraMatrix: Mat,
    distortion: MatOfDouble
): Pair<Mat, Mat> {
    require(calibration.hasDistortion && calibration.skew != 0.0)
    val rotation = Mat.eye(3, 3, CvType.CV_64F).toMat()
    val mapX = Mat()
    val mapY = Mat()
    val correctedMapX = Mat()
    val centeredY = Mat()
    val xShift = Mat()
    var keepMaps = false
    try {
        Calib3d.initUndistortRectifyMap(
            zeroSkewCameraMatrix,
            distortion,
            rotation,
            zeroSkewCameraMatrix,
            Size(calibration.imageWidth.toDouble(), calibration.imageHeight.toDouble()),
            CvType.CV_32FC1,
            mapX,
            mapY
        )
        Core.subtract(mapY, Scalar(calibration.cy), centeredY)
        Core.multiply(centeredY, Scalar(calibration.skew / calibration.fy), xShift)
        Core.add(mapX, xShift, correctedMapX)
        keepMaps = true
        return correctedMapX to mapY
    } finally {
        rotation.release()
        mapX.release()
        centeredY.release()
        xShift.release()
        if (!keepMaps) {
            mapY.release()
            correctedMapX.release()
        }
    }
}
