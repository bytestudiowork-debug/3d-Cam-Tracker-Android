package com.trackstudio.app.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble

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
    try {
        cameraMatrix.create(3, 3, CvType.CV_64F)
        cameraMatrix.put(0, 0, *calibration.cameraMatrixValues())
        if (calibration.distortionCoefficients.isNotEmpty()) {
            distortion.fromArray(*calibration.distortionCoefficients.toDoubleArray())
        }

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
                if (distortion.empty()) {
                    source
                } else {
                    val corrected = Mat()
                    try {
                        Calib3d.undistort(source, corrected, cameraMatrix, distortion, cameraMatrix)
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

        return solve(frameCount, cameraMatrix, calibratedFrameProvider, onProgress)
    } finally {
        distortion.release()
        cameraMatrix.release()
    }
}
