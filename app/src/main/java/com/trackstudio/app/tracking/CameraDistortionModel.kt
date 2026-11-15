package com.trackstudio.app.tracking

import kotlin.math.cos
import kotlin.math.sin

internal object CameraDistortionModel {
    fun projectRawPixel(cameraPoint: DoubleArray, calibration: CameraCalibration): DoubleArray {
        require(cameraPoint.size == 3 && cameraPoint.all { it.isFinite() })
        require(cameraPoint[2] > 1e-8)

        val coefficients = calibration.distortionCoefficients
        fun coefficient(index: Int): Double = coefficients.getOrNull(index) ?: 0.0

        val x = cameraPoint[0] / cameraPoint[2]
        val y = cameraPoint[1] / cameraPoint[2]
        val r2 = x * x + y * y
        val r4 = r2 * r2
        val r6 = r4 * r2
        val numerator = 1.0 + coefficient(0) * r2 + coefficient(1) * r4 + coefficient(4) * r6
        val denominator = 1.0 + coefficient(5) * r2 + coefficient(6) * r4 + coefficient(7) * r6
        require(denominator.isFinite() && kotlin.math.abs(denominator) > 1e-12)
        val radial = numerator / denominator
        val p1 = coefficient(2)
        val p2 = coefficient(3)
        val xDistorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) +
            coefficient(8) * r2 + coefficient(9) * r4
        val yDistorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y +
            coefficient(10) * r2 + coefficient(11) * r4

        val tilted = if (coefficients.size == 14) {
            applySensorTilt(xDistorted, yDistorted, coefficient(12), coefficient(13))
        } else {
            doubleArrayOf(xDistorted, yDistorted)
        }
        val pixel = doubleArrayOf(
            calibration.fx * tilted[0] + calibration.skew * tilted[1] + calibration.cx,
            calibration.fy * tilted[1] + calibration.cy
        )
        require(pixel.all { it.isFinite() })
        return pixel
    }

    private fun applySensorTilt(x: Double, y: Double, tauX: Double, tauY: Double): DoubleArray {
        val cosineX = cos(tauX)
        val sineX = sin(tauX)
        val cosineY = cos(tauY)
        val sineY = sin(tauY)

        val r00 = cosineY
        val r01 = sineY * sineX
        val r02 = -sineY * cosineX
        val r10 = 0.0
        val r11 = cosineX
        val r12 = sineX
        val r20 = sineY
        val r21 = -cosineY * sineX
        val r22 = cosineY * cosineX

        val rotatedX = r00 * x + r01 * y + r02
        val rotatedY = r10 * x + r11 * y + r12
        val rotatedZ = r20 * x + r21 * y + r22
        require(rotatedZ.isFinite() && kotlin.math.abs(rotatedZ) > 1e-12)

        val untiltedX = r22 * rotatedX - r02 * rotatedZ
        val untiltedY = r22 * rotatedY - r12 * rotatedZ
        return doubleArrayOf(untiltedX / rotatedZ, untiltedY / rotatedZ)
    }
}
