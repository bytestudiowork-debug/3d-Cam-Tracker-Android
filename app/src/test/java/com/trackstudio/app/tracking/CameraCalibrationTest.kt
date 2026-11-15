package com.trackstudio.app.tracking

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class CameraCalibrationTest {
    @Test
    fun emitsRowMajorOpenCvIntrinsicMatrix() {
        val calibration = CameraCalibration(
            imageWidth = 1920,
            imageHeight = 1080,
            fx = 1200.0,
            fy = 1180.0,
            cx = 960.0,
            cy = 540.0,
            skew = 0.25
        )

        assertArrayEquals(
            doubleArrayOf(1200.0, 0.25, 960.0, 0.0, 1180.0, 540.0, 0.0, 0.0, 1.0),
            calibration.cameraMatrixValues(),
            0.0
        )
    }

    @Test
    fun acceptsOpenCvDistortionModelVectorSizes() {
        for (size in listOf(4, 5, 8, 12, 14)) {
            val calibration = CameraCalibration(
                imageWidth = 640,
                imageHeight = 480,
                fx = 500.0,
                fy = 500.0,
                cx = 320.0,
                cy = 240.0,
                distortionCoefficients = List(size) { 0.0 }
            )
            assertEquals(size, calibration.distortionCoefficients.size)
        }
    }

    @Test
    fun distinguishesZeroDistortionFromAnActiveLensModel() {
        val zero = CameraCalibration(
            imageWidth = 640,
            imageHeight = 480,
            fx = 500.0,
            fy = 500.0,
            cx = 320.0,
            cy = 240.0,
            distortionCoefficients = List(14) { 0.0 }
        )
        val sourceCoefficients = mutableListOf(0.08, -0.015, 0.001, -0.0007, 0.002)
        val distorted = CameraCalibration(
            imageWidth = 640,
            imageHeight = 480,
            fx = 500.0,
            fy = 500.0,
            cx = 320.0,
            cy = 240.0,
            distortionCoefficients = sourceCoefficients
        )
        sourceCoefficients[0] = 0.0

        assertEquals(false, zero.hasDistortion)
        assertEquals(true, distorted.hasDistortion)
        assertEquals(0.08, distorted.distortionCoefficients[0], 0.0)
        assertEquals(-0.0007, distorted.distortionCoefficients[3], 0.0)
    }

    @Test
    fun projectsFiveCoefficientRadialTangentialModelDeterministically() {
        val calibration = CameraCalibration(
            imageWidth = 1280,
            imageHeight = 720,
            fx = 700.0,
            fy = 710.0,
            cx = 640.0,
            cy = 360.0,
            skew = 2.5,
            distortionCoefficients = listOf(0.08, -0.015, 0.001, -0.0007, 0.003)
        )

        assertArrayEquals(
            doubleArrayOf(923.7450386858302, 179.81256982824215),
            calibration.projectRawPixel(doubleArrayOf(0.8, -0.5, 2.0)),
            1e-10
        )
    }

    @Test
    fun projectsThinPrismAndTiltedSensorModelDeterministically() {
        val calibration = CameraCalibration(
            imageWidth = 1280,
            imageHeight = 720,
            fx = 700.0,
            fy = 710.0,
            cx = 640.0,
            cy = 360.0,
            skew = 2.5,
            distortionCoefficients = listOf(
                0.08, -0.015, 0.001, -0.0007, 0.003,
                0.001, -0.0002, 0.00005,
                0.0001, -0.00002, 0.00003, -0.00001,
                0.04, -0.03
            )
        )

        assertArrayEquals(
            doubleArrayOf(924.4096359485001, 179.68889245948128),
            calibration.projectRawPixel(doubleArrayOf(0.8, -0.5, 2.0)),
            1e-10
        )
    }

    @Test
    fun zeroDistortionProjectionUsesPinholeIntrinsicsAndSkew() {
        val calibration = CameraCalibration(
            imageWidth = 1280,
            imageHeight = 720,
            fx = 700.0,
            fy = 710.0,
            cx = 640.0,
            cy = 360.0,
            skew = 2.5,
            distortionCoefficients = List(14) { 0.0 }
        )

        assertArrayEquals(
            doubleArrayOf(919.375, 182.5),
            calibration.projectRawPixel(doubleArrayOf(0.8, -0.5, 2.0)),
            1e-12
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsMalformedDistortionVector() {
        CameraCalibration(
            imageWidth = 640,
            imageHeight = 480,
            fx = 500.0,
            fy = 500.0,
            cx = 320.0,
            cy = 240.0,
            distortionCoefficients = listOf(0.1, 0.2, 0.3)
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNonFiniteIntrinsics() {
        CameraCalibration(
            imageWidth = 640,
            imageHeight = 480,
            fx = Double.NaN,
            fy = 500.0,
            cx = 320.0,
            cy = 240.0
        )
    }
}
