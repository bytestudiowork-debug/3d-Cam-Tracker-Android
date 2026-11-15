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
