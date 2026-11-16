package com.trackstudio.app.tracking

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import kotlin.math.abs
import kotlin.math.sqrt

class CameraProjectionMathTest {
    @Test
    fun skewAwareProjectionAndNormalizationAreInverses() {
        val intrinsics = PinholeIntrinsics(
            fx = 500.0,
            fy = 600.0,
            skew = 40.0,
            cx = 320.0,
            cy = 240.0
        )
        val cameraPoint = doubleArrayOf(0.4, -0.2, 4.0)

        val pixel = CameraProjectionMath.project(cameraPoint, intrinsics)
        val normalized = CameraProjectionMath.pixelToNormalized(pixel[0], pixel[1], intrinsics)

        assertArrayEquals(doubleArrayOf(0.1, -0.05, 1.0), normalized, 1e-12)
    }

    @Test
    fun skewAwareReprojectionErrorMatchesProjectedPixelResidual() {
        val intrinsics = PinholeIntrinsics(720.0, 680.0, 35.0, 320.0, 240.0)

        val error = CameraProjectionMath.reprojectionError(
            cameraX = 0.7,
            cameraY = 0.3,
            cameraZ = 5.0,
            observedX = 419.9,
            observedY = 282.8,
            intrinsics = intrinsics
        )

        assertEquals(sqrt(13.0), error!!, 1e-12)
    }

    @Test
    fun reprojectionErrorRejectsNonPositiveOrNonFiniteDepth() {
        val intrinsics = PinholeIntrinsics(500.0, 500.0, 0.0, 320.0, 240.0)

        assertNull(CameraProjectionMath.reprojectionError(0.0, 0.0, 0.0, 320.0, 240.0, intrinsics))
        assertNull(CameraProjectionMath.reprojectionError(0.0, 0.0, Double.NaN, 320.0, 240.0, intrinsics))
    }

    @Test
    fun analyticProjectionJacobianMatchesFiniteDifferencesWithSkew() {
        val intrinsics = PinholeIntrinsics(720.0, 680.0, 35.0, 320.0, 240.0)
        val point = doubleArrayOf(0.7, 0.3, 5.0)
        val analytic = CameraProjectionMath.jacobian(point, intrinsics)
        val epsilon = 1e-6

        for (axis in 0..2) {
            val plus = point.copyOf().also { it[axis] += epsilon }
            val minus = point.copyOf().also { it[axis] -= epsilon }
            val projectedPlus = CameraProjectionMath.project(plus, intrinsics)
            val projectedMinus = CameraProjectionMath.project(minus, intrinsics)
            for (row in 0..1) {
                val numeric = (projectedPlus[row] - projectedMinus[row]) / (2.0 * epsilon)
                assertEquals(analytic[row][axis], numeric, maxOf(1e-5, abs(numeric) * 1e-8))
            }
        }
    }
}
