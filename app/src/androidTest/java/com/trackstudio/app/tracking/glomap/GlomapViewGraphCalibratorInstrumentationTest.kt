package com.trackstudio.app.tracking.glomap

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.trackstudio.app.tracking.CameraIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class GlomapViewGraphCalibratorInstrumentationTest {
    @Test
    fun syntheticViewGraphRecoversFocalAndRejectsDegeneratePair() {
        val trueFocal = 1200.0
        val principalX = 640.0
        val principalY = 360.0
        val fundamentals = (0 until 10).map { pairIndex ->
            syntheticFundamental(trueFocal, principalX, principalY, pairIndex)
        } + listOf(DoubleArray(9))

        val result = GlomapViewGraphCalibrator().calibrate(
            fundamentals = fundamentals,
            initialIntrinsics = CameraIntrinsics(
                fx = trueFocal * 0.75,
                skew = 0.0,
                cx = principalX,
                fy = trueFocal * 0.75,
                cy = principalY
            ),
            hasPriorFocalLength = false,
            maxIterations = 100,
            numThreads = 1
        ) ?: throw AssertionError("Native view-graph focal calibration returned no result")

        assertTrue(abs(result.intrinsics.fx - trueFocal) / trueFocal < 0.02)
        assertEquals(result.intrinsics.fx, result.intrinsics.fy, 1e-12)
        assertEquals(10, result.acceptedPairIndices.size)
        assertTrue(result.rejectedPairIndices.contains(10))
        assertTrue(result.finalCost < result.initialCost)
    }

    @Test
    fun calibratedSharedCameraKeepsTheSuppliedFocalPrior() {
        val sceneFocal = 1200.0
        val priorFocal = 900.0
        val principalX = 640.0
        val principalY = 360.0
        val fundamentals = (0 until 10).map { pairIndex ->
            syntheticFundamental(sceneFocal, principalX, principalY, pairIndex)
        }

        val result = GlomapViewGraphCalibrator().calibrate(
            fundamentals = fundamentals,
            initialIntrinsics = CameraIntrinsics(
                fx = priorFocal,
                skew = 0.0,
                cx = principalX,
                fy = priorFocal,
                cy = principalY
            ),
            hasPriorFocalLength = true,
            maxIterations = 100,
            numThreads = 1
        ) ?: throw AssertionError("Fixed-prior shared-camera calibration returned no result")

        assertEquals(priorFocal, result.intrinsics.fx, 0.0)
        assertEquals(priorFocal, result.intrinsics.fy, 0.0)
        assertEquals(priorFocal, result.estimatedFocalLength, 0.0)
        assertFalse(result.focalLengthUpdated)
        assertEquals(0, result.iterations)
        assertEquals(fundamentals.indices.toSet(), result.acceptedPairIndices)
        assertTrue(result.rejectedPairIndices.isEmpty())
    }

    @Test
    fun syntheticCrossCameraGraphRecoversIndependentFocalsAndRespectsPrior() {
        val trueFocal0 = 900.0
        val trueFocal1 = 1200.0
        val principalX0 = 640.0
        val principalY0 = 360.0
        val principalX1 = 800.0
        val principalY1 = 450.0
        val pairs = (0 until 8).map { pairIndex ->
            GlomapViewGraphCalibrator.FundamentalPair(
                syntheticCrossCameraFundamental(
                    trueFocal0, principalX0, principalY0,
                    trueFocal1, principalX1, principalY1, pairIndex
                ),
                firstCameraId = 7,
                secondCameraId = 42
            )
        } + (0 until 4).map { pairIndex ->
            GlomapViewGraphCalibrator.FundamentalPair(
                syntheticFundamental(trueFocal1, principalX1, principalY1, pairIndex + 20),
                firstCameraId = 42,
                secondCameraId = 42
            )
        } + listOf(
            GlomapViewGraphCalibrator.FundamentalPair(
                doubleArrayOf(
                    -1.0037475493, -0.01503000959, -0.6273950898,
                    0.07710208839, 1.087582599, -1.403222236,
                    0.4910696008, 0.7344655981, -0.6644424178
                ),
                firstCameraId = 7,
                secondCameraId = 42
            )
        )
        val cameras = listOf(
            GlomapViewGraphCalibrator.CameraInput(
                cameraId = 42,
                intrinsics = CameraIntrinsics(
                    fx = trueFocal1 * 0.75,
                    skew = 0.0,
                    cx = principalX1,
                    fy = trueFocal1 * 0.75,
                    cy = principalY1
                )
            ),
            GlomapViewGraphCalibrator.CameraInput(
                cameraId = 7,
                intrinsics = CameraIntrinsics(
                    fx = trueFocal0,
                    skew = 0.0,
                    cx = principalX0,
                    fy = trueFocal0,
                    cy = principalY0
                ),
                hasPriorFocalLength = true
            )
        )

        val result = GlomapViewGraphCalibrator().calibrateMultiCamera(
            pairs = pairs,
            cameras = cameras,
            maxIterations = 100,
            numThreads = 1
        ) ?: throw AssertionError("Native multi-camera view-graph calibration returned no result")

        assertTrue(abs(result.intrinsicsByCameraId.getValue(42).fx - trueFocal1) / trueFocal1 < 0.02)
        assertEquals(trueFocal0, result.intrinsicsByCameraId.getValue(7).fx, 0.0)
        assertEquals(trueFocal0, result.estimatedFocalLengthsByCameraId.getValue(7), 0.0)
        assertTrue(result.focalLengthUpdatedCameraIds.contains(42))
        assertEquals(12, result.acceptedPairIndices.size)
        assertTrue(result.rejectedPairIndices.contains(12))
        assertTrue(result.finalCost < result.initialCost)
    }

    private fun syntheticFundamental(
        focal: Double,
        principalX: Double,
        principalY: Double,
        pairIndex: Int
    ): DoubleArray {
        val axis = doubleArrayOf(
            0.06 + 0.011 * pairIndex,
            -0.12 + 0.007 * pairIndex,
            0.04 + 0.009 * pairIndex
        )
        val axisNorm = sqrt(axis.sumOf { it * it })
        for (index in axis.indices) axis[index] /= axisNorm
        val angle = 0.08 + 0.015 * pairIndex
        val axisSkew = skew(axis)
        val axisSkewSquared = multiply(axisSkew, axisSkew)
        val rotation = identity()
        for (index in rotation.indices) {
            rotation[index] += sin(angle) * axisSkew[index] +
                (1.0 - cos(angle)) * axisSkewSquared[index]
        }

        val translation = doubleArrayOf(
            0.42 + 0.03 * pairIndex,
            -0.16 + 0.02 * pairIndex,
            0.09 + 0.013 * pairIndex
        )
        val essential = multiply(skew(translation), rotation)
        val inverseCamera = doubleArrayOf(
            1.0 / focal, 0.0, -principalX / focal,
            0.0, 1.0 / focal, -principalY / focal,
            0.0, 0.0, 1.0
        )
        val fundamental = multiply(
            multiply(transpose(inverseCamera), essential),
            inverseCamera
        )
        val norm = sqrt(fundamental.sumOf { it * it })
        return DoubleArray(fundamental.size) { fundamental[it] / norm }
    }

    private fun syntheticCrossCameraFundamental(
        focal0: Double,
        principalX0: Double,
        principalY0: Double,
        focal1: Double,
        principalX1: Double,
        principalY1: Double,
        pairIndex: Int
    ): DoubleArray {
        val axis = doubleArrayOf(
            0.06 + 0.011 * pairIndex,
            -0.12 + 0.007 * pairIndex,
            0.04 + 0.009 * pairIndex
        )
        val axisNorm = sqrt(axis.sumOf { it * it })
        for (index in axis.indices) axis[index] /= axisNorm
        val angle = 0.08 + 0.015 * pairIndex
        val axisSkew = skew(axis)
        val axisSkewSquared = multiply(axisSkew, axisSkew)
        val rotation = identity()
        for (index in rotation.indices) {
            rotation[index] += sin(angle) * axisSkew[index] +
                (1.0 - cos(angle)) * axisSkewSquared[index]
        }
        val translation = doubleArrayOf(
            0.42 + 0.03 * pairIndex,
            -0.16 + 0.02 * pairIndex,
            0.09 + 0.013 * pairIndex
        )
        val essential = multiply(skew(translation), rotation)
        val inverseCamera0 = doubleArrayOf(
            1.0 / focal0, 0.0, -principalX0 / focal0,
            0.0, 1.0 / focal0, -principalY0 / focal0,
            0.0, 0.0, 1.0
        )
        val inverseCamera1 = doubleArrayOf(
            1.0 / focal1, 0.0, -principalX1 / focal1,
            0.0, 1.0 / focal1, -principalY1 / focal1,
            0.0, 0.0, 1.0
        )
        val fundamental = multiply(
            multiply(transpose(inverseCamera1), essential),
            inverseCamera0
        )
        val norm = sqrt(fundamental.sumOf { it * it })
        return DoubleArray(fundamental.size) { fundamental[it] / norm }
    }

    private fun identity() = doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )

    private fun skew(vector: DoubleArray) = doubleArrayOf(
        0.0, -vector[2], vector[1],
        vector[2], 0.0, -vector[0],
        -vector[1], vector[0], 0.0
    )

    private fun transpose(matrix: DoubleArray) = DoubleArray(9) { index ->
        val row = index / 3
        val col = index % 3
        matrix[col * 3 + row]
    }

    private fun multiply(first: DoubleArray, second: DoubleArray): DoubleArray {
        val result = DoubleArray(9)
        for (row in 0 until 3) {
            for (col in 0 until 3) {
                for (inner in 0 until 3) {
                    result[row * 3 + col] +=
                        first[row * 3 + inner] * second[inner * 3 + col]
                }
            }
        }
        return result
    }
}
