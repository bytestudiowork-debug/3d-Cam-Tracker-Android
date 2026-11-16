package com.trackstudio.app.tracking.glomap

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.trackstudio.app.tracking.CameraIntrinsics
import org.junit.Assert.assertEquals
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
            maxIterations = 100,
            numThreads = 1
        ) ?: throw AssertionError("Native view-graph focal calibration returned no result")

        assertTrue(abs(result.intrinsics.fx - trueFocal) / trueFocal < 0.02)
        assertEquals(result.intrinsics.fx, result.intrinsics.fy, 1e-12)
        assertEquals(10, result.acceptedPairIndices.size)
        assertTrue(result.rejectedPairIndices.contains(10))
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
