package com.trackstudio.app.tracking.colmap

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.trackstudio.app.tracking.PinholeIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class ColmapP4pfAbsolutePoseInstrumentationTest {
    @Test
    fun syntheticPoseAndUnknownFocalAreRecoveredThroughJni() {
        val trueFocal = 920.0
        val cx = 640.0
        val cy = 360.0
        val trueRotation = rotation(
            doubleArrayOf(0.3, -0.4, 0.8),
            angle = 0.11
        )
        val trueTranslation = doubleArrayOf(0.24, -0.17, 0.31)
        val inlierCount = 140
        val outlierCount = 40
        val pointCount = inlierCount + outlierCount
        val worldPoints = DoubleArray(pointCount * 3)
        val imagePoints = DoubleArray(pointCount * 2)

        for (index in 0 until pointCount) {
            val worldX = 1.8 * sin(0.17 * index)
            val worldY = 1.4 * cos(0.13 * index)
            val worldZ = 4.2 + 0.035 * index + 0.4 * sin(0.07 * index)
            worldPoints[index * 3] = worldX
            worldPoints[index * 3 + 1] = worldY
            worldPoints[index * 3 + 2] = worldZ
            val cameraX = trueRotation[0] * worldX + trueRotation[1] * worldY +
                trueRotation[2] * worldZ + trueTranslation[0]
            val cameraY = trueRotation[3] * worldX + trueRotation[4] * worldY +
                trueRotation[5] * worldZ + trueTranslation[1]
            val cameraZ = trueRotation[6] * worldX + trueRotation[7] * worldY +
                trueRotation[8] * worldZ + trueTranslation[2]
            require(cameraZ > 0.0)
            var pixelX = trueFocal * cameraX / cameraZ + cx
            var pixelY = trueFocal * cameraY / cameraZ + cy
            if (index < inlierCount) {
                pixelX += 0.12 * sin(0.31 * index)
                pixelY += 0.10 * cos(0.23 * index)
            } else {
                pixelX += 110.0 + 3.0 * (index % 11)
                pixelY -= 125.0 + 2.0 * (index % 13)
            }
            imagePoints[index * 2] = pixelX
            imagePoints[index * 2 + 1] = pixelY
        }

        val initialIntrinsics = PinholeIntrinsics(
            fx = 720.0,
            fy = 735.0,
            skew = 0.0,
            cx = cx,
            cy = cy
        )
        val result = ColmapP4pfAbsolutePoseEstimator.estimate(
            worldPointsXYZ = worldPoints,
            imagePointsXY = imagePoints,
            initialIntrinsics = initialIntrinsics,
            maxErrorPx = 3.0,
            randomSeed = 20261117
        )
        assertNotNull("Native COLMAP P4PF returned no estimate", result)
        result ?: return
        assertTrue(result.trials in 100..ColmapP4pfAbsolutePoseEstimator.ANDROID_MAX_TRIALS)
        assertTrue(result.inlierIndices.size in (inlierCount - 3)..inlierCount)
        assertTrue(result.inlierIndices.all { it in 0 until inlierCount })
        assertTrue(abs(result.refinedFx - trueFocal) < 12.0)
        assertTrue(abs(result.refinedFy - trueFocal) < 12.0)

        val relative = multiply(result.rotation, transpose(trueRotation))
        val trace = relative[0] + relative[4] + relative[8]
        val rotationError = acos(((trace - 1.0) * 0.5).coerceIn(-1.0, 1.0))
        val translationError = sqrt(result.translation.indices.sumOf { index ->
            val difference = result.translation[index] - trueTranslation[index]
            difference * difference
        })
        assertTrue("Rotation error $rotationError", rotationError < 0.004)
        assertTrue("Translation error $translationError", translationError < 0.04)

        val repeated = ColmapP4pfAbsolutePoseEstimator.estimate(
            worldPointsXYZ = worldPoints,
            imagePointsXY = imagePoints,
            initialIntrinsics = initialIntrinsics,
            maxErrorPx = 3.0,
            randomSeed = 20261117
        ) ?: throw AssertionError("Fixed-seed P4PF repeat returned no estimate")
        assertEquals(result.inlierIndices, repeated.inlierIndices)
        assertEquals(result.trials, repeated.trials)
        assertEquals(result.refinedFx, repeated.refinedFx, 1e-12)
        assertEquals(result.refinedFy, repeated.refinedFy, 1e-12)
        assertTrue(result.rotation.indices.all { abs(result.rotation[it] - repeated.rotation[it]) < 1e-12 })
        assertTrue(result.translation.indices.all { abs(result.translation[it] - repeated.translation[it]) < 1e-12 })

        assertNull(
            "P4PF must reject unsupported nonzero camera skew",
            ColmapP4pfAbsolutePoseEstimator.estimate(
                worldPointsXYZ = worldPoints,
                imagePointsXY = imagePoints,
                initialIntrinsics = initialIntrinsics.copy(skew = 0.25),
                maxErrorPx = 3.0,
                randomSeed = 20261117
            )
        )
    }

    private fun rotation(axisInput: DoubleArray, angle: Double): DoubleArray {
        val norm = sqrt(axisInput.sumOf { it * it })
        val x = axisInput[0] / norm
        val y = axisInput[1] / norm
        val z = axisInput[2] / norm
        val c = cos(angle)
        val s = sin(angle)
        val t = 1.0 - c
        return doubleArrayOf(
            t * x * x + c, t * x * y - s * z, t * x * z + s * y,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c
        )
    }

    private fun transpose(matrix: DoubleArray) = DoubleArray(9) { index ->
        val row = index / 3
        val column = index % 3
        matrix[column * 3 + row]
    }

    private fun multiply(first: DoubleArray, second: DoubleArray): DoubleArray {
        val result = DoubleArray(9)
        for (row in 0 until 3) {
            for (column in 0 until 3) {
                for (inner in 0 until 3) {
                    result[row * 3 + column] +=
                        first[row * 3 + inner] * second[inner * 3 + column]
                }
            }
        }
        return result
    }
}
