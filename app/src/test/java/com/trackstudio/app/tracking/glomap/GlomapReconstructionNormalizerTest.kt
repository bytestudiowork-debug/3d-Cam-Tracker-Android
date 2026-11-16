package com.trackstudio.app.tracking.glomap

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GlomapReconstructionNormalizerTest {
    @Test
    fun usesPinnedPercentileIndicesAndAppliesOneSim3ToCamerasAndPoints() {
        val centers = buildMap {
            for (x in 0..9) put(x, doubleArrayOf(x.toDouble(), 0.0, 0.0))
            put(10, doubleArrayOf(1_000.0, 0.0, 0.0))
        }
        val points = mapOf(4 to doubleArrayOf(2.0, 3.0, 4.0))

        val result = GlomapReconstructionNormalizer.normalize(centers, points)

        assertTrue(result.applied)
        assertEquals(1.25, result.scale, 1e-12)
        assertEquals(-6.25, result.translation[0], 1e-12)
        assertEquals(0.0, result.translation[1], 0.0)
        assertEquals(0.0, result.translation[2], 0.0)
        assertEquals(-5.0, result.frameCentersById.getValue(1)[0], 1e-12)
        assertEquals(5.0, result.frameCentersById.getValue(9)[0], 1e-12)
        assertEquals(1_243.75, result.frameCentersById.getValue(10)[0], 1e-9)
        assertEquals(-3.75, result.pointsByTrackId.getValue(4)[0], 1e-12)
        assertEquals(3.75, result.pointsByTrackId.getValue(4)[1], 1e-12)
        assertEquals(5.0, result.pointsByTrackId.getValue(4)[2], 1e-12)

        val robustMeanX = (1..9).map { result.frameCentersById.getValue(it)[0] }.average()
        val robustExtentX = result.frameCentersById.getValue(9)[0] -
            result.frameCentersById.getValue(1)[0]
        assertEquals(0.0, robustMeanX, 1e-12)
        assertEquals(10.0, robustExtentX, 1e-12)
    }

    @Test
    fun normalizationPreservesPinholeProjectionForTransformedCameraAndPoint() {
        val centers = mapOf(
            0 to doubleArrayOf(-1.0, 0.0, 0.0),
            1 to doubleArrayOf(0.0, 0.5, 0.0),
            2 to doubleArrayOf(1.0, 0.0, 0.0),
            3 to doubleArrayOf(0.0, -0.5, 0.0)
        )
        val point = doubleArrayOf(0.3, 0.2, 5.0)
        val cameraCenter = centers.getValue(1)
        val result = GlomapReconstructionNormalizer.normalize(
            frameCentersById = centers,
            pointsByTrackId = mapOf(7 to point)
        )

        val before = project(point, cameraCenter)
        val after = project(
            result.pointsByTrackId.getValue(7),
            result.frameCentersById.getValue(1)
        )

        assertEquals(before[0], after[0], 1e-10)
        assertEquals(before[1], after[1], 1e-10)
    }

    @Test
    fun smallCameraSetsUseExtremaAndFixedScaleOnlyCentersTheCloud() {
        val centers = mapOf(
            0 to doubleArrayOf(0.0, 1.0, 0.0),
            1 to doubleArrayOf(1.0, 0.0, 0.0),
            2 to doubleArrayOf(2.0, -1.0, 0.0)
        )

        val scaled = GlomapReconstructionNormalizer.normalize(centers, emptyMap())
        val fixed = GlomapReconstructionNormalizer.normalize(
            centers,
            emptyMap(),
            fixedScale = true
        )

        assertEquals(10.0 / kotlin.math.sqrt(8.0), scaled.scale, 1e-12)
        assertEquals(1.0, fixed.scale, 0.0)
        assertEquals(0.0, fixed.frameCentersById.getValue(1)[0], 1e-12)
        assertEquals(0.0, fixed.frameCentersById.getValue(1)[1], 1e-12)
        assertEquals(1.0, centers.getValue(1)[0], 0.0)
    }

    @Test
    fun singleDegenerateCenterTranslatesWithoutChangingScaleAndEmptyInputIsSafe() {
        val single = GlomapReconstructionNormalizer.normalize(
            frameCentersById = mapOf(5 to doubleArrayOf(1.0, 2.0, 3.0)),
            pointsByTrackId = mapOf(8 to doubleArrayOf(2.0, 3.0, 4.0))
        )
        assertEquals(1.0, single.scale, 0.0)
        assertEquals(0.0, single.frameCentersById.getValue(5)[0], 0.0)
        assertEquals(0.0, single.frameCentersById.getValue(5)[1], 0.0)
        assertEquals(0.0, single.frameCentersById.getValue(5)[2], 0.0)
        assertEquals(1.0, single.pointsByTrackId.getValue(8)[0], 0.0)

        val empty = GlomapReconstructionNormalizer.normalize(
            frameCentersById = emptyMap(),
            pointsByTrackId = mapOf(8 to doubleArrayOf(2.0, 3.0, 4.0))
        )
        assertFalse(empty.applied)
        assertEquals(1.0, empty.scale, 0.0)
        assertEquals(2.0, empty.pointsByTrackId.getValue(8)[0], 0.0)
    }

    private fun project(point: DoubleArray, cameraCenter: DoubleArray): DoubleArray {
        val cameraX = point[0] - cameraCenter[0]
        val cameraY = point[1] - cameraCenter[1]
        val cameraZ = point[2] - cameraCenter[2]
        return doubleArrayOf(
            800.0 * cameraX / cameraZ + 320.0,
            800.0 * cameraY / cameraZ + 240.0
        )
    }
}
