package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraProjectionMath
import com.trackstudio.app.tracking.PinholeIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GlomapImagePairInlierScorerTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 800.0,
        fy = 800.0,
        skew = 0.0,
        cx = 640.0,
        cy = 360.0
    )
    private val identity = doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    private val unitBaseline = doubleArrayOf(1.0, 0.0, 0.0)

    @Test
    fun exactTwoViewSceneRetainsEveryMatchAndHasNearZeroCost() {
        val scene = scene(count = 40) { index ->
            doubleArrayOf(
                -0.7 + (index % 10) * 0.14,
                -0.35 + (index / 10) * 0.18,
                4.0 + (index % 7) * 0.7
            )
        }

        val result = scorer(scene.first, scene.second)

        assertNotNull(result)
        assertEquals((0 until 40).toList(), result!!.inlierIndices)
        assertTrue(result.score < result.squaredThreshold * 1e-6)
    }

    @Test
    fun nonzeroSkewAndRotatedRelativePosePreserveExactCorrespondences() {
        val skewedIntrinsics = PinholeIntrinsics(
            fx = 900.0,
            fy = 780.0,
            skew = 24.0,
            cx = 512.0,
            cy = 384.0
        )
        val angle = Math.toRadians(5.0)
        val cosine = kotlin.math.cos(angle)
        val sine = kotlin.math.sin(angle)
        val rotation = doubleArrayOf(
            cosine, 0.0, sine,
            0.0, 1.0, 0.0,
            -sine, 0.0, cosine
        )
        val firstPixels = ArrayList<Pair<Double, Double>>()
        val secondPixels = ArrayList<Pair<Double, Double>>()
        repeat(32) { index ->
            val point = doubleArrayOf(
                -0.6 + (index % 8) * 0.16,
                -0.3 + (index / 8) * 0.18,
                4.0 + (index % 6) * 0.8
            )
            val firstPixel = CameraProjectionMath.project(point, skewedIntrinsics)
            val secondPoint = doubleArrayOf(
                rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2] + 1.0,
                rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2],
                rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2]
            )
            val secondPixel = CameraProjectionMath.project(secondPoint, skewedIntrinsics)
            firstPixels += firstPixel[0] to firstPixel[1]
            secondPixels += secondPixel[0] to secondPixel[1]
        }

        val result = GlomapImagePairInlierScorer.scoreEssential(
            firstPixels = firstPixels,
            secondPixels = secondPixels,
            firstIntrinsics = skewedIntrinsics,
            secondIntrinsics = skewedIntrinsics,
            rotation = rotation,
            translation = unitBaseline
        )

        assertNotNull(result)
        assertEquals((0 until 32).toList(), result!!.inlierIndices)
    }

    @Test
    fun sourceSampsonThresholdRejectsShiftedCorrespondencesAndCapsTheirCost() {
        val scene = scene(count = 40) { index ->
            doubleArrayOf(
                -0.6 + (index % 10) * 0.12,
                -0.3 + (index / 10) * 0.15,
                5.0 + (index % 5)
            )
        }
        val outliers = setOf(2, 9, 17, 31)
        val secondWithOutliers = scene.second.mapIndexed { index, point ->
            if (index in outliers) point.first to (point.second + 5.0) else point
        }

        val result = scorer(scene.first, secondWithOutliers)

        assertNotNull(result)
        assertEquals((0 until 40).filterNot { it in outliers }, result!!.inlierIndices)
        val rejectedCount = 40 - result.inlierIndices.size
        assertTrue(result.score >= rejectedCount * result.squaredThreshold)
    }

    @Test
    fun cheiralityRejectsPointsOutsideThePinnedMaximumDepth() {
        val scene = scene(count = 12) { index ->
            doubleArrayOf(-0.4 + index * 0.07, 0.1 - index * 0.015, 150.0)
        }

        val result = scorer(scene.first, scene.second)

        assertNotNull(result)
        assertTrue(result!!.inlierIndices.isEmpty())
        assertEquals(12.0 * result.squaredThreshold, result.score, 1e-12)
    }

    @Test
    fun epipoleGuardRejectsNearBaselineRaysThatOtherwisePassGeometryChecks() {
        val firstPixel = CameraProjectionMath.project(
            doubleArrayOf(-0.5, 0.0, 0.01), intrinsics
        )
        val secondPixel = CameraProjectionMath.project(
            doubleArrayOf(0.5, 0.0, 0.01), intrinsics
        )

        val result = scorer(
            first = listOf(firstPixel[0] to firstPixel[1]),
            second = listOf(secondPixel[0] to secondPixel[1])
        )

        assertNotNull(result)
        assertTrue(result!!.inlierIndices.isEmpty())
        assertEquals(result.squaredThreshold, result.score, 1e-12)
    }

    @Test
    fun rejectsInvalidRotationOrMismatchedFeatureArrays() {
        val scene = scene(count = 1) { doubleArrayOf(0.1, 0.1, 5.0) }

        assertNull(
            GlomapImagePairInlierScorer.scoreEssential(
                scene.first,
                scene.second,
                intrinsics,
                intrinsics,
                rotation = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0),
                translation = unitBaseline
            )
        )
        assertNull(
            GlomapImagePairInlierScorer.scoreEssential(
                scene.first,
                emptyList(),
                intrinsics,
                intrinsics,
                rotation = identity,
                translation = unitBaseline
            )
        )
    }

    private fun scorer(
        first: List<Pair<Double, Double>>,
        second: List<Pair<Double, Double>>
    ) = GlomapImagePairInlierScorer.scoreEssential(
        firstPixels = first,
        secondPixels = second,
        firstIntrinsics = intrinsics,
        secondIntrinsics = intrinsics,
        rotation = identity,
        translation = unitBaseline
    )

    private fun scene(
        count: Int,
        pointAt: (Int) -> DoubleArray
    ): Pair<List<Pair<Double, Double>>, List<Pair<Double, Double>>> {
        val first = ArrayList<Pair<Double, Double>>(count)
        val second = ArrayList<Pair<Double, Double>>(count)
        repeat(count) { index ->
            val point = pointAt(index)
            val firstPixel = CameraProjectionMath.project(point, intrinsics)
            val secondPixel = CameraProjectionMath.project(
                doubleArrayOf(point[0] + unitBaseline[0], point[1], point[2]),
                intrinsics
            )
            first += firstPixel[0] to firstPixel[1]
            second += secondPixel[0] to secondPixel[1]
        }
        return first to second
    }
}
