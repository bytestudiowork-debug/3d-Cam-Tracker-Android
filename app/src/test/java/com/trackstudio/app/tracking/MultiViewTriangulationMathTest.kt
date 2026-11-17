package com.trackstudio.app.tracking

import kotlin.math.acos
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.sqrt
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point3

class MultiViewTriangulationMathTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 760.0,
        fy = 810.0,
        skew = 18.0,
        cx = 320.0,
        cy = 240.0
    )

    @Test
    fun twoViewDltRecoversKnownPointWithSkewedIntrinsics() {
        val worldPoint = doubleArrayOf(0.25, -0.17, 4.7)
        val first = view(frame = 0, centerX = 0.0, centerY = 0.0, point = worldPoint)
        val second = view(frame = 1, centerX = 1.2, centerY = 0.1, point = worldPoint)

        val estimate = MultiViewTriangulationMath.triangulateDlt(listOf(first, second), intrinsics)

        assertNotNull(estimate)
        assertArrayEquals(worldPoint, estimate!!, 1e-6)
    }

    @Test
    fun triangulationAngleUsesTheEstimatedPointAndCameraCentersForNoisyRays() {
        val worldPoint = doubleArrayOf(0.25, -0.17, 4.7)
        val first = view(frame = 0, centerX = 0.0, centerY = 0.0, point = worldPoint)
        val second = view(frame = 1, centerX = 1.2, centerY = 0.1, point = worldPoint)
            .let { it.copy(pixelY = it.pixelY + 150.0) }

        val triangulatedPoint = MultiViewTriangulationMath.triangulateDlt(
            listOf(first, second), intrinsics
        )
        assertNotNull(triangulatedPoint)
        val angle = MultiViewTriangulationMath.triangulationAngleDegrees(first, second, intrinsics)
        assertNotNull(angle)

        val center1 = CameraPoseGeometry.cameraCenterWorld(first.pose)
        val center2 = CameraPoseGeometry.cameraCenterWorld(second.pose)
        val point = triangulatedPoint!!
        val ray1Length = sqrt(center1.indices.sumOf { axis ->
            val distance = point[axis] - center1[axis]
            distance * distance
        })
        val ray2Length = sqrt(center2.indices.sumOf { axis ->
            val distance = point[axis] - center2[axis]
            distance * distance
        })
        val baselineLength = sqrt(center1.indices.sumOf { axis ->
            val distance = center1[axis] - center2[axis]
            distance * distance
        })
        val cosine = ((ray1Length * ray1Length + ray2Length * ray2Length - baselineLength * baselineLength) /
            (2.0 * ray1Length * ray2Length)).coerceIn(-1.0, 1.0)
        val rawAngle = acos(cosine)
        val expectedAngle = Math.toDegrees(min(rawAngle, Math.PI - rawAngle))

        assertEquals(expectedAngle, angle!!, 1e-9)
        assertTrue(
            "Noisy ray-angle proxy should differ from the triangulated-point angle",
            abs(
                angle - MultiViewTriangulationMath.rayAngleDegrees(first, second, intrinsics)
            ) > 2.0
        )
    }

    @Test
    fun multiViewProjectorDltRecoversPointFromThreeCameras() {
        val worldPoint = doubleArrayOf(-0.3, 0.22, 5.4)
        val observations = listOf(
            view(0, 0.0, 0.0, worldPoint),
            view(1, 0.8, -0.1, worldPoint),
            view(2, 1.7, 0.15, worldPoint)
        )

        val estimate = MultiViewTriangulationMath.estimate(
            observations = observations,
            intrinsics = intrinsics,
            minimumParallaxDegrees = 0.1,
            maximumAngularErrorDegrees = 0.05,
            minimumInlierRatio = 1.0
        )

        assertNotNull(estimate)
        assertArrayEquals(worldPoint, estimate!!.pointWorld, 1e-6)
        assertEquals(listOf(0, 1, 2), estimate.inlierIndices)
        assertTrue(estimate.triangulationAngleDegrees > 0.1)
    }

    @Test
    fun boundedPairConsensusRejectsOneLargeAngularOutlierThenRefitsAllInliers() {
        val worldPoint = doubleArrayOf(0.12, -0.08, 6.0)
        val observations = listOf(
            view(0, 0.0, 0.0, worldPoint),
            view(1, 0.5, 0.0, worldPoint),
            view(2, 1.0, 0.1, worldPoint),
            view(3, 1.5, -0.1, worldPoint).let { it.copy(pixelX = it.pixelX + 120.0) }
        )

        val estimate = MultiViewTriangulationMath.estimate(
            observations = observations,
            intrinsics = intrinsics,
            minimumParallaxDegrees = 0.1,
            maximumAngularErrorDegrees = 1.0,
            minimumInlierRatio = 0.7
        )

        assertNotNull(estimate)
        assertEquals(listOf(0, 1, 2), estimate!!.inlierIndices)
        assertArrayEquals(worldPoint, estimate.pointWorld, 1e-5)
        assertTrue(estimate.localOptimizationPasses > 0)
    }

    @Test
    fun tracksWithAtMost15ViewsEnumerateEveryTwoViewHypothesis() {
        val worldPoint = doubleArrayOf(0.12, -0.08, 8.0)
        val observations = (0 until 10).map { frame ->
            view(frame, centerX = frame * 0.1, centerY = (frame % 2) * 0.02, point = worldPoint)
        }

        val estimate = MultiViewTriangulationMath.estimate(
            observations = observations,
            intrinsics = intrinsics,
            minimumParallaxDegrees = 0.1,
            maximumAngularErrorDegrees = 0.05,
            minimumInlierRatio = 1.0
        )

        assertNotNull(estimate)
        assertEquals(45, estimate!!.ransacTrials)
        assertEquals((0 until 10).toList(), estimate.inlierIndices)
        assertArrayEquals(worldPoint, estimate.pointWorld, 1e-6)
    }

    @Test
    fun longTrackUsesRepeatableConfidenceLimitedHypothesisBudget() {
        val worldPoint = doubleArrayOf(0.12, -0.08, 8.0)
        val observations = (0 until 20).map { frame ->
            view(frame, centerX = frame * 0.3, centerY = (frame % 3) * 0.03, point = worldPoint)
        }
        val estimate = MultiViewTriangulationMath.estimate(
            observations = observations,
            intrinsics = intrinsics,
            minimumParallaxDegrees = 1.5,
            maximumAngularErrorDegrees = 0.05,
            minimumInlierRatio = 1.0
        )
        val repeat = MultiViewTriangulationMath.estimate(
            observations = observations,
            intrinsics = intrinsics,
            minimumParallaxDegrees = 1.5,
            maximumAngularErrorDegrees = 0.05,
            minimumInlierRatio = 1.0
        )

        assertNotNull(estimate)
        assertNotNull(repeat)
        assertEquals(2, estimate!!.ransacTrials)
        assertEquals(estimate.ransacTrials, repeat!!.ransacTrials)
        assertEquals(estimate.inlierIndices, repeat.inlierIndices)
        assertArrayEquals(estimate.pointWorld, repeat.pointWorld, 1e-12)
    }

    @Test
    fun pixelConsensusRejectsModerateAngularResidualBeyondFinalPixelLimit() {
        val worldPoint = doubleArrayOf(0.12, -0.08, 6.0)
        val observations = listOf(
            view(0, 0.0, 0.0, worldPoint),
            view(1, 0.5, 0.0, worldPoint),
            view(2, 1.0, 0.1, worldPoint),
            view(3, 1.5, -0.1, worldPoint).let { it.copy(pixelX = it.pixelX + 20.0) }
        )

        val estimate = MultiViewTriangulationMath.estimate(
            observations = observations,
            intrinsics = intrinsics,
            minimumParallaxDegrees = 0.1,
            maximumAngularErrorDegrees = 2.0,
            minimumInlierRatio = 0.75,
            maximumReprojectionErrorPx = 4.0
        )

        assertNotNull(estimate)
        assertEquals(listOf(0, 1, 2), estimate!!.inlierIndices)
        assertArrayEquals(worldPoint, estimate.pointWorld, 1e-5)
    }

    @Test
    fun coincidentCameraCentersCannotSeedDepthEvenWithDifferentViewingRays() {
        val first = view(0, 0.0, 0.0, doubleArrayOf(0.0, 0.0, 5.0))
        val rotatedAtSameCenter = first.copy(
            pose = CameraPose(
                frameIndex = 1,
                rotation = doubleArrayOf(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                translation = doubleArrayOf(0.0, 0.0, 0.0),
                status = CameraPoseStatus.PNP_REGISTERED
            )
        )

        val estimate = MultiViewTriangulationMath.estimate(
            listOf(first, rotatedAtSameCenter),
            intrinsics,
            minimumParallaxDegrees = 0.001
        )

        assertNull(estimate)
    }

    @Test
    fun finalTriangulationStatsMatchFinalLandmarkCountAndMeans() {
        val initial = TriangulationStats(
            inputTracks = 5,
            triangulatedLandmarks = 2,
            rejectedShortTracks = 1,
            rejectedMissingPoses = 1,
            rejectedInvalidGeometry = 1,
            rejectedReprojection = 0,
            meanReprojectionErrorPx = 9.0,
            meanTriangulationAngleDegrees = 2.0
        )
        val finalLandmarks = listOf(
            CameraLandmark(10, Point3(1.0, 2.0, 3.0), 3, 0.5, 12.0),
            CameraLandmark(11, Point3(4.0, 5.0, 6.0), 4, 1.5, 18.0)
        )

        val result = initial.withFinalLandmarks(finalLandmarks)

        assertEquals(2, result.triangulatedLandmarks)
        assertEquals(1.0, result.meanReprojectionErrorPx, 1e-12)
        assertEquals(15.0, result.meanTriangulationAngleDegrees, 1e-12)
        assertEquals(5, result.inputTracks)
        assertEquals(1, result.rejectedShortTracks)
        assertEquals(1, result.rejectedMissingPoses)
        assertEquals(1, result.rejectedInvalidGeometry)
        assertEquals(0, result.rejectedReprojection)
    }

    private fun view(frame: Int, centerX: Double, centerY: Double, point: DoubleArray): TriangulationView {
        val pose = CameraPose(
            frameIndex = frame,
            rotation = IDENTITY.copyOf(),
            translation = doubleArrayOf(-centerX, -centerY, 0.0),
            status = if (frame == 0) CameraPoseStatus.REFERENCE else CameraPoseStatus.PNP_REGISTERED
        )
        val cameraPoint = doubleArrayOf(
            point[0] - centerX,
            point[1] - centerY,
            point[2]
        )
        val pixel = CameraProjectionMath.project(cameraPoint, intrinsics)
        return TriangulationView(pixel[0], pixel[1], pose)
    }

    private companion object {
        val IDENTITY = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    }
}
