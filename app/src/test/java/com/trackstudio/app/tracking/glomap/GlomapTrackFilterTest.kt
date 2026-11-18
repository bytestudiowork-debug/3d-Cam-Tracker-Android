package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GlomapTrackFilterTest {
    private val intrinsics = CameraIntrinsics(
        fx = 100.0, skew = 0.0, cx = 0.0, fy = 100.0, cy = 0.0
    )
    private val identity = doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )

    @Test
    fun reprojectionFilterUsesNormalizedResidualAndRejectsBehindCameraObservations() {
        val tracks = listOf(
            track(7, observation(0, 0.0, 0.0), observation(1, 2.0, 0.0)),
            track(8, observation(0, 0.0, 0.0))
        )
        val result = GlomapTrackFilter.filterTracksByReprojection(
            tracks = tracks,
            cameraCentersByFrame = mapOf(0 to center(0.0), 1 to center(0.0)),
            rotationsByFrame = mapOf(0 to identity, 1 to identity),
            pointsByTrackId = mapOf(
                7 to doubleArrayOf(0.0, 0.0, 10.0),
                8 to doubleArrayOf(0.0, 0.0, -10.0)
            ),
            intrinsics = intrinsics
        )

        assertEquals(2, result.changedTrackCount)
        assertEquals(2, result.removedObservationCount)
        assertEquals(listOf(0), result.tracks[0].observations.map { it.frameIndex })
        assertTrue(result.tracks[1].observations.isEmpty())
    }

    @Test
    fun angleFilterUsesSourceThresholdAndDoublesItWithoutFocalPrior() {
        val track = track(3, observation(0, 0.0, 0.0), observation(1, 3.0, 0.0))
        val centers = mapOf(0 to center(0.0), 1 to center(0.0))
        val rotations = mapOf(0 to identity, 1 to identity)
        val points = mapOf(3 to doubleArrayOf(0.0, 0.0, 10.0))

        val calibrated = GlomapTrackFilter.filterTracksByAngle(
            listOf(track), centers, rotations, points, intrinsics,
            hasPriorFocalLength = true
        )
        val uncalibrated = GlomapTrackFilter.filterTracksByAngle(
            listOf(track), centers, rotations, points, intrinsics,
            hasPriorFocalLength = false
        )

        assertEquals(listOf(0), calibrated.tracks.single().observations.map { it.frameIndex })
        assertEquals(listOf(0, 1), uncalibrated.tracks.single().observations.map { it.frameIndex })
    }

    @Test
    fun postPositioningUsesUncalibratedAngleToleranceWhenThereIsNoFocalPrior() {
        val track = track(
            4,
            observation(0, 100.0 * Math.tan(Math.toRadians(1.5)), 0.0),
            observation(1, -10.0, 0.0)
        )
        val centers = mapOf(0 to center(0.0), 1 to center(1.0))
        val rotations = mapOf(0 to identity, 1 to identity)
        val points = mapOf(4 to doubleArrayOf(0.0, 0.0, 10.0))

        val uncalibrated = GlomapTrackFilter.filterAfterPositioning(
            tracks = listOf(track),
            cameraCentersByFrame = centers,
            rotationsByFrame = rotations,
            pointsByTrackId = points,
            intrinsics = intrinsics,
            hasPriorFocalLength = false
        )
        val calibrated = GlomapTrackFilter.filterAfterPositioning(
            tracks = listOf(track),
            cameraCentersByFrame = centers,
            rotationsByFrame = rotations,
            pointsByTrackId = points,
            intrinsics = intrinsics,
            hasPriorFocalLength = true
        )

        assertEquals(listOf(0, 1), uncalibrated.single().observations.map { it.frameIndex })
        assertTrue(calibrated.single().observations.isEmpty())
    }

    @Test
    fun triangulationAngleFilterClearsTracksWithoutAViewBaseline() {
        val tracks = listOf(
            track(1, observation(0, 0.0, 0.0), observation(1, 0.0, 0.0)),
            track(2, observation(0, 0.0, 0.0), observation(2, 0.0, 0.0))
        )
        val result = GlomapTrackFilter.filterTrackTriangulationAngle(
            tracks = tracks,
            cameraCentersByFrame = mapOf(
                0 to doubleArrayOf(0.0, 0.0, 0.0),
                1 to doubleArrayOf(0.01, 0.0, 0.0),
                2 to doubleArrayOf(1.0, 0.0, 0.0)
            ),
            pointsByTrackId = mapOf(
                1 to doubleArrayOf(0.0, 0.0, 10.0),
                2 to doubleArrayOf(0.0, 0.0, 10.0)
            )
        )

        assertEquals(1, result.changedTrackCount)
        assertEquals(2, result.removedObservationCount)
        assertTrue(result.tracks[0].observations.isEmpty())
        assertEquals(2, result.tracks[1].observations.size)
    }

    private fun track(id: Int, vararg observations: CameraFeatureObservation) =
        CameraFeatureTrack(id, observations.toList())

    private fun observation(frame: Int, x: Double, y: Double) =
        CameraFeatureObservation(frame, x, y)

    private fun center(x: Double) = doubleArrayOf(x, 0.0, 0.0)
}
