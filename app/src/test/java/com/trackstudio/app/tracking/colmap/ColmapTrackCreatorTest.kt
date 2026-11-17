package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.CameraProjectionMath
import com.trackstudio.app.tracking.PinholeIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ColmapTrackCreatorTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 800.0,
        fy = 810.0,
        skew = 0.0,
        cx = 640.0,
        cy = 360.0
    )

    @Test
    fun recursivelyCreatesAnotherPointFromThreeResidualObservations() {
        val track = mixedTrack(secondClusterFrameCount = 3)
        val result = ColmapTrackCreator.create(
            tracks = listOf(track),
            poses = poses(6),
            intrinsics = intrinsics
        )

        assertEquals(2, result.tracks.size)
        assertEquals(2, result.landmarks.size)
        assertEquals(1, result.additionalTrackCount)
        assertEquals(6, result.tracks.sumOf { it.observations.size })
        assertEquals(
            setOf(setOf(0, 1, 2), setOf(3, 4, 5)),
            result.tracks.map { component ->
                component.observations.mapTo(linkedSetOf()) { it.frameIndex }
            }.toSet()
        )
        val recoveredX = result.landmarks.map { it.position.x }.sorted()
        assertTrue(abs(recoveredX[0]) < 1e-4)
        assertTrue(abs(recoveredX[1] - 4.0) < 1e-4)
        assertTrue(result.landmarks.all { it.observationCount == 3 })
    }

    @Test
    fun doesNotRetryWhenOnlyTwoResidualObservationsRemain() {
        val track = mixedTrack(secondClusterFrameCount = 2)
        val result = ColmapTrackCreator.create(
            tracks = listOf(track),
            poses = poses(5),
            intrinsics = intrinsics
        )

        assertEquals(1, result.tracks.size)
        assertEquals(1, result.landmarks.size)
        assertEquals(0, result.additionalTrackCount)
        assertEquals(3, result.tracks.single().observations.size)
        assertEquals(setOf(0, 1, 2), result.tracks.single().observations.map { it.frameIndex }.toSet())
        assertEquals(1, result.unassignedTracks.size)
        assertEquals(setOf(3, 4), result.unassignedTracks.single().observations.map { it.frameIndex }.toSet())
    }

    private fun mixedTrack(secondClusterFrameCount: Int): CameraFeatureTrack {
        val firstPoint = doubleArrayOf(0.0, 0.0, 8.0)
        val secondPoint = doubleArrayOf(4.0, 0.0, 8.0)
        val observationCount = 3 + secondClusterFrameCount
        val observations = (0 until observationCount).map { frame ->
            val point = if (frame < 3) firstPoint else secondPoint
            val centerX = frame * 0.5
            val pixel = CameraProjectionMath.project(
                doubleArrayOf(point[0] - centerX, point[1], point[2]), intrinsics
            )
            CameraFeatureObservation(frame, pixel[0], pixel[1], frame)
        }
        return CameraFeatureTrack(id = 9, observations = observations)
    }

    private fun poses(count: Int): List<CameraPose> = (0 until count).map { frame ->
        val centerX = frame * 0.5
        CameraPose(
            frameIndex = frame,
            rotation = doubleArrayOf(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ),
            translation = doubleArrayOf(-centerX, 0.0, 0.0),
            status = CameraPoseStatus.RELATIVE_ESTIMATE
        )
    }
}
