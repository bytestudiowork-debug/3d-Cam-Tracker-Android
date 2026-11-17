package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SfMTriangulatorTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 760.0,
        fy = 810.0,
        skew = 18.0,
        cx = 320.0,
        cy = 240.0
    )

    @Test
    fun returnedLandmarkTrackContainsOnlyRobustlyAcceptedObservations() {
        val worldPoint = doubleArrayOf(0.12, -0.08, 6.0)
        val observations = (0..3).map { frame ->
            val centerX = frame * 0.5
            val pixel = CameraProjectionMath.project(
                doubleArrayOf(worldPoint[0] - centerX, worldPoint[1], worldPoint[2]),
                intrinsics
            )
            CameraFeatureObservation(
                frameIndex = frame,
                x = pixel[0] + if (frame == 3) 120.0 else 0.0,
                y = pixel[1],
                featureIndex = frame
            )
        }
        val track = CameraFeatureTrack(id = 17, observations = observations)
        val poses = (0..3).map { frame ->
            CameraPose(
                frameIndex = frame,
                rotation = IDENTITY.copyOf(),
                translation = doubleArrayOf(-frame * 0.5, 0.0, 0.0),
                status = if (frame == 0) CameraPoseStatus.REFERENCE
                else CameraPoseStatus.PNP_REGISTERED
            )
        }

        val result = SfMTriangulator.triangulate(
            featureTracks = listOf(track),
            poses = poses,
            intrinsics = intrinsics,
            maximumReprojectionErrorPx = 4.0,
            minimumParallaxDegrees = 0.1,
            minimumObservationInlierRatio = 0.70
        )

        assertEquals(1, result.landmarks.size)
        assertEquals(1, result.inlierTracks.size)
        val returnedTrack = result.inlierTracks.single()
        assertEquals(track.id, returnedTrack.id)
        assertEquals(listOf(0, 1, 2), returnedTrack.observations.map { it.frameIndex })
        assertEquals(result.landmarks.single().observationCount, returnedTrack.observations.size)
        assertEquals(4, track.observations.size)
        assertTrue(result.landmarks.single().meanReprojectionErrorPx < 1e-5)
    }

    private companion object {
        val IDENTITY = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    }
}
