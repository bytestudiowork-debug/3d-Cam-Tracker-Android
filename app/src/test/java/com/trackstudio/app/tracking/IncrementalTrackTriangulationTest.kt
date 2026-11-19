package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point3

class IncrementalTrackTriangulationTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 800.0,
        fy = 800.0,
        skew = 0.0,
        cx = 640.0,
        cy = 360.0
    )
    private val worldPoint = doubleArrayOf(0.0, 0.0, 8.0)

    @Test
    fun finalPassContinuesAnExistingPointOntoAnUnownedVerifiedObservation() {
        val poses = (0..2).map { frame ->
            CameraPose(
                frameIndex = frame,
                rotation = IDENTITY.copyOf(),
                translation = doubleArrayOf(-frame.toDouble(), 0.0, 0.0),
                status = if (frame == 0) CameraPoseStatus.REFERENCE
                else CameraPoseStatus.PNP_REGISTERED
            )
        }
        val existingTrack = CameraFeatureTrack(
            id = 7,
            observations = listOf(observation(0), observation(1))
        )
        val featureLocations = mapOf((2 to 0) to observation(2))

        val result = IncrementalTrackTriangulation.finalize(
            featureTracks = listOf(existingTrack),
            landmarks = listOf(
                CameraLandmark(
                    trackId = existingTrack.id,
                    position = Point3(worldPoint[0], worldPoint[1], worldPoint[2]),
                    observationCount = 2,
                    meanReprojectionErrorPx = 0.0
                )
            ),
            pairMatches = listOf(
                com.trackstudio.app.tracking.colmap.ColmapTrackRetriangulator.PairMatches(
                    firstFrame = 1,
                    secondFrame = 2,
                    matches = listOf(0 to 0)
                )
            ),
            poses = poses,
            intrinsics = intrinsics,
            featureAt = { frame, feature -> featureLocations[frame to feature] }
        )

        assertEquals(1, result.addedObservations)
        assertEquals(0, result.createdTrackCount)
        assertEquals(setOf(existingTrack.id), result.changedTrackIds)
        assertEquals(listOf(existingTrack.id), result.triangulation.landmarks.map { it.trackId })
        assertEquals(
            listOf(0, 1, 2),
            result.triangulation.inlierTracks.single().observations.map { it.frameIndex }
        )
        assertTrue(result.triangulation.landmarks.single().meanReprojectionErrorPx < 1e-5)
    }

    private fun observation(frame: Int) = CameraFeatureObservation(
        frameIndex = frame,
        x = intrinsics.fx * (worldPoint[0] - frame.toDouble()) / worldPoint[2] + intrinsics.cx,
        y = intrinsics.cy,
        featureIndex = 0
    )

    private companion object {
        val IDENTITY = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    }
}
