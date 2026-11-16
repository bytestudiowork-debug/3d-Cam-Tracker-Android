package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.PinholeIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point3

class ColmapTrackCompleterTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 100.0,
        fy = 100.0,
        skew = 0.0,
        cx = 50.0,
        cy = 50.0
    )

    @Test
    fun completesVerifiedObservationsRecursivelyUpToFiveHops() {
        val track = CameraFeatureTrack(7, listOf(observation(0, 0)))
        val locations = (0..6).associate { frame ->
            (frame to 0) to observation(frame, 0, 50.0, 50.0)
        }
        val edges = (0 until 6).map { frame ->
            ColmapTrackCompleter.PairMatches(frame, frame + 1, listOf(0 to 0))
        }

        val result = ColmapTrackCompleter.completeTracks(
            tracks = listOf(track),
            landmarks = listOf(landmark(7)),
            pairMatches = edges,
            activeFrames = (0..6).toSet(),
            poses = (0..6).map(::pose),
            intrinsics = intrinsics,
            featureAt = { frame, feature -> locations[frame to feature] },
            minimumPairMatches = 1
        )

        assertEquals(5, result.addedObservations)
        assertEquals(listOf(0, 1, 2, 3, 4, 5), result.tracks.single().observations.map { it.frameIndex })
        assertTrue(result.tracks.single().observations.none { it.frameIndex == 6 })
    }

    @Test
    fun acceptsTheFifteenPixelThresholdAndRejectsLargerReprojectionError() {
        val track = CameraFeatureTrack(3, listOf(observation(0, 0)))
        val locations = mapOf(
            (1 to 0) to observation(1, 0, 65.0, 50.0),
            (2 to 0) to observation(2, 0, 65.0001, 50.0)
        )
        val result = ColmapTrackCompleter.completeTracks(
            tracks = listOf(track),
            landmarks = listOf(landmark(3)),
            pairMatches = listOf(
                ColmapTrackCompleter.PairMatches(0, 1, listOf(0 to 0)),
                ColmapTrackCompleter.PairMatches(0, 2, listOf(0 to 0))
            ),
            activeFrames = setOf(0, 1, 2),
            poses = (0..2).map(::pose),
            intrinsics = intrinsics,
            featureAt = { frame, feature -> locations[frame to feature] },
            minimumPairMatches = 1
        )

        assertEquals(1, result.addedObservations)
        assertEquals(listOf(0, 1), result.tracks.single().observations.map { it.frameIndex })
    }

    @Test
    fun doesNotClaimAnObservationOwnedByAnotherTriangulatedTrack() {
        val tracks = listOf(
            CameraFeatureTrack(1, listOf(observation(0, 0))),
            CameraFeatureTrack(2, listOf(observation(1, 1)))
        )
        val result = ColmapTrackCompleter.completeTracks(
            tracks = tracks,
            landmarks = listOf(landmark(1), landmark(2)),
            pairMatches = listOf(ColmapTrackCompleter.PairMatches(0, 1, listOf(0 to 1))),
            activeFrames = setOf(0, 1),
            poses = listOf(pose(0), pose(1)),
            intrinsics = intrinsics,
            featureAt = { frame, feature -> observation(frame, feature) },
            minimumPairMatches = 1
        )

        assertEquals(0, result.addedObservations)
        assertEquals(tracks, result.tracks)
    }

    @Test
    fun skipsFeaturesFromFramesOutsideTheSolvedPoseSet() {
        val track = CameraFeatureTrack(4, listOf(observation(0, 0)))
        val result = ColmapTrackCompleter.completeTracks(
            tracks = listOf(track),
            landmarks = listOf(landmark(4)),
            pairMatches = listOf(ColmapTrackCompleter.PairMatches(0, 1, listOf(0 to 0))),
            activeFrames = setOf(0, 1),
            poses = listOf(pose(0)),
            intrinsics = intrinsics,
            featureAt = { frame, feature -> observation(frame, feature) },
            minimumPairMatches = 1
        )

        assertEquals(0, result.addedObservations)
        assertEquals(track, result.tracks.single())
    }

    private fun observation(
        frame: Int,
        feature: Int,
        x: Double = 50.0,
        y: Double = 50.0
    ) = CameraFeatureObservation(frame, x, y, feature)

    private fun landmark(trackId: Int) = CameraLandmark(
        trackId = trackId,
        position = Point3(0.0, 0.0, 10.0),
        observationCount = 2,
        meanReprojectionErrorPx = 0.0
    )

    private fun pose(frame: Int) = CameraPose(
        frameIndex = frame,
        rotation = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation = doubleArrayOf(0.0, 0.0, 0.0),
        status = CameraPoseStatus.RELATIVE_ESTIMATE
    )
}
