package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.PinholeIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point3

class ColmapTrackMergerTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 100.0,
        fy = 100.0,
        skew = 0.0,
        cx = 50.0,
        cy = 50.0
    )

    @Test
    fun mergesVerifiedTracksUsingTrackLengthWeightedPosition() {
        val tracks = listOf(
            track(2, listOf(0 to 0, 1 to 1), pointX = 0.0),
            track(7, listOf(2 to 2, 3 to 3, 4 to 4), pointX = 0.06)
        )
        val landmarks = listOf(landmark(2, 0.0), landmark(7, 0.06))

        val result = ColmapTrackMerger.mergeTracks(
            tracks = tracks,
            landmarks = landmarks,
            pairMatches = listOf(pair(1, 2, 1, 2)),
            poses = (0..4).map(::pose),
            intrinsics = intrinsics
        )

        assertEquals(1, result.mergedTrackCount)
        assertEquals(setOf(7), result.removedTrackIds)
        assertEquals(listOf(2), result.tracks.map { it.id })
        assertEquals((0..4).toList(), result.tracks.single().observations.map { it.frameIndex })
        assertEquals(1, result.landmarks.size)
        assertEquals(0.036, result.landmarks.single().position.x, 1e-12)
        assertEquals(5, result.landmarks.single().observationCount)
        assertEquals(1, result.mergedLandmarks.size)
    }

    @Test
    fun rejectsMergeWhenAnyObservationExceedsTheReprojectionThreshold() {
        val first = CameraFeatureTrack(1, listOf(observation(0, 0, 50.0)))
        val second = CameraFeatureTrack(
            2,
            listOf(observation(1, 1, 50.6), observation(2, 2, 70.0))
        )
        val tracks = listOf(first, second)
        val landmarks = listOf(landmark(1, 0.0), landmark(2, 0.06))

        val result = ColmapTrackMerger.mergeTracks(
            tracks = tracks,
            landmarks = landmarks,
            pairMatches = listOf(pair(0, 1, 0, 1)),
            poses = (0..2).map(::pose),
            intrinsics = intrinsics
        )

        assertEquals(0, result.mergedTrackCount)
        assertTrue(result.removedTrackIds.isEmpty())
        assertEquals(tracks, result.tracks)
        assertEquals(landmarks, result.landmarks)
        assertTrue(result.mergedLandmarks.isEmpty())
    }

    @Test
    fun repeatsMergeFromTheCombinedTrackToReachConnectedPoints() {
        val tracks = listOf(
            track(3, listOf(0 to 0, 1 to 1), 0.0),
            track(5, listOf(2 to 2, 3 to 3), 0.04),
            track(8, listOf(4 to 4, 5 to 5), 0.08)
        )
        val landmarks = listOf(landmark(3, 0.0), landmark(5, 0.04), landmark(8, 0.08))
        val edges = listOf(pair(1, 2, 1, 2), pair(3, 4, 3, 4))

        val result = ColmapTrackMerger.mergeTracks(
            tracks = tracks,
            landmarks = landmarks,
            pairMatches = edges,
            poses = (0..5).map(::pose),
            intrinsics = intrinsics
        )

        assertEquals(2, result.mergedTrackCount)
        assertEquals(setOf(5, 8), result.removedTrackIds)
        assertEquals(listOf(3), result.tracks.map { it.id })
        assertEquals(6, result.tracks.single().observations.size)
        assertEquals(0.04, result.landmarks.single().position.x, 1e-12)
    }

    @Test
    fun acceptsTheConfiguredReprojectionBoundary() {
        val tracks = listOf(
            CameraFeatureTrack(0, listOf(observation(0, 0, 50.0), observation(1, 1, 50.0))),
            CameraFeatureTrack(1, listOf(observation(2, 2, 58.0), observation(3, 3, 58.0)))
        )
        val result = ColmapTrackMerger.mergeTracks(
            tracks = tracks,
            landmarks = listOf(landmark(0, 0.0), landmark(1, 0.8)),
            pairMatches = listOf(pair(1, 2, 1, 2)),
            poses = (0..3).map(::pose),
            intrinsics = intrinsics,
            maximumReprojectionErrorPx = 4.0
        )

        assertEquals(1, result.mergedTrackCount)
        assertEquals(0.4, result.landmarks.single().position.x, 1e-12)
    }

    @Test
    fun leavesTracksUnmergedWhenTheyAlreadyContainTheSameFrame() {
        val tracks = listOf(
            CameraFeatureTrack(1, listOf(observation(0, 0, 50.0), observation(1, 1, 50.0))),
            CameraFeatureTrack(2, listOf(observation(1, 2, 50.0), observation(2, 3, 50.0)))
        )
        val landmarks = listOf(landmark(1, 0.0), landmark(2, 0.0))
        val result = ColmapTrackMerger.mergeTracks(
            tracks = tracks,
            landmarks = landmarks,
            pairMatches = listOf(pair(0, 2, 0, 3)),
            poses = (0..2).map(::pose),
            intrinsics = intrinsics
        )

        assertEquals(0, result.mergedTrackCount)
        assertEquals(tracks, result.tracks)
        assertEquals(landmarks, result.landmarks)
    }

    @Test
    fun ignoresPointsWithoutACompleteActiveTrackAndDoesNotMutateInputs() {
        val sourceTracks = listOf(
            CameraFeatureTrack(1, listOf(observation(0, 0))),
            CameraFeatureTrack(2, listOf(observation(1, 1)))
        )
        val sourceLandmarks = listOf(landmark(1, 0.0))
        val result = ColmapTrackMerger.mergeTracks(
            tracks = sourceTracks,
            landmarks = sourceLandmarks,
            pairMatches = listOf(pair(0, 1, 0, 1)),
            poses = listOf(pose(0), pose(1)),
            intrinsics = intrinsics
        )

        assertSame(sourceTracks, result.tracks)
        assertSame(sourceLandmarks, result.landmarks)
        assertFalse(result.mergedTrackCount > 0)
    }

    private fun pair(firstFrame: Int, secondFrame: Int, firstFeature: Int, secondFeature: Int) =
        ColmapTrackMerger.PairMatches(firstFrame, secondFrame, listOf(firstFeature to secondFeature))

    private fun track(id: Int, observations: List<Pair<Int, Int>>, pointX: Double) =
        CameraFeatureTrack(id, observations.map { (frame, feature) ->
            val pixelX = 100.0 * pointX / 10.0 + 50.0
            observation(frame, feature, pixelX)
        })

    private fun observation(frame: Int, feature: Int, x: Double = 50.0) =
        CameraFeatureObservation(frame, x, 50.0, feature)

    private fun landmark(id: Int, pointX: Double) = CameraLandmark(
        trackId = id,
        position = Point3(pointX, 0.0, 10.0),
        observationCount = 2,
        meanReprojectionErrorPx = 0.0,
        triangulationAngleDegrees = id.toDouble()
    )

    private fun pose(frame: Int) = CameraPose(
        frameIndex = frame,
        rotation = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation = doubleArrayOf(0.0, 0.0, 0.0),
        status = CameraPoseStatus.RELATIVE_ESTIMATE
    )
}
