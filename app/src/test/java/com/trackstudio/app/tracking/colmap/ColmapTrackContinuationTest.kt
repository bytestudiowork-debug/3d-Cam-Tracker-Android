package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.PinholeIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.atan
import kotlin.math.tan

class ColmapTrackContinuationTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 800.0,
        fy = 800.0,
        skew = 0.0,
        cx = 640.0,
        cy = 360.0
    )

    @Test
    fun selectsTheLowestAngularErrorPointAcrossDirectFindCandidates() {
        val tracks = listOf(
            pointTrack(trackId = 2, pointX = 0.0, frames = listOf(0 to 0, 3 to 3)),
            pointTrack(trackId = 8, pointX = 0.1, frames = listOf(1 to 1, 3 to 4)),
            CameraFeatureTrack(
                12,
                listOf(observation(2, 0, centerX = 2.0, pointX = 0.0, angularOffsetDegrees = 1.5))
            )
        )
        val result = ColmapTrackContinuation.continueTracks(
            tracks = tracks,
            pointsByTrackId = mapOf(
                2 to doubleArrayOf(0.0, 0.0, 8.0),
                8 to doubleArrayOf(0.1, 0.0, 8.0)
            ),
            pairMatches = listOf(
                ColmapTrackContinuation.PairMatches(2, 0, listOf(0 to 0)),
                ColmapTrackContinuation.PairMatches(2, 1, listOf(0 to 1))
            ),
            poses = poses(4),
            intrinsics = intrinsics
        )

        assertEquals(1, result.addedObservations)
        val byId = result.tracks.associateBy { it.id }
        assertEquals(listOf(0, 3), byId.getValue(2).observations.map { it.frameIndex })
        assertEquals(listOf(1, 2, 3), byId.getValue(8).observations.map { it.frameIndex })
        assertEquals(null, byId[12])
    }

    @Test
    fun rejectsContinuationOutsideThePinnedTwoDegreeThreshold() {
        val tracks = listOf(
            pointTrack(trackId = 2, pointX = 0.0, frames = listOf(0 to 0, 1 to 1)),
            CameraFeatureTrack(
                9,
                listOf(observation(2, 0, centerX = 2.0, pointX = 0.0, angularOffsetDegrees = 2.1))
            )
        )
        val result = ColmapTrackContinuation.continueTracks(
            tracks = tracks,
            pointsByTrackId = mapOf(2 to doubleArrayOf(0.0, 0.0, 8.0)),
            pairMatches = listOf(ColmapTrackContinuation.PairMatches(1, 2, listOf(1 to 0))),
            poses = poses(3),
            intrinsics = intrinsics
        )

        assertEquals(0, result.addedObservations)
        assertEquals(listOf(0, 1), result.tracks.first { it.id == 2 }.observations.map { it.frameIndex })
        assertEquals(listOf(2), result.tracks.first { it.id == 9 }.observations.map { it.frameIndex })
    }

    @Test
    fun findUsesOneHopByDefaultAndSupportsTheConfiguredTransitivity() {
        val tracks = listOf(
            pointTrack(trackId = 2, pointX = 0.0, frames = listOf(0 to 0, 3 to 3)),
            CameraFeatureTrack(9, listOf(observation(2, 0, centerX = 2.0, pointX = 0.0)))
        )
        val pairMatches = listOf(
            ColmapTrackContinuation.PairMatches(2, 1, listOf(0 to 1)),
            ColmapTrackContinuation.PairMatches(1, 0, listOf(1 to 0))
        )
        val points = mapOf(2 to doubleArrayOf(0.0, 0.0, 8.0))
        val poses = poses(4)

        val defaultResult = ColmapTrackContinuation.continueTracks(
            tracks, points, pairMatches, poses, intrinsics
        )
        val twoHopResult = ColmapTrackContinuation.continueTracks(
            tracks, points, pairMatches, poses, intrinsics, maximumTransitivity = 2
        )

        assertEquals(0, defaultResult.addedObservations)
        assertEquals(1, twoHopResult.addedObservations)
        assertEquals(listOf(0, 2, 3), twoHopResult.tracks.first { it.id == 2 }.observations.map { it.frameIndex })
    }

    private fun pointTrack(
        trackId: Int,
        pointX: Double,
        frames: List<Pair<Int, Int>>
    ) = CameraFeatureTrack(
        trackId,
        frames.map { (frame, feature) ->
            observation(frame, feature, centerX = frame.toDouble(), pointX = pointX)
        }
    )

    private fun observation(
        frame: Int,
        feature: Int,
        centerX: Double,
        pointX: Double,
        angularOffsetDegrees: Double = 0.0
    ): CameraFeatureObservation {
        val idealAngle = atan((pointX - centerX) / 8.0)
        val observedAngle = idealAngle + Math.toRadians(angularOffsetDegrees)
        return CameraFeatureObservation(
            frameIndex = frame,
            x = intrinsics.cx + intrinsics.fx * tan(observedAngle),
            y = intrinsics.cy,
            featureIndex = feature
        )
    }

    private fun poses(count: Int): List<CameraPose> = (0 until count).map { frame ->
        CameraPose(
            frameIndex = frame,
            rotation = IDENTITY.copyOf(),
            translation = doubleArrayOf(-frame.toDouble(), 0.0, 0.0),
            status = CameraPoseStatus.RELATIVE_ESTIMATE
        )
    }

    private companion object {
        val IDENTITY = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    }
}
