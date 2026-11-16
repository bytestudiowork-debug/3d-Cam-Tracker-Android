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
import kotlin.math.atan
import kotlin.math.tan

class ColmapTrackRetriangulatorTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 800.0,
        fy = 800.0,
        skew = 0.0,
        cx = 640.0,
        cy = 360.0
    )
    private val worldPoint = doubleArrayOf(0.0, 0.0, 8.0)

    @Test
    fun createsAnUnderReconstructedPointThenContinuesItThroughThePairGraph() {
        val locations = (0..2).associate { frame ->
            (frame to 0) to observation(frame, 0, centerX = frame * 1.5)
        }
        val result = ColmapTrackRetriangulator.retriangulate(
            tracks = emptyList(),
            landmarks = emptyList(),
            pairMatches = listOf(
                ColmapTrackRetriangulator.PairMatches(0, 1, listOf(0 to 0)),
                ColmapTrackRetriangulator.PairMatches(1, 2, listOf(0 to 0))
            ),
            poses = (0..2).map { pose(it, it * 1.5) },
            intrinsics = intrinsics,
            featureAt = { frame, feature -> locations[frame to feature] }
        )

        assertEquals(1, result.createdTracks)
        assertEquals(3, result.addedObservations)
        assertEquals(2, result.underReconstructedPairCount)
        assertEquals(2, result.processedPairCount)
        assertEquals(listOf(0, 1, 2), result.tracks.single().observations.map { it.frameIndex })
    }

    @Test
    fun ignoresPairOnlyObservationsByDefault() {
        val locations = mapOf(
            (0 to 0) to observation(0, 0, centerX = 0.0),
            (1 to 0) to observation(1, 0, centerX = 1.5)
        )
        val result = ColmapTrackRetriangulator.retriangulate(
            tracks = emptyList(),
            landmarks = emptyList(),
            pairMatches = listOf(
                ColmapTrackRetriangulator.PairMatches(0, 1, listOf(0 to 0))
            ),
            poses = listOf(pose(0, 0.0), pose(1, 1.5)),
            intrinsics = intrinsics,
            featureAt = { frame, feature -> locations[frame to feature] }
        )

        assertEquals(0, result.createdTracks)
        assertEquals(0, result.addedObservations)
        assertTrue(result.tracks.isEmpty())
    }

    @Test
    fun skipsPairsAtTheMinimumTriangulatedRatio() {
        val track = CameraFeatureTrack(
            4,
            listOf(observation(0, 0, 0.0), observation(1, 0, 1.5))
        )
        val verifiedMatches = listOf(0 to 0, 1 to 1, 2 to 2, 3 to 3, 4 to 4)
        val result = ColmapTrackRetriangulator.retriangulate(
            tracks = listOf(track),
            landmarks = listOf(landmark(4)),
            pairMatches = listOf(
                ColmapTrackRetriangulator.PairMatches(0, 1, verifiedMatches)
            ),
            poses = listOf(pose(0, 0.0), pose(1, 1.5)),
            intrinsics = intrinsics,
            featureAt = { frame, feature -> observation(frame, feature, frame * 1.5) }
        )

        assertEquals(0, result.underReconstructedPairCount)
        assertEquals(0, result.processedPairCount)
        assertEquals(listOf(track), result.tracks)
    }

    @Test
    fun continuesAnExistingPointOnlyWithinTheFiveDegreeAngularLimit() {
        val resultAtFourDegrees = continueExistingPoint(4.0)
        val resultAtSixDegrees = continueExistingPoint(6.0)

        assertEquals(1, resultAtFourDegrees.addedObservations)
        assertEquals(listOf(0, 1, 2), resultAtFourDegrees.tracks.single().observations.map { it.frameIndex })
        assertEquals(0, resultAtSixDegrees.addedObservations)
        assertEquals(listOf(0, 1), resultAtSixDegrees.tracks.single().observations.map { it.frameIndex })
    }

    @Test
    fun doesNotMergeTwoDifferentExistingPointsDuringRetriangulation() {
        val tracks = listOf(
            CameraFeatureTrack(2, listOf(observation(0, 0, 0.0), observation(2, 1, 3.0))),
            CameraFeatureTrack(8, listOf(observation(1, 0, 1.5), observation(2, 2, 3.0)))
        )
        val result = ColmapTrackRetriangulator.retriangulate(
            tracks = tracks,
            landmarks = listOf(landmark(2), landmark(8, Point3(0.2, 0.0, 8.0))),
            pairMatches = listOf(
                ColmapTrackRetriangulator.PairMatches(0, 1, listOf(0 to 0))
            ),
            poses = (0..2).map { pose(it, it * 1.5) },
            intrinsics = intrinsics,
            featureAt = { frame, feature -> observation(frame, feature, frame * 1.5) }
        )

        assertEquals(0, result.addedObservations)
        assertEquals(0, result.createdTracks)
        assertEquals(tracks, result.tracks)
    }

    @Test
    fun doesNotDuplicateAnObservationAlreadyInAnUntriangulatedTrack() {
        val track = CameraFeatureTrack(5, listOf(observation(1, 0, 1.5), observation(2, 1, 3.0)))
        val locations = mapOf((0 to 0) to observation(0, 0, 0.0))
        val result = ColmapTrackRetriangulator.retriangulate(
            tracks = listOf(track),
            landmarks = emptyList(),
            pairMatches = listOf(
                ColmapTrackRetriangulator.PairMatches(0, 1, listOf(0 to 0))
            ),
            poses = (0..2).map { pose(it, it * 1.5) },
            intrinsics = intrinsics,
            featureAt = { frame, feature -> locations[frame to feature] }
        )

        assertEquals(0, result.createdTracks)
        assertEquals(listOf(track), result.tracks)
    }

    private fun continueExistingPoint(errorDegrees: Double): ColmapTrackRetriangulator.Result {
        val existing = CameraFeatureTrack(
            7,
            listOf(observation(0, 0, 0.0), observation(1, 0, 1.0))
        )
        val candidateX = angularOffsetPixelX(centerX = 2.0, errorDegrees = errorDegrees)
        val candidate = observation(2, 0, centerX = 2.0, pixelX = candidateX)
        return ColmapTrackRetriangulator.retriangulate(
            tracks = listOf(existing),
            landmarks = listOf(landmark(7)),
            pairMatches = listOf(
                ColmapTrackRetriangulator.PairMatches(1, 2, listOf(0 to 0))
            ),
            poses = listOf(pose(0, 0.0), pose(1, 1.0), pose(2, 2.0)),
            intrinsics = intrinsics,
            featureAt = { frame, feature ->
                if (frame == 2 && feature == 0) candidate
                else observation(frame, feature, frame.toDouble())
            }
        )
    }

    private fun observation(
        frame: Int,
        feature: Int,
        centerX: Double,
        pixelX: Double = intrinsics.fx * (worldPoint[0] - centerX) / worldPoint[2] + intrinsics.cx
    ) = CameraFeatureObservation(frame, pixelX, intrinsics.cy, feature)

    private fun pose(frame: Int, centerX: Double) = CameraPose(
        frameIndex = frame,
        rotation = IDENTITY.copyOf(),
        translation = doubleArrayOf(-centerX, 0.0, 0.0),
        status = if (frame == 0) CameraPoseStatus.REFERENCE else CameraPoseStatus.RELATIVE_ESTIMATE
    )

    private fun landmark(trackId: Int, position: Point3 = Point3(worldPoint[0], worldPoint[1], worldPoint[2])) =
        CameraLandmark(
            trackId = trackId,
            position = position,
            observationCount = 2,
            meanReprojectionErrorPx = 0.0
        )

    private fun angularOffsetPixelX(centerX: Double, errorDegrees: Double): Double {
        val idealAngle = atan((worldPoint[0] - centerX) / worldPoint[2])
        val observedRayAngle = idealAngle + Math.toRadians(errorDegrees)
        return intrinsics.cx + intrinsics.fx * tan(observedRayAngle)
    }

    private companion object {
        val IDENTITY = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    }
}
