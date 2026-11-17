package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ColmapNextImageSelectorTest {
    @Test
    fun sixLevelPyramidScoresOnePointExactlyAsPinnedSource() {
        val score = ColmapNextImageSelector.visibilityScore(
            observations = listOf(observation(frame = 1, feature = 0, x = 50.0, y = 50.0)),
            imageWidth = 100,
            imageHeight = 100
        )

        // One occupied cell at each 2, 4, 8, 16, 32, and 64 grid level.
        assertEquals(5460L, score)
    }

    @Test
    fun minUncertaintyPrefersSpatiallyDistributedVisiblePoints() {
        val clustered = candidate(
            frame = 4,
            visiblePoints = List(31) { 10.0 to 10.0 },
            totalObservations = 60,
            trials = 1
        )
        val distributed = candidate(
            frame = 7,
            visiblePoints = (0 until 30).map { index ->
                (5.0 + (index % 5) * 18.0) to (5.0 + (index / 5) * 16.0)
            },
            totalObservations = 60,
            trials = 1
        )

        val ranked = ColmapNextImageSelector.rank(
            candidates = listOf(clustered, distributed),
            imageWidth = 100,
            imageHeight = 100
        )

        assertEquals(listOf(7, 4), ranked.map { it.frameIndex })
    }

    @Test
    fun untriedCandidatesPrecedeRetriedCandidatesEvenWithLowerCoverageScore() {
        val fresh = candidate(
            frame = 2,
            visiblePoints = List(30) { 10.0 to 10.0 },
            totalObservations = 40,
            trials = 0
        )
        val previouslyTried = candidate(
            frame = 5,
            visiblePoints = (0 until 80).map { index ->
                (5.0 + (index % 10) * 9.0) to (5.0 + (index / 10) * 12.0)
            },
            totalObservations = 90,
            trials = 1
        )

        val ranked = ColmapNextImageSelector.rank(
            listOf(previouslyTried, fresh), imageWidth = 100, imageHeight = 100
        )

        assertEquals(listOf(2, 5), ranked.map { it.frameIndex })
    }

    @Test
    fun appliesVisiblePointRegistrationTrialAndRegisteredFrameGates() {
        val candidates = listOf(
            candidate(0, List(29) { 20.0 to 20.0 }, 50, 0),
            candidate(1, List(30) { 20.0 to 20.0 }, 50, 3),
            candidate(2, List(30) { 20.0 to 20.0 }, 50, 0, isRegistered = true)
        )

        assertTrue(
            ColmapNextImageSelector.rank(candidates, imageWidth = 100, imageHeight = 100).isEmpty()
        )
    }

    @Test
    fun implementsVisibleCountAndVisibleRatioRankingOptions() {
        val morePoints = candidate(3, List(20) { 10.0 to 10.0 }, 100, 1)
        val higherRatio = candidate(8, List(10) { 80.0 to 80.0 }, 10, 1)
        val baseOptions = ColmapNextImageSelector.Options(minimumVisiblePoints = 1)

        val byCount = ColmapNextImageSelector.rank(
            listOf(higherRatio, morePoints), 100, 100,
            baseOptions.copy(method = ColmapNextImageSelector.Method.MAX_VISIBLE_POINTS_NUM)
        )
        val byRatio = ColmapNextImageSelector.rank(
            listOf(morePoints, higherRatio), 100, 100,
            baseOptions.copy(method = ColmapNextImageSelector.Method.MAX_VISIBLE_POINTS_RATIO)
        )

        assertEquals(listOf(3, 8), byCount.map { it.frameIndex })
        assertEquals(listOf(8, 3), byRatio.map { it.frameIndex })
    }

    private fun candidate(
        frame: Int,
        visiblePoints: List<Pair<Double, Double>>,
        totalObservations: Int,
        trials: Int,
        isRegistered: Boolean = false
    ) = ColmapNextImageSelector.Candidate(
        frameIndex = frame,
        visibleObservations = visiblePoints.mapIndexed { index, (x, y) ->
            observation(frame, index, x, y)
        },
        totalObservations = totalObservations,
        registrationTrials = trials,
        isRegistered = isRegistered
    )

    private fun observation(frame: Int, feature: Int, x: Double, y: Double) =
        CameraFeatureObservation(frame, x, y, feature)
}
