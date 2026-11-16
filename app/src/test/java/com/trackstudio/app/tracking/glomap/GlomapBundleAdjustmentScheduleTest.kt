package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class GlomapBundleAdjustmentScheduleTest {
    private val intrinsics = CameraIntrinsics(
        fx = 100.0, skew = 0.0, cx = 50.0, fy = 100.0, cy = 50.0
    )

    @Test
    fun repeatedTrackChangesRunPositionThenRotationStagesAtProgressivelyTighterThresholds() {
        val stageModes = mutableListOf<Boolean>()
        val thresholds = mutableListOf<Double>()
        val result = GlomapBundleAdjustmentSchedule.solve(
            initialState = state(trackCount = 1_000),
            stageSolver = { current, optimizeRotations, _ ->
                stageModes += optimizeRotations
                solved(current)
            },
            filterByReprojection = { current, threshold ->
                thresholds += threshold
                GlomapTrackFilter.Result(current.tracks, changedTrackCount = 2, removedObservationCount = 2)
            },
            filterByTriangulationAngle = { current ->
                GlomapTrackFilter.Result(current.tracks, changedTrackCount = 0, removedObservationCount = 0)
            }
        )

        assertNotNull(result)
        assertEquals(listOf(false, true, false, true, false, true), stageModes)
        assertEquals(listOf(0.03, 0.02, 0.01, 0.01), thresholds)
        assertEquals(6, result!!.completedStages)
        assertEquals(24, result.bundleResult.iterations)
        assertEquals(10.0, result.bundleResult.initialCost, 0.0)
        assertEquals(9.0, result.bundleResult.finalCost, 0.0)
    }

    @Test
    fun smallCumulativeTrackChangesTightenFiltersThenStopAfterOneBaPair() {
        val stageModes = mutableListOf<Boolean>()
        val thresholds = mutableListOf<Double>()
        val result = GlomapBundleAdjustmentSchedule.solve(
            initialState = state(trackCount = 1_000),
            stageSolver = { current, optimizeRotations, _ ->
                stageModes += optimizeRotations
                solved(current)
            },
            filterByReprojection = { current, threshold ->
                thresholds += threshold
                GlomapTrackFilter.Result(current.tracks, changedTrackCount = 0, removedObservationCount = 0)
            },
            filterByTriangulationAngle = { current ->
                GlomapTrackFilter.Result(current.tracks, changedTrackCount = 0, removedObservationCount = 0)
            }
        )

        assertNotNull(result)
        assertEquals(listOf(false, true), stageModes)
        assertEquals(listOf(0.03, 0.02, 0.01, 0.01), thresholds)
        assertEquals(2, result!!.completedStages)
    }

    @Test
    fun sourceChangeCountIsCumulativeAcrossTheProgressiveFilterLoop() {
        val stageModes = mutableListOf<Boolean>()
        val thresholds = mutableListOf<Double>()
        val result = GlomapBundleAdjustmentSchedule.solve(
            initialState = state(trackCount = 1_000),
            stageSolver = { current, optimizeRotations, _ ->
                stageModes += optimizeRotations
                solved(current)
            },
            filterByReprojection = { current, threshold ->
                thresholds += threshold
                GlomapTrackFilter.Result(current.tracks, changedTrackCount = 1, removedObservationCount = 1)
            },
            filterByTriangulationAngle = { current ->
                GlomapTrackFilter.Result(current.tracks, changedTrackCount = 0, removedObservationCount = 0)
            }
        )

        assertNotNull(result)
        assertEquals(listOf(false, true, false, true), stageModes)
        assertEquals(listOf(0.03, 0.02, 0.01, 0.01), thresholds)
        assertEquals(4, result!!.completedStages)
    }

    @Test
    fun normalizesCameraCentersAndPointsBetweenBaIterations() {
        val stateAtNextIteration = mutableListOf<GlomapBundleAdjustmentSchedule.State>()
        val initial = state(trackCount = 1_000).copy(
            pointsByTrackId = mapOf(5 to doubleArrayOf(0.5, 2.0, 3.0))
        )

        val result = GlomapBundleAdjustmentSchedule.solve(
            initialState = initial,
            numIterations = 2,
            stageSolver = { current, _, stageIndex ->
                if (stageIndex == 2) stateAtNextIteration += current
                solved(current)
            },
            filterByReprojection = { current, _ ->
                GlomapTrackFilter.Result(current.tracks, changedTrackCount = 2, removedObservationCount = 0)
            },
            filterByTriangulationAngle = { current ->
                GlomapTrackFilter.Result(current.tracks, changedTrackCount = 0, removedObservationCount = 0)
            }
        )

        assertNotNull(result)
        val normalized = stateAtNextIteration.single()
        assertEquals(-5.0, normalized.frameCentersById.getValue(0)[0], 1e-12)
        assertEquals(5.0, normalized.frameCentersById.getValue(1)[0], 1e-12)
        assertEquals(0.0, normalized.pointsByTrackId.getValue(5)[0], 1e-12)
        assertEquals(-5.0, result!!.bundleResult.frameCentersById.getValue(0)[0], 1e-12)
        assertEquals(0.0, result.bundleResult.pointsByTrackId.getValue(5)[0], 1e-12)
    }

    private fun state(trackCount: Int): GlomapBundleAdjustmentSchedule.State =
        GlomapBundleAdjustmentSchedule.State(
            frameCentersById = mapOf(
                0 to doubleArrayOf(0.0, 0.0, 0.0),
                1 to doubleArrayOf(1.0, 0.0, 0.0)
            ),
            rotationsByFrame = mapOf(0 to identity(), 1 to identity()),
            pointsByTrackId = emptyMap(),
            tracks = List(trackCount) { CameraFeatureTrack(it, emptyList()) },
            intrinsics = intrinsics
        )

    private fun solved(state: GlomapBundleAdjustmentSchedule.State) =
        GlomapGlobalBundleAdjuster.Result(
            frameCentersById = state.frameCentersById,
            rotationsByFrame = state.rotationsByFrame,
            pointsByTrackId = state.pointsByTrackId,
            intrinsics = state.intrinsics,
            initialCost = 10.0,
            finalCost = 9.0,
            positiveDepthRatio = 1.0,
            iterations = 4,
            principalPointRefinementEnabled = false
        )

    private fun identity() = doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
}
