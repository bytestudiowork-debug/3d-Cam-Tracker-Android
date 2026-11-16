package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GlomapPostRetriangulationRefinementTest {
    private val intrinsics = CameraIntrinsics(
        fx = 100.0, skew = 0.0, cx = 50.0, fy = 100.0, cy = 50.0
    )

    @Test
    fun fullPassRunsJointAdjustmentReprojectionAdjustmentThenFinalFilters() {
        val events = mutableListOf<String>()
        val thresholds = mutableListOf<Double>()
        var reprojectionCalls = 0

        val result = GlomapPostRetriangulationRefinement.refine(
            initialState = state(),
            solveBundleAdjustment = { current, solveIndex ->
                events += "ba$solveIndex:${current.tracks.single().id}"
                bundleResult(
                    current = current,
                    centerX = (solveIndex + 1).toDouble(),
                    initialCost = if (solveIndex == 0) 10.0 else 9.0,
                    finalCost = if (solveIndex == 0) 9.0 else 8.0
                )
            },
            filterByReprojection = { current, threshold ->
                reprojectionCalls++
                events += "reprojection$reprojectionCalls"
                thresholds += threshold
                val nextId = if (reprojectionCalls == 1) 8 else 9
                GlomapTrackFilter.Result(
                    tracks = current.tracks.map { it.copy(id = nextId) },
                    changedTrackCount = 1,
                    removedObservationCount = 1
                )
            },
            filterByTriangulationAngle = { current, threshold ->
                events += "angle:$threshold"
                GlomapTrackFilter.Result(
                    tracks = current.tracks.map { it.copy(id = 10) },
                    changedTrackCount = 0,
                    removedObservationCount = 0
                )
            }
        )

        assertEquals(
            listOf("ba0:7", "reprojection1", "ba1:8", "reprojection2", "angle:1.0"),
            events
        )
        assertEquals(listOf(1e-2, 1e-2), thresholds)
        assertEquals(2, result.acceptedBundleAdjustmentCount)
        assertTrue(result.firstBundleAdjustmentAccepted)
        assertTrue(result.secondBundleAdjustmentAccepted)
        assertEquals(0.0, result.state.frameCentersById.getValue(0)[0], 0.0)
        assertEquals(-2.0, result.state.pointsByTrackId.getValue(7)!![0], 0.0)
        assertEquals(-2.0, result.normalization.translation[0], 0.0)
        assertEquals(10, result.state.tracks.single().id)
        assertEquals(8.0, result.lastAcceptedBundleAdjustment?.finalCost ?: Double.NaN, 0.0)
    }

    @Test
    fun increasedFirstObjectiveFallsBackAndStillRunsFinalFilters() {
        val events = mutableListOf<String>()
        val initial = state()
        val result = GlomapPostRetriangulationRefinement.refine(
            initialState = initial,
            solveBundleAdjustment = { current, solveIndex ->
                events += "ba$solveIndex"
                bundleResult(current, centerX = 5.0, initialCost = 10.0, finalCost = 11.0)
            },
            filterByReprojection = { current, _ ->
                events += "reprojection"
                GlomapTrackFilter.Result(current.tracks, 0, 0)
            },
            filterByTriangulationAngle = { current, _ ->
                events += "angle"
                GlomapTrackFilter.Result(current.tracks, 0, 0)
            }
        )

        assertEquals(listOf("ba0", "reprojection", "angle"), events)
        assertEquals(0, result.acceptedBundleAdjustmentCount)
        assertFalse(result.firstBundleAdjustmentAccepted)
        assertNull(result.secondBundleAdjustment)
        assertEquals(0.0, result.state.frameCentersById.getValue(0)[0], 0.0)
    }

    @Test
    fun unavailableSecondSolveKeepsFirstAcceptedStateAndFilteredTracks() {
        val events = mutableListOf<String>()
        var reprojectionCalls = 0
        val result = GlomapPostRetriangulationRefinement.refine(
            initialState = state(),
            solveBundleAdjustment = { current, solveIndex ->
                events += "ba$solveIndex:${current.tracks.single().id}"
                if (solveIndex == 1) null
                else bundleResult(current, centerX = 1.0, initialCost = 10.0, finalCost = 9.0)
            },
            filterByReprojection = { current, _ ->
                reprojectionCalls++
                events += "reprojection$reprojectionCalls"
                val nextId = if (reprojectionCalls == 1) 8 else 9
                GlomapTrackFilter.Result(current.tracks.map { it.copy(id = nextId) }, 1, 1)
            },
            filterByTriangulationAngle = { current, _ ->
                events += "angle"
                GlomapTrackFilter.Result(current.tracks, 0, 0)
            }
        )

        assertEquals(
            listOf("ba0:7", "reprojection1", "ba1:8", "reprojection2", "angle"),
            events
        )
        assertEquals(1, result.acceptedBundleAdjustmentCount)
        assertTrue(result.firstBundleAdjustmentAccepted)
        assertFalse(result.secondBundleAdjustmentAccepted)
        assertEquals(0.0, result.state.frameCentersById.getValue(0)[0], 0.0)
        assertEquals(-1.0, result.state.pointsByTrackId.getValue(7)!![0], 0.0)
        assertEquals(-1.0, result.normalization.translation[0], 0.0)
        assertEquals(9, result.state.tracks.single().id)
    }

    private fun state() = GlomapPostRetriangulationRefinement.State(
        frameCentersById = mapOf(0 to doubleArrayOf(0.0, 0.0, 0.0)),
        rotationsByFrame = mapOf(0 to identity()),
        pointsByTrackId = mapOf(7 to doubleArrayOf(0.0, 0.0, 5.0)),
        tracks = listOf(
            CameraFeatureTrack(
                7,
                listOf(CameraFeatureObservation(0, 50.0, 50.0))
            )
        ),
        intrinsics = intrinsics
    )

    private fun bundleResult(
        current: GlomapPostRetriangulationRefinement.State,
        centerX: Double,
        initialCost: Double,
        finalCost: Double
    ) = GlomapGlobalBundleAdjuster.Result(
        frameCentersById = mapOf(0 to doubleArrayOf(centerX, 0.0, 0.0)),
        rotationsByFrame = current.rotationsByFrame,
        pointsByTrackId = current.pointsByTrackId,
        intrinsics = current.intrinsics,
        initialCost = initialCost,
        finalCost = finalCost,
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
