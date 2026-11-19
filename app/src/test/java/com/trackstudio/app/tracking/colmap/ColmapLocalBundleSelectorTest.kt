package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.tan

class ColmapLocalBundleSelectorTest {
    @Test
    fun underBudgetNeighborsBypassTriangulationAngleGate() {
        val scene = scene(
            NeighborSpec(1, 1.0, listOf(1.0, 1.0)),
            NeighborSpec(2, 2.0, listOf(1.0))
        )

        val result = select(scene, maxImages = 3, minimumAngleDegrees = 80.0)

        assertEquals(listOf(0, 1, 2), result.frameIds)
        assertEquals(listOf(1, 2), result.neighborFrameIds)
        assertTrue(result.triangulationAngle75RadiansByFrame.isEmpty())
    }

    @Test
    fun relaxesAngleAndSharedTrackThresholdsInPinnedOrder() {
        val scene = scene(
            NeighborSpec(1, 1.0, List(12) { 3.5 }),
            NeighborSpec(2, 2.0, List(8) { 1.0 })
        )

        val result = select(scene, maxImages = 2)

        assertEquals(20, result.queryPointCount)
        assertEquals(listOf(1), result.neighborFrameIds)
        assertEquals(12, result.sharedObservationsByFrame[1])
        assertEquals(8, result.sharedObservationsByFrame[2])
        assertEquals(3.5, Math.toDegrees(result.triangulationAngle75RadiansByFrame.getValue(1)), 1e-6)
    }

    @Test
    fun computesTheInterpolatedSeventyFifthPercentile() {
        val scene = scene(
            NeighborSpec(1, 1.0, listOf(10.0, 20.0, 30.0, 80.0)),
            NeighborSpec(2, 2.0, listOf(1.0))
        )

        val result = select(scene, maxImages = 2)

        assertEquals(listOf(1), result.neighborFrameIds)
        assertEquals(42.5, Math.toDegrees(result.triangulationAngle75RadiansByFrame.getValue(1)), 1e-6)
    }

    @Test
    fun fillsUnselectedSlotsByOverlapAfterAllAnglePassesFail() {
        val scene = scene(
            NeighborSpec(1, 1.0, List(5) { 1.0 }),
            NeighborSpec(2, 2.0, List(4) { 2.0 }),
            NeighborSpec(3, 3.0, List(3) { 3.0 })
        )

        val result = select(scene, maxImages = 3, minimumAngleDegrees = 100.0)

        assertEquals(listOf(1, 2), result.neighborFrameIds)
        assertEquals(5, result.sharedObservationsByFrame[1])
        assertEquals(4, result.sharedObservationsByFrame[2])
    }

    @Test
    fun unavailableAdjustmentFiltersSelectedImagesOnlyWhenLocalBundleHasNeighbors() {
        val connected = ColmapLocalBundleSelector.Selection(
            referenceFrameId = 3,
            neighborFrameIds = listOf(5, 8),
            sharedObservationsByFrame = mapOf(5 to 30, 8 to 20),
            triangulationAngle75RadiansByFrame = emptyMap(),
            queryPointCount = 40
        )
        val disconnected = connected.copy(neighborFrameIds = emptyList())

        assertEquals(
            setOf(3, 5, 8),
            ColmapLocalBundleSelector.filterFrameIdsAfterUnavailableAdjustment(connected)
        )
        assertEquals(
            emptySet<Int>(),
            ColmapLocalBundleSelector.filterFrameIdsAfterUnavailableAdjustment(disconnected)
        )
    }

    private fun select(
        scene: Scene,
        maxImages: Int = ColmapLocalBundleSelector.DEFAULT_MAX_NUM_IMAGES,
        minimumAngleDegrees: Double = ColmapLocalBundleSelector.DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES
    ) = ColmapLocalBundleSelector.select(
        referenceFrameId = 0,
        cameraCentersByFrame = scene.centers,
        tracks = scene.tracks,
        pointsByTrackId = scene.points,
        maxNumImages = maxImages,
        minTriangulationAngleDegrees = minimumAngleDegrees
    )

    private fun scene(vararg specs: NeighborSpec): Scene {
        val centers = linkedMapOf(0 to doubleArrayOf(0.0, 0.0, 0.0))
        val tracks = mutableListOf<CameraFeatureTrack>()
        val points = linkedMapOf<Int, DoubleArray>()
        var trackId = 0
        for (spec in specs) {
            centers[spec.frameId] = doubleArrayOf(spec.centerX, 0.0, 0.0)
            for (angleDegrees in spec.anglesDegrees) {
                val halfAngle = Math.toRadians(angleDegrees) / 2.0
                val depth = spec.centerX / (2.0 * tan(halfAngle))
                tracks += CameraFeatureTrack(
                    trackId,
                    listOf(observation(0), observation(spec.frameId))
                )
                points[trackId] = doubleArrayOf(spec.centerX / 2.0, 0.0, depth)
                trackId++
            }
        }
        return Scene(centers, tracks, points)
    }

    private fun observation(frameId: Int) = CameraFeatureObservation(frameId, 0.0, 0.0)

    private data class NeighborSpec(
        val frameId: Int,
        val centerX: Double,
        val anglesDegrees: List<Double>
    )

    private data class Scene(
        val centers: Map<Int, DoubleArray>,
        val tracks: List<CameraFeatureTrack>,
        val points: Map<Int, DoubleArray>
    )
}
