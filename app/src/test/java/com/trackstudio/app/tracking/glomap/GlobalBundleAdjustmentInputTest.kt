package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import org.junit.Assert.assertEquals
import org.junit.Test

class GlobalBundleAdjustmentInputTest {
    @Test
    fun excludesCamerasWithoutEligiblePointResiduals() {
        val frameIds = (0..4).toList()
        val centers = frameIds.associateWith { doubleArrayOf(it.toDouble(), 0.0, 0.0) }
        val rotations = frameIds.associateWith { identity() }
        val tracks = listOf(
            track(7, 0, 1, 2),
            track(8, 0, 1, 2),
            track(9, 2, 3),
            track(10, 1, 3, 4)
        )

        val selection = GlobalBundleAdjustmentInput.select(
            frameCentersById = centers,
            rotationsByFrame = rotations,
            pointsByTrackId = mapOf(
                7 to doubleArrayOf(0.0, 0.0, 5.0),
                8 to doubleArrayOf(0.0, 1.0, 5.0),
                9 to doubleArrayOf(0.0, 2.0, 5.0)
            ),
            tracks = tracks
        )

        assertEquals(listOf(0, 1, 2), selection.frameIds)
        assertEquals(listOf(7, 8), selection.tracks.map { it.id })
        assertEquals(6, selection.tracks.sumOf { it.observations.size })
    }

    @Test
    fun countsDistinctViewsRatherThanObservationRows() {
        val centers = (0..2).associateWith { doubleArrayOf(it.toDouble(), 0.0, 0.0) }
        val rotations = centers.keys.associateWith { identity() }
        val repeatedViewTrack = CameraFeatureTrack(
            12,
            listOf(
                CameraFeatureObservation(0, 10.0, 10.0),
                CameraFeatureObservation(0, 11.0, 10.0),
                CameraFeatureObservation(1, 12.0, 10.0)
            )
        )

        val selection = GlobalBundleAdjustmentInput.select(
            frameCentersById = centers,
            rotationsByFrame = rotations,
            pointsByTrackId = mapOf(12 to doubleArrayOf(0.0, 0.0, 5.0)),
            tracks = listOf(repeatedViewTrack)
        )

        assertEquals(emptyList<Int>(), selection.frameIds)
        assertEquals(emptyList<Int>(), selection.tracks.map { it.id })
    }

    private fun track(id: Int, vararg frameIndices: Int) = CameraFeatureTrack(
        id,
        frameIndices.map { frameIndex ->
            CameraFeatureObservation(frameIndex, 50.0 + frameIndex, 50.0)
        }
    )

    private fun identity() = doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
}
