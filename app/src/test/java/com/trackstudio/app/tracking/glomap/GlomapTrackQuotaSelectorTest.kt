package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GlomapTrackQuotaSelectorTest {
    @Test
    fun negativeDefaultKeepsTheAndroidQuotaDisabledAndFiltersRegisteredViews() {
        val tracks = listOf(
            track(id = 30, frames = listOf(0, 1, 2, 99)),
            track(id = 20, frames = listOf(0, 1, 99)),
            track(id = 10, frames = (0..100).toList())
        )

        val result = GlomapTrackQuotaSelector.select(
            priorityOrderedTracks = tracks,
            registeredFrameIds = setOf(0, 1, 2),
            minTracksPerView = -1
        )

        assertEquals(listOf(30), result.selectedTracks.map { it.id })
        assertEquals(listOf(0, 1, 2), result.selectedTracks.single().observations.map { it.frameIndex })
        assertEquals(2, result.lengthEligibleTrackCount)
        assertFalse(result.stoppedWhenViewsReachedQuota)
    }

    @Test
    fun nonnegativeQuotaUsesPinnedPostIncrementBoundary() {
        val result = GlomapTrackQuotaSelector.select(
            priorityOrderedTracks = listOf(
                track(id = 3, frames = listOf(0, 1, 2)),
                track(id = 2, frames = listOf(0, 1, 2)),
                track(id = 1, frames = listOf(0, 1, 2))
            ),
            registeredFrameIds = setOf(0, 1, 2),
            minTracksPerView = 1
        )

        assertEquals(listOf(3, 2), result.selectedTracks.map { it.id })
        assertEquals(mapOf(0 to 2, 1 to 2, 2 to 2), result.quotaCounterByFrame)
        assertTrue(result.stoppedWhenViewsReachedQuota)
    }

    @Test
    fun quotaSelectsWholeTrackEvenWhenSomeOfItsViewsAlreadyReachedTheThreshold() {
        val result = GlomapTrackQuotaSelector.select(
            priorityOrderedTracks = listOf(
                track(id = 10, frames = listOf(0, 1, 2)),
                track(id = 9, frames = listOf(0, 1, 3))
            ),
            registeredFrameIds = setOf(0, 1, 2, 3),
            minTracksPerView = 0
        )

        assertEquals(listOf(10, 9), result.selectedTracks.map { it.id })
        assertEquals(listOf(0, 1, 3), result.selectedTracks.last().observations.map { it.frameIndex })
        assertEquals(mapOf(0 to 1, 1 to 1, 2 to 1, 3 to 1), result.quotaCounterByFrame)
        assertTrue(result.stoppedWhenViewsReachedQuota)
    }

    @Test
    fun globalTrackCapRetainsPinnedPostInsertionGreaterThanCheck() {
        val result = GlomapTrackQuotaSelector.select(
            priorityOrderedTracks = listOf(
                track(id = 3, frames = listOf(0, 1, 2)),
                track(id = 2, frames = listOf(0, 1, 2)),
                track(id = 1, frames = listOf(0, 1, 2))
            ),
            registeredFrameIds = setOf(0, 1, 2),
            minTracksPerView = -1,
            maxNumTracks = 1
        )

        assertEquals(listOf(3, 2), result.selectedTracks.map { it.id })
        assertTrue(result.stoppedAfterExceedingMaxTracks)
    }

    private fun track(id: Int, frames: List<Int>): CameraFeatureTrack = CameraFeatureTrack(
        id = id,
        observations = frames.mapIndexed { index, frame ->
            CameraFeatureObservation(
                frameIndex = frame,
                x = index.toDouble(),
                y = 0.0
            )
        }
    )
}
