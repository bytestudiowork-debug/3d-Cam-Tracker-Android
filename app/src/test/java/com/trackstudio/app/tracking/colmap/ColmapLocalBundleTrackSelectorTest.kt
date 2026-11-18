package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ColmapLocalBundleTrackSelectorTest {
    @Test
    fun retainsValidTwoObservationTracksWithinTheLocalWindow() {
        val localFrameIds = setOf(1, 2, 3)
        val twoLocalObservations = track(21, 1, 2, 9)
        val oneLocalObservation = track(22, 1, 9)
        val threeLocalObservations = track(23, 1, 2, 3, 9)

        val selected = ColmapLocalBundleTrackSelector.select(
            tracks = listOf(twoLocalObservations, oneLocalObservation, threeLocalObservations),
            localFrameIds = localFrameIds
        )

        assertEquals(listOf(21, 23), selected.map { it.id })
        assertTrue(ColmapLocalBundleTrackSelector.hasLocalSupport(twoLocalObservations, localFrameIds))
        assertFalse(ColmapLocalBundleTrackSelector.hasLocalSupport(oneLocalObservation, localFrameIds))
    }

    private fun track(id: Int, vararg frameIds: Int) = CameraFeatureTrack(
        id,
        frameIds.map { frameId -> CameraFeatureObservation(frameId, 100.0 + frameId, 200.0 - frameId) }
    )
}
