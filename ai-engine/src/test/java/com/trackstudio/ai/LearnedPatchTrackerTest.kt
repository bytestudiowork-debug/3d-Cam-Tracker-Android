package com.trackstudio.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class LearnedPatchTrackerTest {
    @Test
    fun carriesMatchedTrackIdentityIntoNextFrame() {
        val old = FeatureObservation(
            frameIndex = 0,
            x = 10f,
            y = 10f,
            confidence = 0.9f,
            trackId = 123L,
            descriptor = floatArrayOf(1f, 0f)
        )
        val next = old.copy(frameIndex = 1, x = 12f)
        val detector = object : FeatureDetector {
            override fun detect(frame: TrackingFrame) = listOf(next)
        }
        val tracker = LearnedPatchTracker(detector)

        val result = tracker.track(
            previous = frame(0),
            current = frame(1),
            previousObservations = listOf(old)
        )

        assertEquals(1, result.size)
        assertEquals(123L, result.single().trackId)
        assertEquals(1, result.single().frameIndex)
    }

    private fun frame(index: Int) = TrackingFrame(
        frameIndex = index,
        timestampNs = index.toLong(),
        width = 1,
        height = 1,
        pixels = byteArrayOf(0),
        channels = 1
    )
}
