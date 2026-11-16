package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class TrackingFrameDimensionsTest {
    @Test
    fun firstCandidateDefinesReferenceDimensions() {
        assertEquals(1920 to 1080, TrackingFrameDimensions.validate(null, 1920, 1080))
    }

    @Test
    fun laterFramesMayMatchReferenceDimensions() {
        var reference = TrackingFrameDimensions.validate(null, 1280, 720)
        reference = TrackingFrameDimensions.validate(reference, 1280, 720)
        assertEquals(1280 to 720, reference)
    }

    @Test
    fun laterFrameWithDifferentDimensionsIsRejected() {
        val reference = TrackingFrameDimensions.validate(null, 1280, 720)
        val failure = assertThrows(IllegalArgumentException::class.java) {
            TrackingFrameDimensions.validate(reference, 640, 360)
        }
        assertEquals(TrackingFrameDimensions.MISMATCH_MESSAGE, failure.message)
    }

    @Test
    fun emptyDecodedFramesAreRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            TrackingFrameDimensions.validate(null, 0, 720)
        }
    }
}
