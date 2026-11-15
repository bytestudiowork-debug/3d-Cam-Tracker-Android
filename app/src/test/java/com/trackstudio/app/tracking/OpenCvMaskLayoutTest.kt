package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class OpenCvMaskLayoutTest {
    @Test
    fun linearIndicesMapEquallyAcrossColumnAndRowMasks() {
        val columnCoordinates = (0 until 5).map {
            OpenCvMaskLayout.coordinates(rows = 5, columns = 1, index = it, expectedSize = 5)
        }
        val rowCoordinates = (0 until 5).map {
            OpenCvMaskLayout.coordinates(rows = 1, columns = 5, index = it, expectedSize = 5)
        }

        assertEquals((0 until 5).map { it to 0 }, columnCoordinates)
        assertEquals((0 until 5).map { 0 to it }, rowCoordinates)
    }

    @Test
    fun malformedShapesAndOutOfRangeIndicesAreRejected() {
        assertNull(OpenCvMaskLayout.coordinates(2, 2, 0, 3))
        assertNull(OpenCvMaskLayout.coordinates(3, 1, 3, 3))
        assertNull(OpenCvMaskLayout.coordinates(0, 1, 0, 0))
    }
}
