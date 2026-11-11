package com.trackstudio.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DenseFeatureDecoderTest {
    @Test
    fun decodesPointAndNormalizesDescriptor() {
        val scores = FloatArray(65) { -10f }.also {
            it[0] = 10f
            it[64] = -10f
        }
        val descriptors = FloatArray(256).also {
            it[0] = 3f
            it[1] = 4f
        }

        val result = DenseFeatureDecoder.decode(
            frameIndex = 7,
            imageWidth = 8,
            imageHeight = 8,
            cellHeight = 1,
            cellWidth = 1,
            scores = scores,
            descriptors = descriptors,
            confidenceThreshold = 0.1f
        )

        assertEquals(1, result.size)
        assertEquals(7, result.single().frameIndex)
        assertEquals(0f, result.single().x, 0f)
        assertEquals(0f, result.single().y, 0f)
        assertEquals(0.6f, result.single().descriptor!![0], 0.0001f)
        assertEquals(0.8f, result.single().descriptor!![1], 0.0001f)
    }

    @Test
    fun suppressesNearbyLowerScoringPoints() {
        val scores = FloatArray(65 * 2) { -10f }
        scores[0] = 10f
        scores[1] = 9f
        scores[64] = -10f
        scores[65 + 64] = -10f
        val descriptors = FloatArray(2 * 256).also {
            it[0] = 1f
            it[256] = 1f
        }

        val result = DenseFeatureDecoder.decode(
            frameIndex = 0,
            imageWidth = 16,
            imageHeight = 8,
            cellHeight = 1,
            cellWidth = 2,
            scores = scores,
            descriptors = descriptors,
            confidenceThreshold = 0.1f,
            nmsRadius = 4
        )

        assertEquals(1, result.size)
        assertTrue(result.single().confidence > 0.9f)
    }
}
