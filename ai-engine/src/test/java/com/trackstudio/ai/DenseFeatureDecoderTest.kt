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
    fun decodesDensePixelScoresAndChannelFirstDescriptors() {
        val scores = FloatArray(16 * 16)
        scores[5 * 16 + 6] = 0.9f
        val descriptors = FloatArray(2 * 2 * 256)
        for (index in 0 until 4) {
            descriptors[index] = 3f
            descriptors[4 + index] = 4f
        }

        val result = DenseFeatureDecoder.decodePixelScores(
            frameIndex = 3,
            imageWidth = 16,
            imageHeight = 16,
            cellHeight = 2,
            cellWidth = 2,
            scores = scores,
            descriptors = descriptors,
            confidenceThreshold = 0.5f,
            nmsRadius = 4,
            descriptorChannelsFirst = true
        )

        assertEquals(1, result.size)
        assertEquals(6f, result.single().x, 0f)
        assertEquals(5f, result.single().y, 0f)
        assertEquals(0.6f, result.single().descriptor!![0], 0.0001f)
        assertEquals(0.8f, result.single().descriptor!![1], 0.0001f)
    }

    @Test
    fun suppressesNearbyLowerScoringPoints() {
        val scores = FloatArray(65 * 2) { -10f }
        scores[0] = 10f
        scores[1] = 5f
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
    @Test
    fun decodesCompactChannelFirstDescriptors() {
        val scores = FloatArray(8 * 8)
        scores[3 * 8 + 2] = 0.95f
        val descriptors = FloatArray(64)
        descriptors[0] = 3f
        descriptors[1] = 4f

        val result = DenseFeatureDecoder.decodePixelScores(
            frameIndex = 9,
            imageWidth = 8,
            imageHeight = 8,
            cellHeight = 1,
            cellWidth = 1,
            scores = scores,
            descriptors = descriptors,
            descriptorSize = 64,
            confidenceThreshold = 0.5f,
            nmsRadius = 1,
            descriptorChannelsFirst = true
        )

        assertEquals(1, result.size)
        assertEquals(64, result.single().descriptor!!.size)
        assertEquals(0.6f, result.single().descriptor!![0], 0.0001f)
        assertEquals(0.8f, result.single().descriptor!![1], 0.0001f)
    }

}
