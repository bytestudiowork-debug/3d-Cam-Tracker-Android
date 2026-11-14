package com.trackstudio.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class FeatureMapDecoderTest {
    @Test
    fun decodesCellLogitsAndSamplesNormalizedDescriptors() {
        val width = 16
        val height = 16
        val cellCount = 4
        val descriptors = FloatArray(64 * cellCount)
        for (channel in 0 until 64) {
            for (cell in 0 until cellCount) descriptors[channel * cellCount + cell] =
                if (channel == 0) 1f else 0f
        }
        val logits = FloatArray(65 * cellCount) { -12f }
        logits[3 * cellCount + 0] = 12f
        val reliability = FloatArray(cellCount) { 1f }

        val observations = FeatureMapDecoder.decode(
            frameIndex = 7,
            inputWidth = width,
            inputHeight = height,
            descriptorMap = descriptors,
            keypointLogits = logits,
            reliabilityMap = reliability,
            maximumFeatures = 20
        )

        assertTrue(observations.isNotEmpty())
        val feature = observations.first()
        assertEquals(7, feature.frameIndex)
        assertEquals(3f, feature.x, 0f)
        assertEquals(0f, feature.y, 0f)
        val descriptor = requireNotNull(feature.descriptor)
        assertEquals(64, descriptor.size)
        assertEquals(1f, descriptor[0], 1e-5f)
        assertEquals(0f, descriptor.drop(1).sum(), 1e-5f)
    }

    @Test
    fun appliesTopKAndRejectsLowConfidenceCells() {
        val descriptors = FloatArray(64) { if (it == 0) 1f else 0f }
        val logits = FloatArray(65) { -10f }
        logits[0] = 10f
        val reliability = floatArrayOf(0f)

        val result = FeatureMapDecoder.decode(
            frameIndex = 0,
            inputWidth = 8,
            inputHeight = 8,
            descriptorMap = descriptors,
            keypointLogits = logits,
            reliabilityMap = reliability,
            maximumFeatures = 1
        )

        assertEquals(0, result.size)
    }
    @Test
    fun decodesVariableWidthDescriptorsWithoutChangingFeatureCoordinates() {
        val width = 16
        val height = 16
        val cellCount = 4
        val descriptorChannels = 128
        val descriptors = FloatArray(descriptorChannels * cellCount)
        for (channel in 0 until descriptorChannels) {
            for (cell in 0 until cellCount) {
                descriptors[channel * cellCount + cell] =
                    if (channel == 7) 1f else 0f
            }
        }
        val logits = FloatArray(65 * cellCount) { -12f }
        logits[3 * cellCount] = 12f
        val reliability = FloatArray(cellCount) { 1f }

        val observations = FeatureMapDecoder.decode(
            frameIndex = 2,
            inputWidth = width,
            inputHeight = height,
            descriptorMap = descriptors,
            keypointLogits = logits,
            reliabilityMap = reliability,
            maximumFeatures = 10,
            descriptorChannels = descriptorChannels
        )

        assertTrue(observations.isNotEmpty())
        assertEquals(3f, observations.first().x, 0f)
        assertEquals(0f, observations.first().y, 0f)
        val descriptor = requireNotNull(observations.first().descriptor)
        assertEquals(descriptorChannels, descriptor.size)
        assertEquals(1f, descriptor[7], 1e-5f)
        assertEquals(0f, descriptor.filterIndexed { index, _ -> index != 7 }.sum(), 1e-5f)
    }

}
