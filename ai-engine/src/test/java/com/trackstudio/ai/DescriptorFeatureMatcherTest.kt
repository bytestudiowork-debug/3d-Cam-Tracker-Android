package com.trackstudio.ai

import org.junit.Assert.assertEquals
import org.junit.Test

class DescriptorFeatureMatcherTest {
    @Test
    fun matchesMutualNearestDescriptors() {
        val first = listOf(
            feature(0, floatArrayOf(1f, 0f, 0f)),
            feature(1, floatArrayOf(0f, 1f, 0f))
        )
        val second = listOf(
            feature(0, floatArrayOf(0f, 1f, 0f)),
            feature(1, floatArrayOf(1f, 0f, 0f))
        )

        val matches = DescriptorFeatureMatcher().match(first, second)

        assertEquals(2, matches.size)
        assertEquals(0, matches[0].firstFeatureIndex)
        assertEquals(1, matches[0].secondFeatureIndex)
        assertEquals(1, matches[1].firstFeatureIndex)
        assertEquals(0, matches[1].secondFeatureIndex)
    }

    @Test
    fun ignoresMissingOrInvalidDescriptors() {
        val first = listOf(
            FeatureObservation(0, 1f, 1f, 0.9f),
            feature(1, floatArrayOf(0f, 0f, 0f))
        )
        val second = listOf(feature(0, floatArrayOf(1f, 0f, 0f)))

        assertEquals(0, DescriptorFeatureMatcher().match(first, second).size)
    }

    @Test
    fun rejectsAmbiguousNearestDescriptors() {
        val first = listOf(feature(0, floatArrayOf(1f, 0f)))
        val second = listOf(
            feature(0, floatArrayOf(1f, 0f)),
            feature(1, floatArrayOf(1f, 0f))
        )

        assertEquals(0, DescriptorFeatureMatcher().match(first, second).size)
    }

    private fun feature(index: Int, descriptor: FloatArray) =
        FeatureObservation(
            frameIndex = 0,
            x = index.toFloat(),
            y = 0f,
            confidence = 0.9f,
            descriptor = descriptor
        )
}
