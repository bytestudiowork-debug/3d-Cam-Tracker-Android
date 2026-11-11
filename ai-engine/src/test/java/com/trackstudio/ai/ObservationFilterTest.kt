package com.trackstudio.ai

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ObservationFilterTest {
    @Test
    fun removesInvalidAndOutOfBoundsObservations() {
        val result = ObservationFilter.filter(
            listOf(
                FeatureObservation(0, 10f, 10f, 0.9f),
                FeatureObservation(0, Float.NaN, 5f, 0.8f),
                FeatureObservation(0, 20f, 20f, 0.7f),
                FeatureObservation(0, 100f, 5f, 0.9f)
            ),
            frameWidth = 80,
            frameHeight = 60
        )

        assertEquals(2, result.size)
    }

    @Test
    fun keepsHighestConfidenceForNearbyDuplicates() {
        val result = ObservationFilter.filter(
            listOf(
                FeatureObservation(0, 12.00f, 14.00f, 0.4f),
                FeatureObservation(0, 12.05f, 14.02f, 0.95f)
            ),
            frameWidth = 100,
            frameHeight = 100
        )

        assertEquals(1, result.size)
        assertEquals(0.95f, result.single().confidence)
    }

    @Test
    fun respectsMaximumObservationCount() {
        val observations = (0 until 20).map {
            FeatureObservation(0, it * 2f, 10f, it / 20f)
        }

        val result = ObservationFilter.filter(
            observations,
            frameWidth = 100,
            frameHeight = 100,
            maximumCount = 5
        )

        assertEquals(5, result.size)
        assertTrue(result.zipWithNext().all { it.first.confidence >= it.second.confidence })
    }
}
