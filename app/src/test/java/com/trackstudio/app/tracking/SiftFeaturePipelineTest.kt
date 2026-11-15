package com.trackstudio.app.tracking

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.sqrt

class SiftFeaturePipelineTest {
    @Test
    fun l1RootNormalizationProducesReferenceDescriptorScale() {
        val normalized = RootSiftDescriptorMath.normalizeDescriptor(floatArrayOf(3f, 1f, 0f))

        assertArrayEquals(
            floatArrayOf((sqrt(0.75) * 512.0).toFloat(), 256f, 0f),
            normalized,
            1e-4f
        )
        assertEquals(512.0 * 512.0, normalized.sumOf { it.toDouble() * it }, 1e-2)
    }

    @Test
    fun l1RootNormalizationIsInvariantToPositiveDescriptorScale() {
        val first = RootSiftDescriptorMath.normalizeDescriptor(floatArrayOf(3f, 1f, 2f))
        val scaled = RootSiftDescriptorMath.normalizeDescriptor(floatArrayOf(30f, 10f, 20f))

        assertArrayEquals(first, scaled, 1e-4f)
    }

    @Test
    fun appliesRatioDistanceAndReciprocalMatchRules() {
        val forward = listOf(
            row(0, 0, 100.0, 1, 200.0),
            row(1, 1, 80.0, 0, 300.0),
            row(2, 2, 300.0, 3, 350.0),
            row(3, 3, 360.0, 2, 600.0)
        )
        val reverse = listOf(
            row(0, 0, 110.0, 1, 250.0),
            row(1, 1, 85.0, 0, 280.0),
            row(2, 2, 100.0, 3, 250.0),
            row(3, 3, 100.0, 2, 250.0)
        )

        assertEquals(
            listOf(0 to 0, 1 to 1),
            SiftMatchPolicy.selectMutualMatches(forward, reverse)
        )
    }

    @Test
    fun acceptsReciprocalSingletonRowsWithoutASecondNeighbor() {
        val onlyCandidate = listOf(RankedDescriptorMatch(0, 0, 100.0))

        assertEquals(
            listOf(0 to 0),
            SiftMatchPolicy.selectMutualMatches(listOf(onlyCandidate), listOf(onlyCandidate))
        )
    }

    @Test
    fun appliesMaximumDescriptorDistanceAtThe512ScaleBoundary() {
        val maximumDistance = SiftMatchPolicy.DEFAULT_MAX_DISTANCE *
            RootSiftDescriptorMath.DESCRIPTOR_NORM
        val atBoundary = row(0, 0, maximumDistance, 1, 500.0)
        assertEquals(
            listOf(0 to 0),
            SiftMatchPolicy.selectMutualMatches(listOf(atBoundary), listOf(atBoundary))
        )

        val aboveBoundary = row(0, 0, maximumDistance + 1e-4, 1, 500.0)
        assertEquals(
            emptyList<Pair<Int, Int>>(),
            SiftMatchPolicy.selectMutualMatches(listOf(aboveBoundary), listOf(aboveBoundary))
        )
    }

    @Test
    fun rejectsNonReciprocalMatchesAndRatioBoundary() {
        val forward = listOf(
            row(0, 0, 80.0, 1, 100.0),
            row(1, 0, 70.0, 1, 200.0)
        )
        val reverse = listOf(
            row(0, 0, 70.0, 1, 200.0),
            row(1, 1, 80.0, 0, 100.0)
        )

        assertEquals(emptyList<Pair<Int, Int>>(), SiftMatchPolicy.selectMutualMatches(forward, reverse))
    }

    @Test
    fun capsMatchesInQueryIndexOrder() {
        val forward = listOf(
            row(2, 2, 50.0, 0, 200.0),
            row(0, 0, 50.0, 2, 200.0),
            row(1, 1, 50.0, 0, 200.0)
        )
        val reverse = listOf(
            row(0, 0, 50.0, 1, 200.0),
            row(1, 1, 50.0, 2, 200.0),
            row(2, 2, 50.0, 0, 200.0)
        )

        assertEquals(
            listOf(0 to 0, 1 to 1),
            SiftMatchPolicy.selectMutualMatches(forward, reverse, maximumMatches = 2)
        )
    }

    @Test
    fun duplicateQueryRowsWithEqualDistancesChooseTheLowerTrainIndex() {
        val forward = listOf(
            row(0, 1, 10.0, 2, 30.0),
            row(0, 0, 10.0, 2, 30.0)
        )
        val reverse = listOf(
            row(0, 0, 10.0, 1, 30.0),
            row(1, 0, 10.0, 1, 30.0)
        )

        assertEquals(
            listOf(0 to 0),
            SiftMatchPolicy.selectMutualMatches(forward, reverse)
        )
    }

    private fun row(
        query: Int,
        bestTrain: Int,
        bestDistance: Double,
        secondTrain: Int,
        secondDistance: Double
    ) = listOf(
        RankedDescriptorMatch(query, bestTrain, bestDistance),
        RankedDescriptorMatch(query, secondTrain, secondDistance)
    )
}
