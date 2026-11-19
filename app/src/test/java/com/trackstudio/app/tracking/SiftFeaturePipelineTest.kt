package com.trackstudio.app.tracking

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin
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
            row(0, 0, 0.10, 1, 0.20),
            row(1, 1, 0.08, 0, 0.30),
            row(2, 2, 0.30, 3, 0.35),
            row(3, 3, 0.72, 2, 0.90)
        )
        val reverse = listOf(
            row(0, 0, 0.11, 1, 0.25),
            row(1, 1, 0.085, 0, 0.28),
            row(2, 2, 0.10, 3, 0.25),
            row(3, 3, 0.10, 2, 0.25)
        )

        assertEquals(
            listOf(0 to 0, 1 to 1),
            SiftMatchPolicy.selectMutualMatches(forward, reverse)
        )
    }

    @Test
    fun acceptsReciprocalSingletonRowsWithoutASecondNeighbor() {
        val onlyCandidate = listOf(RankedDescriptorMatch(0, 0, 0.10))

        assertEquals(
            listOf(0 to 0),
            SiftMatchPolicy.selectMutualMatches(listOf(onlyCandidate), listOf(onlyCandidate))
        )
    }

    @Test
    fun appliesMaximumNormalizedL2DistanceAtThe07Boundary() {
        val maximumDistance = SiftMatchPolicy.DEFAULT_MAX_DISTANCE
        val atBoundary = row(0, 0, maximumDistance, 1, 1.0)
        assertEquals(
            listOf(0 to 0),
            SiftMatchPolicy.selectMutualMatches(listOf(atBoundary), listOf(atBoundary))
        )

        val aboveBoundary = row(0, 0, maximumDistance + 1e-4, 1, 1.0)
        assertEquals(
            emptyList<Pair<Int, Int>>(),
            SiftMatchPolicy.selectMutualMatches(listOf(aboveBoundary), listOf(aboveBoundary))
        )
    }

    @Test
    fun rejectsNonReciprocalMatchesAndRatioBoundary() {
        val forward = listOf(
            row(0, 0, 0.40, 1, 0.50),
            row(1, 0, 0.07, 1, 0.20)
        )
        val reverse = listOf(
            row(0, 0, 0.07, 1, 0.20),
            row(1, 1, 0.40, 0, 0.50)
        )

        assertEquals(emptyList<Pair<Int, Int>>(), SiftMatchPolicy.selectMutualMatches(forward, reverse))
    }

    @Test
    fun capsMatchesInQueryIndexOrder() {
        val forward = listOf(
            row(2, 2, 0.05, 0, 0.20),
            row(0, 0, 0.05, 2, 0.20),
            row(1, 1, 0.05, 0, 0.20)
        )
        val reverse = listOf(
            row(0, 0, 0.05, 1, 0.20),
            row(1, 1, 0.05, 2, 0.20),
            row(2, 2, 0.05, 0, 0.20)
        )

        assertEquals(
            listOf(0 to 0, 1 to 1),
            SiftMatchPolicy.selectMutualMatches(forward, reverse, maximumMatches = 2)
        )
    }

    @Test
    fun duplicateQueryRowsWithEqualDistancesChooseTheLowerTrainIndex() {
        val forward = listOf(
            row(0, 1, 0.01, 2, 0.03),
            row(0, 0, 0.01, 2, 0.03)
        )
        val reverse = listOf(
            row(0, 0, 0.01, 1, 0.03),
            row(1, 0, 0.01, 1, 0.03)
        )

        assertEquals(
            listOf(0 to 0),
            SiftMatchPolicy.selectMutualMatches(forward, reverse)
        )
    }

    @Test
    fun ratioTestUsesSameNormalizedL2MetricAsNearestNeighborSearch() {
        val query = ByteArray(RootSiftDescriptorMath.DESCRIPTOR_LENGTH) { index ->
            if (index < 64) 64 else 0
        }
        fun candidate(angle: Double): ByteArray = ByteArray(query.size) { index ->
            val base = if (index < 64) 64.0 else 0.0
            val signedOrthogonal = when (index) {
                in 0 until 32 -> 64.0
                in 32 until 64 -> -64.0
                else -> 0.0
            }
            (base * cos(angle) + signedOrthogonal * sin(angle)).roundToInt().toByte()
        }

        val closer = candidate(0.54)
        val second = candidate(0.68)
        val closerAngle = RootSiftDescriptorMath.angularDistance(query, 0, closer, 0)
        val secondAngle = RootSiftDescriptorMath.angularDistance(query, 0, second, 0)
        fun l2(candidate: ByteArray) = sqrt(query.indices.sumOf { index ->
            val difference = (query[index].toInt() and 0xff) -
                (candidate[index].toInt() and 0xff)
            difference.toDouble() * difference
        })
        val closerL2 = l2(closer) / RootSiftDescriptorMath.DESCRIPTOR_NORM
        val secondL2 = l2(second) / RootSiftDescriptorMath.DESCRIPTOR_NORM

        assertTrue(closerAngle < SiftMatchPolicy.DEFAULT_MAX_DISTANCE)
        assertTrue(closerAngle < SiftMatchPolicy.DEFAULT_RATIO * secondAngle)
        assertTrue(closerL2 < secondL2)
        assertTrue(closerL2 >= SiftMatchPolicy.DEFAULT_RATIO * secondL2)
        assertEquals(
            emptyList<Pair<Int, Int>>(),
            SiftMatchPolicy.selectMutualMatches(
                forwardRows = listOf(
                    listOf(
                        RankedDescriptorMatch(0, 0, closerL2),
                        RankedDescriptorMatch(0, 1, secondL2)
                    )
                ),
                reverseRows = listOf(
                    listOf(RankedDescriptorMatch(0, 0, closerL2)),
                    listOf(RankedDescriptorMatch(1, 0, secondL2))
                )
            )
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
