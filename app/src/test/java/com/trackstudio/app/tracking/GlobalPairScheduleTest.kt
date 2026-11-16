package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GlobalPairScheduleTest {
    @Test
    fun sixFrameScheduleMatchesReferenceQuadraticPairSet() {
        val actual = GlobalReconstructionEngine().buildPairSchedule(
            frameCount = 6,
            maximumPairs = 12
        )
        val expectedReferencePairs = setOf(
            0 to 1,
            0 to 2,
            0 to 4,
            1 to 2,
            1 to 3,
            1 to 5,
            2 to 3,
            2 to 4,
            3 to 4,
            3 to 5,
            4 to 5
        )

        assertEquals(expectedReferencePairs, actual.toSet())
        assertEquals(actual.size, actual.distinct().size)
    }

    @Test
    fun longSequenceScheduleSpreadsLongBaselinePairsAcrossTimeline() {
        val frameCount = 588
        val actual = GlobalReconstructionEngine().buildPairSchedule(
            frameCount = frameCount,
            maximumPairs = frameCount * 2
        )

        assertEquals(frameCount * 2, actual.size)
        assertEquals(actual.size, actual.distinct().size)
        assertEquals(
            (0 until frameCount - 1).map { it to it + 1 }.toSet(),
            actual.filter { it.second == it.first + 1 }.toSet()
        )
        assertTrue(actual.any { it.second - it.first >= 128 })
        assertTrue(actual.any {
            it.first >= frameCount * 3 / 4 && it.second - it.first >= 16
        })
    }

    @Test
    fun retainsConnectedAdjacentBackboneWhenBudgetIsSmallerThanFrameCount() {
        val actual = GlobalReconstructionEngine().buildPairSchedule(
            frameCount = 6,
            maximumPairs = 4
        )

        assertEquals(
            setOf(0 to 1, 1 to 2, 2 to 3, 3 to 4, 4 to 5),
            actual.toSet()
        )
    }

    @Test
    fun returnsNoPairsForFewerThanTwoFrames() {
        assertEquals(emptyList<Pair<Int, Int>>(), GlobalReconstructionEngine().buildPairSchedule(1, 1))
    }
}
