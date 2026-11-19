package com.trackstudio.app.tracking.colmap

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ColmapIterativeLocalRefinementTest {
    @Test
    fun repeatsAfterChangeAtOrAboveThresholdAndDropsRobustLossAfterFirstRound() {
        val lossModes = mutableListOf<ColmapIterativeLocalRefinement.LossMode>()
        val result = ColmapIterativeLocalRefinement.run(
            maxRefinements = 3,
            maxRefinementChange = 0.001
        ) { lossMode ->
            lossModes += lossMode
            when (lossModes.size) {
                1 -> ColmapIterativeLocalRefinement.Report(
                    adjustedObservations = 1000,
                    completedObservations = 1
                )
                else -> ColmapIterativeLocalRefinement.Report(adjustedObservations = 1000)
            }
        }

        assertEquals(
            listOf(
                ColmapIterativeLocalRefinement.LossMode.SOFT_L1,
                ColmapIterativeLocalRefinement.LossMode.TRIVIAL
            ),
            lossModes
        )
        assertEquals(listOf(0.001, 0.0), result.changedObservationRatios)
    }

    @Test
    fun zeroAdjustedObservationsConvergesImmediately() {
        var calls = 0
        val result = ColmapIterativeLocalRefinement.run { _ ->
            calls++
            ColmapIterativeLocalRefinement.Report(adjustedObservations = 0)
        }

        assertEquals(1, calls)
        assertEquals(listOf(0.0), result.changedObservationRatios)
    }

    @Test
    fun filtersModifiedPointsWhenLocalAdjustmentIsUnavailable() {
        var adjustmentCalls = 0
        var filteredObservations = 0
        val result = ColmapIterativeLocalRefinement.run(
            filterAfterUnavailableAdjustment = { filteredObservations += 3; 3 }
        ) {
            adjustmentCalls++
            null
        }

        assertEquals(1, adjustmentCalls)
        assertEquals(3, filteredObservations)
        assertTrue(result.iterations.isEmpty())
    }

    @Test
    fun zeroMaximumRefinementsDoesNotInvokeAdjustment() {
        val result = ColmapIterativeLocalRefinement.run(maxRefinements = 0) {
            error("No adjustment should run")
        }

        assertTrue(result.iterations.isEmpty())
    }
}
