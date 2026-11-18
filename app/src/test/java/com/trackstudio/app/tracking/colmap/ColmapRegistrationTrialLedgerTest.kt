package com.trackstudio.app.tracking.colmap

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ColmapRegistrationTrialLedgerTest {
    @Test
    fun samePassFallbacksShareTrialAndMapUpdateRetriesAdvanceBudget() {
        val trials = ColmapRegistrationTrialLedger(frameCount = 3, maximumTrials = 3)

        assertEquals(1, trials.beginTrial(frameIndex = 1))
        assertEquals(1, trials.currentTrialForFallback(frameIndex = 1))
        assertEquals(1, trials.currentTrialForFallback(frameIndex = 1))
        assertEquals(1, trials[1])

        // A failed retry after a map update is a new registration trial.
        assertEquals(2, trials.beginTrial(frameIndex = 1))
        assertEquals(3, trials.beginTrial(frameIndex = 1))
        assertNull(trials.beginTrial(frameIndex = 1))

        assertEquals(3, trials[1])
        assertEquals(0, trials[0])
        assertEquals(1, trials.beginTrial(frameIndex = 2))
    }
}
