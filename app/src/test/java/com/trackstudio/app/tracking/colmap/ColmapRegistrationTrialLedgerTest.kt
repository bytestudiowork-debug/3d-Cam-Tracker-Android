package com.trackstudio.app.tracking.colmap

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ColmapRegistrationTrialLedgerTest {
    @Test
    fun countsSeedFallbacksTowardPerFrameMaximumAndKeepsFramesIndependent() {
        val trials = ColmapRegistrationTrialLedger(frameCount = 3, maximumTrials = 3)

        assertEquals(1, trials.beginTrial(frameIndex = 1))
        assertEquals(2, trials.beginTrial(frameIndex = 1))
        assertEquals(3, trials.beginTrial(frameIndex = 1))
        assertNull(trials.beginTrial(frameIndex = 1))

        assertEquals(3, trials[1])
        assertEquals(0, trials[0])
        assertEquals(1, trials.beginTrial(frameIndex = 2))
    }
}
