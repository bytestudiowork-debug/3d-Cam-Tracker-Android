package com.trackstudio.app.tracking

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SolverObjectiveTest {
    @Test
    fun acceptsImprovedUnchangedAndTinyRoundoffCosts() {
        assertTrue(SolverObjective.didNotIncrease(10.0, 4.0))
        assertTrue(SolverObjective.didNotIncrease(10.0, 10.0))
        assertTrue(SolverObjective.didNotIncrease(10.0, 10.0 + 1e-9))
    }

    @Test
    fun rejectsMaterialCostRegressionAndNonFiniteCosts() {
        assertFalse(SolverObjective.didNotIncrease(10.0, 10.1))
        assertFalse(SolverObjective.didNotIncrease(Double.NaN, 1.0))
        assertFalse(SolverObjective.didNotIncrease(1.0, Double.POSITIVE_INFINITY))
    }
}
