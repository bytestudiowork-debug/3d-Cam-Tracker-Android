package com.trackstudio.app.tracking

import kotlin.math.abs
import kotlin.math.max

internal object SolverObjective {
    fun didNotIncrease(initialCost: Double, finalCost: Double): Boolean {
        if (!initialCost.isFinite() || !finalCost.isFinite()) return false
        val tolerance = 1e-9 * max(1.0, abs(initialCost))
        return finalCost <= initialCost + tolerance
    }
}
