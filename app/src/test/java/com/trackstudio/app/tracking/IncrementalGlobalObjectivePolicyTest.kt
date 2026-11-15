package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Test

class IncrementalGlobalObjectivePolicyTest {
    @Test
    fun pixelSpacePoseAndLandmarkRefinersShareOnePixelHuberScale() {
        assertEquals(1.0, CameraPoseRefiner.HUBER_SCALE_PX, 0.0)
        assertEquals(1.0, MobileBundleAdjuster.HUBER_DELTA_PX, 0.0)
    }

    @Test
    fun incrementalPixelHuberMatchesCeresHuberCostAndIrlsWeight() {
        assertEquals(0.125, MobileBundleAdjuster.huberPixelCost(0.5), 1e-12)
        assertEquals(0.5, MobileBundleAdjuster.huberPixelCost(1.0), 1e-12)
        assertEquals(1.5, MobileBundleAdjuster.huberPixelCost(2.0), 1e-12)
        assertEquals(0.5, MobileBundleAdjuster.huberPixelWeight(2.0), 1e-12)
    }

    @Test
    fun incrementalLandmarkBlockKeepsItsMobileIterationAndGrossOutlierGuards() {
        assertEquals(5, MobileBundleAdjuster.MAX_ITERATIONS)
        assertEquals(30.0, MobileBundleAdjuster.MAX_OBSERVATION_ERROR_PX, 0.0)
    }
}
