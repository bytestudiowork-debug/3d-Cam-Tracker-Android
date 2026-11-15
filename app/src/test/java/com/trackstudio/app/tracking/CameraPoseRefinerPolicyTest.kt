package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Test

class CameraPoseRefinerPolicyTest {
    @Test
    fun robustPoseRefinementUsesPinnedHuberAndIterationSettings() {
        assertEquals(100, CameraPoseRefiner.MAX_ITERATIONS)
        assertEquals(1.0, CameraPoseRefiner.HUBER_SCALE_PX, 0.0)
        assertEquals(1.0, CameraPoseRefiner.GRADIENT_TOLERANCE, 0.0)
    }
}
