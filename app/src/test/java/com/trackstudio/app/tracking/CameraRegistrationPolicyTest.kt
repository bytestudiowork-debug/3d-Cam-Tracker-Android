package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Test

class CameraRegistrationPolicyTest {
    @Test
    fun poseHypothesisAndLocalRefitThresholdsStayPinned() {
        assertEquals(30, CameraRegistrationPolicy.MINIMUM_CORRESPONDENCES)
        assertEquals(0.25, CameraRegistrationPolicy.MINIMUM_INLIER_RATIO, 0.0)
        assertEquals(12.0f, CameraRegistrationPolicy.RANSAC_REPROJECTION_ERROR_PX)
        assertEquals(10000, CameraRegistrationPolicy.RANSAC_MAX_ITERATIONS)
        assertEquals(0.99999, CameraRegistrationPolicy.RANSAC_CONFIDENCE, 0.0)
        assertEquals(2, CameraRegistrationPolicy.LOCAL_OPTIMIZATION_ROUNDS)
    }

    @Test
    fun androidQualityGatesStayStricterThanTheRobustHypothesisThreshold() {
        assertEquals(4.0f, CameraRegistrationPolicy.FINAL_REPROJECTION_ERROR_PX)
        assertEquals(0.90, CameraRegistrationPolicy.MINIMUM_CHEIRALITY_RATIO, 0.0)
        assertEquals(4, CameraRegistrationPolicy.MINIMUM_OCCUPIED_GRID_CELLS)
    }
}
