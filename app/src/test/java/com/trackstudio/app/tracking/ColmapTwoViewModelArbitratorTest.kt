package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Test

private typealias ModelConfig = ColmapTwoViewModelArbitrator.Configuration
private typealias InlierModel = ColmapTwoViewModelArbitrator.InlierModel

class ColmapTwoViewModelArbitratorTest {

    @Test
    fun calibratedModelUsesStrictEssentialToFundamentalRatioAndLargestEpipolarSupport() {
        val belowBoundary = arbitrate(e = 95, f = 100, h = 20)
        val aboveBoundaryWithFundamentalWinner = arbitrate(e = 96, f = 100, h = 20)
        val tie = arbitrate(e = 100, f = 100, h = 20)

        assertEquals(ModelConfig.UNCALIBRATED, belowBoundary.configuration)
        assertEquals(InlierModel.FUNDAMENTAL, aboveBoundaryWithFundamentalWinner.inlierModel)
        assertEquals(100, aboveBoundaryWithFundamentalWinner.inlierCount)
        assertEquals(InlierModel.ESSENTIAL, tie.inlierModel)
    }

    @Test
    fun calibratedEpipolarSupportBecomesPlanarWhenHomographyRatioIsStrictlyAboveThreshold() {
        val exactBoundary = arbitrate(e = 100, f = 100, h = 80)
        val aboveBoundaryWithEpipolarMaskWinner = arbitrate(e = 100, f = 100, h = 81)
        val aboveBoundaryWithHomographyMaskWinner = arbitrate(e = 100, f = 100, h = 101)

        assertEquals(ModelConfig.CALIBRATED, exactBoundary.configuration)
        assertEquals(ModelConfig.PLANAR_OR_PANORAMIC, aboveBoundaryWithEpipolarMaskWinner.configuration)
        assertEquals(InlierModel.ESSENTIAL, aboveBoundaryWithEpipolarMaskWinner.inlierModel)
        assertEquals(ModelConfig.PLANAR_OR_PANORAMIC, aboveBoundaryWithHomographyMaskWinner.configuration)
        assertEquals(InlierModel.HOMOGRAPHY, aboveBoundaryWithHomographyMaskWinner.inlierModel)
        assertEquals(101, aboveBoundaryWithHomographyMaskWinner.inlierCount)
    }

    @Test
    fun fallbackUsesFundamentalThenHomographyAccordingToPinnedDecisionOrder() {
        val uncalibrated = arbitrate(e = 90, f = 100, h = 70)
        val planarWithFundamentalMask = arbitrate(e = 90, f = 100, h = 81)
        val planarWithHomographyMask = arbitrate(e = 90, f = 100, h = 101)
        val homographyOnly = ColmapTwoViewModelArbitrator.arbitrate(
            essentialSucceeded = false,
            essentialInliers = 0,
            fundamentalSucceeded = false,
            fundamentalInliers = 0,
            homographySucceeded = true,
            homographyInliers = 15
        )

        assertEquals(ModelConfig.UNCALIBRATED, uncalibrated.configuration)
        assertEquals(InlierModel.FUNDAMENTAL, planarWithFundamentalMask.inlierModel)
        assertEquals(ModelConfig.PLANAR_OR_PANORAMIC, planarWithFundamentalMask.configuration)
        assertEquals(InlierModel.HOMOGRAPHY, planarWithHomographyMask.inlierModel)
        assertEquals(ModelConfig.PLANAR_OR_PANORAMIC, homographyOnly.configuration)
        assertEquals(InlierModel.HOMOGRAPHY, homographyOnly.inlierModel)
    }

    @Test
    fun failedOrUnderSupportedModelsAreDegenerate() {
        val decision = ColmapTwoViewModelArbitrator.arbitrate(
            essentialSucceeded = true,
            essentialInliers = 14,
            fundamentalSucceeded = true,
            fundamentalInliers = 14,
            homographySucceeded = true,
            homographyInliers = 14
        )

        assertEquals(ModelConfig.DEGENERATE, decision.configuration)
        assertEquals(InlierModel.NONE, decision.inlierModel)
        assertEquals(0, decision.inlierCount)
    }

    private fun arbitrate(e: Int, f: Int, h: Int) =
        ColmapTwoViewModelArbitrator.arbitrate(
            essentialSucceeded = e > 0,
            essentialInliers = e,
            fundamentalSucceeded = f > 0,
            fundamentalInliers = f,
            homographySucceeded = h > 0,
            homographyInliers = h
        )
}
