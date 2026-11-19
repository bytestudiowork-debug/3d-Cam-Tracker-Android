package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class InitialPairSelectorTest {

    @Test
    fun selectsWellConditionedPairInsteadOfLargestRawMatchCount() {
        val highCountButWeak = candidate(
            secondFrame = 20,
            matched = 1500,
            inliers = 420,
            coverage = 3,
            medianParallax = 0.4,
            upperParallax = 1.2,
            cheirality = 0.9
        )
        val wellConditioned = candidate(
            secondFrame = 12,
            matched = 900,
            inliers = 360,
            coverage = 13,
            medianParallax = 7.0,
            upperParallax = 14.0,
            cheirality = 0.82
        )

        val result = InitialPairSelector.select(listOf(highCountButWeak, wellConditioned))

        assertEquals(wellConditioned, result.selected)
        assertTrue(result.evaluations.first().reason.contains("concentrated"))
    }

    @Test
    fun seedInitializationFallsBackInRankOrderWhenTheBestCandidateCannotBuildAMap() {
        val bestButUnusable = candidate(
            secondFrame = 20,
            matched = 900,
            inliers = 500,
            coverage = 14,
            medianParallax = 22.0,
            upperParallax = 28.0,
            cheirality = 0.92
        )
        val nextUsable = candidate(
            secondFrame = 12,
            matched = 700,
            inliers = 360,
            coverage = 10,
            medianParallax = 18.0,
            upperParallax = 23.0,
            cheirality = 0.78
        )
        val attempts = mutableListOf<Int>()

        val initialized = InitialPairSelector.tryRankedCandidates(
            candidates = listOf(nextUsable, bestButUnusable),
            policy = InitialPairPolicy.initialization()
        ) { candidate ->
            attempts += candidate.secondFrame
            if (candidate == bestButUnusable) null else "seed-map"
        }

        assertEquals(listOf(20, 12), attempts)
        assertEquals(nextUsable, initialized?.first)
        assertEquals("seed-map", initialized?.second)
    }

    @Test
    fun rejectsHomographyDominatedPairWithoutParallax() {
        val planar = candidate(
            matched = 800,
            inliers = 500,
            medianParallax = 1.5,
            upperParallax = 3.0,
            homographyRatio = 0.98
        )

        val result = InitialPairSelector.select(listOf(planar))

        assertNull(result.selected)
        assertTrue(result.evaluations.single().reason.contains("Homography-dominated"))
    }

    @Test
    fun rejectsPairWithPoorCheirality() {
        val candidate = candidate(cheirality = 0.2)
        val result = InitialPairSelector.select(listOf(candidate))

        assertNull(result.selected)
        assertTrue(result.evaluations.single().reason.contains("cheirality"))
    }

    @Test
    fun temporalSearchIsBoundedUniqueAndOrdered() {
        val pairs = InitialPairSelector.temporalCandidates(
            frameCount = 100,
            anchorCount = 5,
            offsets = listOf(2, 4, 4, 8),
            maximumPairs = 10
        )

        assertEquals(10, pairs.size)
        assertEquals(10, pairs.distinct().size)
        assertTrue(pairs.all { it.first < it.second && it.second < 100 })
    }

    @Test
    fun uniformSeedSamplesAndPairGraphAreBoundedDeterministicAndCoverSequenceEnds() {
        val frames = InitialPairSelector.uniformFrameSamples(frameCount = 100, maximumFrames = 24)
        val pairs = InitialPairSelector.sampledFramePairs(frames, minimumFrameGap = 2)

        assertEquals(24, frames.size)
        assertEquals(0, frames.first())
        assertEquals(99, frames.last())
        assertEquals(frames.size, frames.distinct().size)
        assertEquals(frames.size * (frames.size - 1) / 2, pairs.size)
        assertEquals(pairs, InitialPairSelector.sampledFramePairs(frames, minimumFrameGap = 2))
    }

    @Test
    fun coarseMatchDegreeOrdersHighCorrespondenceFramesBeforePairSupportTies() {
        val pairs = listOf(0 to 1, 0 to 2, 0 to 3, 4 to 5)
        val support = mapOf(
            (0 to 1) to 7,
            (0 to 2) to 7,
            (0 to 3) to 7,
            (4 to 5) to 18
        )

        val ranked = InitialPairSelector.correspondenceRankedCandidates(
            candidatePairs = pairs,
            temporalFallbackPairs = emptyList(),
            matchSupportByPair = support,
            maximumCandidates = 4,
            correspondenceRankBudget = 4
        )

        assertEquals(listOf(0 to 1, 0 to 2, 0 to 3, 4 to 5), ranked)
    }

    @Test
    fun correspondenceSearchCanPromoteSupportedPairOutsideTemporalSchedule() {
        val temporal = InitialPairSelector.temporalCandidates(
            frameCount = 100,
            anchorCount = 6,
            offsets = listOf(2, 4, 6, 9, 13, 18, 24, 32),
            maximumPairs = 36
        )
        val sampled = InitialPairSelector.sampledFramePairs(
            InitialPairSelector.uniformFrameSamples(100, 24), minimumFrameGap = 2
        )
        val outsideTemporal = sampled.first { it !in temporal }
        val support = (sampled + temporal).distinct().associateWith { 1 }.toMutableMap()
        support[outsideTemporal] = 48

        val ranked = InitialPairSelector.correspondenceRankedCandidates(
            candidatePairs = sampled + temporal,
            temporalFallbackPairs = temporal,
            matchSupportByPair = support,
            maximumCandidates = 36,
            correspondenceRankBudget = 24
        )

        assertTrue("Supported non-temporal edge was omitted: $outsideTemporal", outsideTemporal in ranked)
        assertEquals(ranked.size, ranked.distinct().size)
        assertTrue(ranked.size <= 36)
    }

    @Test
    fun temporalCandidatesRemainAsBoundedFallbackWhenCoarseGraphHasNoMatches() {
        val temporal = listOf(0 to 5, 10 to 15, 20 to 25)

        val selected = InitialPairSelector.correspondenceRankedCandidates(
            candidatePairs = temporal,
            temporalFallbackPairs = temporal,
            matchSupportByPair = emptyMap(),
            maximumCandidates = 2,
            correspondenceRankBudget = 1
        )

        assertEquals(temporal.take(2), selected)
    }

    @Test
    fun selectedCandidatePreservesRecoveredRelativePoseForViewGraph() {
        val rotation = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0
        )
        val direction = doubleArrayOf(1.0, 0.0, 0.0)
        val matches = listOf(3 to 7, 8 to 11)
        val measured = candidate().copy(
            relativeRotation = rotation,
            relativeTranslationDirection = direction,
            verifiedMatches = matches
        )

        val result = InitialPairSelector.select(listOf(measured))

        assertNotNull(result.selected)
        assertTrue(result.selected!!.relativeRotation!!.contentEquals(rotation))
        assertTrue(result.selected!!.relativeTranslationDirection!!.contentEquals(direction))
        assertEquals(matches, result.selected!!.verifiedMatches)
    }

    @Test
    fun initializationPolicyUsesReferenceInlierCountAndMedianParallaxMinimums() {
        val policy = InitialPairPolicy.initialization()
        val tooFew = candidate(matched = 300, inliers = 99, medianParallax = 20.0, upperParallax = 22.0)
        val tooLittleParallax = candidate(matched = 300, inliers = 100, medianParallax = 15.999, upperParallax = 20.0)
        val exactAngleBoundary = candidate(matched = 300, inliers = 100, medianParallax = 16.0, upperParallax = 20.0)
        val aboveAngleBoundary = candidate(matched = 300, inliers = 100, medianParallax = 16.001, upperParallax = 20.0)

        assertFalse(InitialPairSelector.evaluate(tooFew, policy).accepted)
        assertFalse(InitialPairSelector.evaluate(tooLittleParallax, policy).accepted)
        assertFalse(InitialPairSelector.evaluate(exactAngleBoundary, policy).accepted)
        assertTrue(InitialPairSelector.evaluate(aboveAngleBoundary, policy).accepted)
    }

    @Test
    fun calibratedPairRequiresEssentialSupportStrictlyAboveNinetyFivePercentOfFundamentalSupport() {
        val below = candidate().copy(essentialToFundamentalInlierRatio = 0.949)
        val exactBoundary = candidate().copy(essentialToFundamentalInlierRatio = 0.95)
        val above = candidate().copy(essentialToFundamentalInlierRatio = 0.951)

        assertFalse(InitialPairSelector.evaluate(below).accepted)
        assertFalse(InitialPairSelector.evaluate(exactBoundary).accepted)
        assertTrue(InitialPairSelector.evaluate(above).accepted)
    }

    @Test
    fun fundamentalPoseRecoveryBypassesOnlyTheCalibratedModelAgreementGate() {
        val recoveredFromFundamental = candidate().copy(
            essentialToFundamentalInlierRatio = 0.90,
            fundamentalPoseRecovered = true
        )
        val valid = InitialPairSelector.evaluate(recoveredFromFundamental)
        val insufficientParallax = InitialPairSelector.evaluate(
            recoveredFromFundamental.copy(medianParallaxDegrees = 0.99)
        )

        assertTrue(valid.accepted)
        assertFalse(insufficientParallax.accepted)
        assertTrue(insufficientParallax.reason.contains("parallax"))
        assertFalse(
            InitialPairSelector.evaluate(
                recoveredFromFundamental.copy(fundamentalPoseRecovered = false)
            ).accepted
        )
    }

    @Test
    fun rejectsForwardMotionAtTheReferenceMaximumBoundary() {
        val boundary = candidate(medianParallax = 20.0, upperParallax = 25.0)
            .copy(forwardMotionRatio = 0.95)

        val evaluation = InitialPairSelector.evaluate(boundary, InitialPairPolicy.initialization())

        assertFalse(evaluation.accepted)
        assertTrue(evaluation.reason.contains("forward motion"))
    }

    @Test
    fun homographyDominanceUsesEssentialSupportAndRejectsAboveTheStrictBoundary() {
        val candidate = candidate(
            medianParallax = 6.0,
            upperParallax = 12.0
        )
        val exactDominanceBoundary = InitialPairSelector.evaluate(
            candidate.copy(homographyInlierRatio = 0.8)
        )
        val planar = InitialPairSelector.evaluate(
            candidate.copy(homographyInlierRatio = 1.2)
        )

        assertTrue(exactDominanceBoundary.accepted)
        assertFalse(planar.accepted)
        assertTrue(planar.reason.contains("planar or panoramic"))
    }

    @Test
    fun selectorDoesNotTreatHigherParallaxAsAnEscapeFromPlanarClassification() {
        val candidate = candidate(
            medianParallax = 20.0,
            upperParallax = 25.0,
            homographyRatio = 1.2
        )

        val evaluation = InitialPairSelector.evaluate(candidate)

        assertFalse(evaluation.accepted)
        assertTrue(evaluation.reason.contains("planar or panoramic"))
    }

    @Test
    fun sourceGatedHomographyPoseUsesItsOwnPlanarAndForwardMotionChecks() {
        val candidate = candidate(
            medianParallax = 20.0,
            upperParallax = 25.0,
            homographyRatio = 1.2,
            essentialToFundamentalRatio = 0.5
        ).copy(
            forwardMotionRatio = 0.99,
            homographyPoseRecovered = true,
            poseSource = SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY
        )

        val evaluation = InitialPairSelector.evaluate(
            candidate,
            InitialPairPolicy.initialization()
        )

        assertTrue(evaluation.accepted)
    }

    @Test
    fun returnsNoSelectionWhenEveryPairFailsQualityGates() {
        val result = InitialPairSelector.select(
            listOf(candidate(inliers = 20), candidate(secondFrame = 9, inliers = 25))
        )

        assertNotNull(result)
        assertNull(result.selected)
        assertFalse(result.evaluations.any { it.accepted })
    }

    private fun candidate(
        secondFrame: Int = 10,
        matched: Int = 700,
        inliers: Int = 420,
        coverage: Int = 12,
        medianParallax: Double = 6.0,
        upperParallax: Double = 12.0,
        cheirality: Double = 0.8,
        homographyRatio: Double = 0.55,
        essentialToFundamentalRatio: Double = 1.0
    ) = InitialPairCandidate(
        firstFrame = 0,
        secondFrame = secondFrame,
        matchedCount = matched,
        inlierCount = inliers,
        coveredGridCells = coverage,
        medianParallaxDegrees = medianParallax,
        upperQuartileParallaxDegrees = upperParallax,
        cheiralityRatio = cheirality,
        homographyInlierRatio = homographyRatio,
        forwardMotionRatio = 0.35,
        essentialToFundamentalInlierRatio = essentialToFundamentalRatio
    )
}
