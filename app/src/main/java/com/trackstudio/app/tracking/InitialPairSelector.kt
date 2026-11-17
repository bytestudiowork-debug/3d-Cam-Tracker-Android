package com.trackstudio.app.tracking

import kotlin.math.ln

/**
 * Quality gate and ranking policy for two-view SfM initialization.
 *
 * Metrics are produced by the OpenCV pair evaluator. Keeping ranking pure Kotlin
 * makes the policy deterministic, inexpensive, and independently testable.
 * This selector deliberately rejects weak geometry instead of returning the
 * least-bad pair and pretending that it is a valid reconstruction.
 */
internal object InitialPairSelector {

    fun select(
        candidates: List<InitialPairCandidate>,
        policy: InitialPairPolicy = InitialPairPolicy()
    ): InitialPairSelection {
        if (candidates.isEmpty()) {
            return InitialPairSelection(null, emptyList(), "No candidate frame pairs were evaluated")
        }

        val evaluations = candidates.map { candidate ->
            evaluate(candidate, policy)
        }
        val accepted = evaluations
            .filter { it.accepted }
            .sortedWith(
                compareByDescending<InitialPairEvaluation> { it.score }
                    .thenByDescending { it.candidate.inlierCount }
                    .thenBy { it.candidate.frameGap }
                    .thenBy { it.candidate.firstFrame }
            )
        val best = accepted.firstOrNull()
        return if (best == null) {
            InitialPairSelection(
                selected = null,
                evaluations = evaluations,
                reason = "No pair passed the geometric initialization quality gates"
            )
        } else {
            InitialPairSelection(
                selected = best.candidate,
                evaluations = evaluations,
                reason = "Selected the best-conditioned pair from ${accepted.size} valid candidates"
            )
        }
    }

    fun evaluate(
        candidate: InitialPairCandidate,
        policy: InitialPairPolicy = InitialPairPolicy()
    ): InitialPairEvaluation {
        val rejection = when {
            candidate.firstFrame < 0 || candidate.secondFrame <= candidate.firstFrame ->
                "Invalid frame order"
            candidate.frameGap < policy.minimumFrameGap ->
                "Frames are too close in time"
            candidate.matchedCount <= 0 || candidate.inlierCount < policy.minimumInliers ->
                "Insufficient verified correspondences"
            candidate.inlierCount > candidate.matchedCount ->
                "Inlier count exceeds match count"
            candidate.inlierRatio < policy.minimumInlierRatio ->
                "Low geometric inlier ratio"
            candidate.coveredGridCells < policy.minimumCoveredGridCells ->
                "Features are concentrated in a small image region"
            candidate.coveredGridCells > policy.gridCellCount ->
                "Invalid image coverage metric"
            !candidate.cheiralityRatio.isFinite() ||
                candidate.cheiralityRatio !in 0.0..1.0 ->
                "Invalid cheirality metric"
            !candidate.homographyInlierRatio.isFinite() ||
                candidate.homographyInlierRatio < 0.0 ->
                "Invalid homography metric"
            !candidate.essentialToFundamentalInlierRatio.isFinite() ||
                candidate.essentialToFundamentalInlierRatio < 0.0 ->
                "Invalid calibrated-model agreement metric"
            candidate.essentialToFundamentalInlierRatio <=
                policy.minimumEssentialToFundamentalInlierRatio ->
                "Essential geometry is inconsistent with the fundamental model"
            !candidate.forwardMotionRatio.isFinite() ||
                candidate.forwardMotionRatio !in 0.0..1.0 ->
                "Invalid forward-motion metric"
            !candidate.medianParallaxDegrees.isFinite() ||
                candidate.medianParallaxDegrees <= policy.minimumMedianParallaxDegrees ->
                "Insufficient median triangulation parallax"
            !candidate.upperQuartileParallaxDegrees.isFinite() ||
                candidate.upperQuartileParallaxDegrees < policy.minimumUpperQuartileParallaxDegrees ->
                "Too few points have useful triangulation parallax"
            candidate.cheiralityRatio < policy.minimumCheiralityRatio ->
                "Too few points satisfy positive-depth cheirality"
            candidate.forwardMotionRatio >= policy.maximumForwardMotionRatio ->
                "Pair is dominated by degenerate forward motion"
            candidate.homographyInlierRatio > policy.planarDominanceRatio ->
                "Homography-dominated pair is planar or panoramic for calibrated initialization"
            else -> null
        }

        if (rejection != null) {
            return InitialPairEvaluation(candidate, accepted = false, score = 0.0, reason = rejection)
        }

        val countScore = (ln(1.0 + candidate.inlierCount) /
            ln(1.0 + policy.inlierCountSaturation)).coerceIn(0.0, 1.0)
        val ratioScore = ((candidate.inlierRatio - policy.minimumInlierRatio) /
            (policy.targetInlierRatio - policy.minimumInlierRatio).coerceAtLeast(1e-6))
            .coerceIn(0.0, 1.0)
        val coverageScore = (candidate.coveredGridCells.toDouble() /
            policy.gridCellCount).coerceIn(0.0, 1.0)
        val medianParallaxScore = (candidate.medianParallaxDegrees /
            policy.targetMedianParallaxDegrees).coerceIn(0.0, 1.0)
        val upperParallaxScore = (candidate.upperQuartileParallaxDegrees /
            policy.targetUpperQuartileParallaxDegrees).coerceIn(0.0, 1.0)
        val parallaxScore = 0.65 * medianParallaxScore + 0.35 * upperParallaxScore
        val cheiralityScore = candidate.cheiralityRatio.coerceIn(0.0, 1.0)
        // Balanced scoring avoids selecting a pair on raw match count alone.
        val score = (
            0.18 * countScore +
                0.17 * ratioScore +
                0.18 * coverageScore +
                0.27 * parallaxScore +
                0.20 * cheiralityScore
            )

        return InitialPairEvaluation(
            candidate = candidate,
            accepted = true,
            score = score,
            reason = "Passed correspondence, coverage, parallax and cheirality gates"
        )
    }

    /**
     * Creates a bounded, deterministic set of temporal pairs for a coarse
     * initialization search. The caller still evaluates image geometry for
     * every pair; frame distance is only a search schedule, never a quality score.
     */
    fun temporalCandidates(
        frameCount: Int,
        anchorCount: Int = 8,
        offsets: List<Int> = listOf(2, 4, 6, 9, 13, 18, 24, 32),
        maximumPairs: Int = 64
    ): List<Pair<Int, Int>> {
        if (frameCount < 2 || anchorCount <= 0 || maximumPairs <= 0) return emptyList()
        val validOffsets = offsets.filter { it > 0 }.distinct().sorted()
        if (validOffsets.isEmpty()) return emptyList()

        val lastAnchor = (frameCount - 2).coerceAtLeast(0)
        val anchors = if (anchorCount == 1) {
            listOf(0)
        } else {
            (0 until anchorCount).map { index ->
                (index * lastAnchor / (anchorCount - 1)).coerceIn(0, lastAnchor)
            }.distinct()
        }

        val pairs = linkedSetOf<Pair<Int, Int>>()
        for (anchor in anchors) {
            for (offset in validOffsets) {
                val target = anchor + offset
                if (target < frameCount) pairs += anchor to target
                if (pairs.size >= maximumPairs) return pairs.toList()
            }
        }
        return pairs.toList()
    }

    /** Uniform, deterministic frame samples used by the bounded correspondence pre-search. */
    fun uniformFrameSamples(frameCount: Int, maximumFrames: Int): List<Int> {
        if (frameCount <= 0 || maximumFrames <= 0) return emptyList()
        if (frameCount <= maximumFrames) return (0 until frameCount).toList()
        if (maximumFrames == 1) return listOf(0)
        val lastFrame = frameCount - 1
        return (0 until maximumFrames).map { sampleIndex ->
            (sampleIndex.toLong() * lastFrame / (maximumFrames - 1)).toInt()
        }.distinct()
    }

    /** Enumerates unique chronological pairs from a bounded frame sample. */
    fun sampledFramePairs(
        frameIndices: List<Int>,
        minimumFrameGap: Int = 2,
        maximumPairs: Int = Int.MAX_VALUE
    ): List<Pair<Int, Int>> {
        if (minimumFrameGap < 1 || maximumPairs <= 0) return emptyList()
        val frames = frameIndices.filter { it >= 0 }.distinct().sorted()
        val pairs = ArrayList<Pair<Int, Int>>()
        for (firstIndex in frames.indices) {
            for (secondIndex in firstIndex + 1 until frames.size) {
                val first = frames[firstIndex]
                val second = frames[secondIndex]
                if (second - first < minimumFrameGap) continue
                pairs += first to second
                if (pairs.size >= maximumPairs) return pairs
            }
        }
        return pairs
    }

    /**
     * Ranks bounded candidate edges by coarse per-frame match degree, then pair support.
     * This mirrors COLMAP's image-first/connected-image ordering using Android's sampled
     * descriptor graph; temporal candidates remain as a bounded fallback.
     */
    fun correspondenceRankedCandidates(
        candidatePairs: List<Pair<Int, Int>>,
        temporalFallbackPairs: List<Pair<Int, Int>>,
        matchSupportByPair: Map<Pair<Int, Int>, Int>,
        maximumCandidates: Int = 36,
        correspondenceRankBudget: Int = 24
    ): List<Pair<Int, Int>> {
        if (maximumCandidates <= 0) return emptyList()
        require(correspondenceRankBudget >= 0)
        val uniquePairs = linkedSetOf<Pair<Int, Int>>()
        (candidatePairs + temporalFallbackPairs).forEach { pair ->
            val normalized = normalizePair(pair) ?: return@forEach
            uniquePairs += normalized
        }
        if (uniquePairs.isEmpty()) return emptyList()

        val support = uniquePairs.associateWith { pair ->
            matchSupportByPair[ pair ] ?: matchSupportByPair[ pair.second to pair.first ] ?: 0
        }
        val frameDegree = HashMap<Int, Int>()
        support.forEach { (pair, count) ->
            val nonNegativeCount = count.coerceAtLeast(0)
            frameDegree[pair.first] = (frameDegree[pair.first] ?: 0) + nonNegativeCount
            frameDegree[pair.second] = (frameDegree[pair.second] ?: 0) + nonNegativeCount
        }
        val rankedEdges = uniquePairs
            .filter { (support[it] ?: 0) > 0 }
            .sortedWith(
                compareByDescending<Pair<Int, Int>> {
                    maxOf(frameDegree[it.first] ?: 0, frameDegree[it.second] ?: 0)
                }.thenByDescending { support[it] ?: 0 }
                    .thenBy { it.first }
                    .thenBy { it.second }
            )
        val rankedBudget = minOf(maximumCandidates, correspondenceRankBudget)
        val selected = ArrayList<Pair<Int, Int>>(maximumCandidates)
        selected += rankedEdges.take(rankedBudget)

        val temporalOrder = temporalFallbackPairs.mapIndexedNotNull { index, pair ->
            normalizePair(pair)?.let { normalized -> Triple(index, normalized, support[normalized] ?: 0) }
        }.distinctBy { it.second }
            .sortedWith(compareByDescending<Triple<Int, Pair<Int, Int>, Int>> { it.third }.thenBy { it.first })
        for ((_, pair, _) in temporalOrder) {
            if (selected.size >= maximumCandidates) break
            if (pair !in selected) selected += pair
        }
        return selected
    }

    private fun normalizePair(pair: Pair<Int, Int>): Pair<Int, Int>? {
        if (pair.first == pair.second || pair.first < 0 || pair.second < 0) return null
        return minOf(pair.first, pair.second) to maxOf(pair.first, pair.second)
    }
}

internal data class InitialPairCandidate(
    val firstFrame: Int,
    val secondFrame: Int,
    val matchedCount: Int,
    val inlierCount: Int,
    val coveredGridCells: Int,
    val medianParallaxDegrees: Double,
    val upperQuartileParallaxDegrees: Double,
    val cheiralityRatio: Double,
    val homographyInlierRatio: Double,
    /** Fraction of translation aligned with the viewing direction, in [0, 1]. */
    val forwardMotionRatio: Double,
    /** Relative rotation R_second_from_first recovered from this image pair, row-major. */
    val relativeRotation: DoubleArray? = null,
    /** Unit translation direction in the second camera coordinate system. */
    val relativeTranslationDirection: DoubleArray? = null,
    /** Verified keypoint index pairs from this exact geometry estimate. */
    val verifiedMatches: List<Pair<Int, Int>> = emptyList(),
    /** Essential support divided by fundamental support; values above one are valid. */
    val essentialToFundamentalInlierRatio: Double = 1.0
) {
    val frameGap: Int
        get() = secondFrame - firstFrame

    val inlierRatio: Double
        get() = if (matchedCount <= 0) 0.0 else inlierCount.toDouble() / matchedCount
}

internal data class InitialPairPolicy(
    val minimumFrameGap: Int = 2,
    val minimumInliers: Int = 60,
    val minimumInlierRatio: Double = 0.25,
    val targetInlierRatio: Double = 0.55,
    val minimumCoveredGridCells: Int = 5,
    val gridCellCount: Int = 16,
    val minimumMedianParallaxDegrees: Double = 1.0,
    val minimumUpperQuartileParallaxDegrees: Double = 2.5,
    val targetMedianParallaxDegrees: Double = 8.0,
    val targetUpperQuartileParallaxDegrees: Double = 16.0,
    val minimumCheiralityRatio: Double = 0.55,
    val maximumForwardMotionRatio: Double = 0.95,
    val minimumEssentialToFundamentalInlierRatio: Double = 0.95,
    /** Homography inlier support divided by essential-matrix inlier support. */
    val planarDominanceRatio: Double = 0.8,
    val inlierCountSaturation: Int = 500
) {
    init {
        require(minimumFrameGap >= 1)
        require(minimumInliers >= 5)
        require(minimumInlierRatio in 0.0..1.0)
        require(targetInlierRatio in minimumInlierRatio..1.0)
        require(gridCellCount > 0)
        require(minimumCoveredGridCells in 0..gridCellCount)
        require(minimumMedianParallaxDegrees.isFinite() && minimumMedianParallaxDegrees >= 0.0)
        require(minimumUpperQuartileParallaxDegrees.isFinite() && minimumUpperQuartileParallaxDegrees >= 0.0)
        require(targetMedianParallaxDegrees.isFinite() && targetMedianParallaxDegrees > 0.0)
        require(targetUpperQuartileParallaxDegrees.isFinite() && targetUpperQuartileParallaxDegrees > 0.0)
        require(minimumCheiralityRatio in 0.0..1.0)
        require(maximumForwardMotionRatio in 0.0..1.0)
        require(minimumEssentialToFundamentalInlierRatio.isFinite() &&
            minimumEssentialToFundamentalInlierRatio in 0.0..1.0)
        require(planarDominanceRatio.isFinite() && planarDominanceRatio in 0.0..1.0)
        require(inlierCountSaturation > 0)
    }

    companion object {
        /** Stricter thresholds for choosing a stable two-view reconstruction seed. */
        fun initialization() = InitialPairPolicy(
            minimumInliers = 100,
            minimumMedianParallaxDegrees = 16.0,
            targetMedianParallaxDegrees = 20.0
        )
    }
}

internal data class InitialPairEvaluation(
    val candidate: InitialPairCandidate,
    val accepted: Boolean,
    val score: Double,
    val reason: String
)

internal data class InitialPairSelection(
    val selected: InitialPairCandidate?,
    val evaluations: List<InitialPairEvaluation>,
    val reason: String
)
