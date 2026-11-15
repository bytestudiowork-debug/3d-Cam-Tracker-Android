package com.trackstudio.app.tracking

import kotlin.math.min

internal object LocalConsensusOptimizer {
    data class Result<M>(
        val model: M,
        val inlierIndices: List<Int>,
        val truncatedSquaredError: Double,
        val refitRounds: Int
    )

    fun <M> refine(
        initialModel: M,
        observationCount: Int,
        initialInliers: List<Int>,
        minimumInliers: Int,
        threshold: Double,
        maximumRefitRounds: Int,
        fitCandidates: (M, List<Int>) -> List<M>,
        residual: (M, Int) -> Double
    ): Result<M>? {
        if (observationCount <= 0 || minimumInliers <= 0 ||
            !threshold.isFinite() || threshold <= 0.0 || threshold > 1e150 ||
            maximumRefitRounds < 0
        ) return null
        val seed = initialInliers.asSequence()
            .filter { it in 0 until observationCount }
            .distinct()
            .sorted()
            .toList()
        if (seed.size < minimumInliers) return null

        var current = score(initialModel, observationCount, threshold, residual)
        if (current.inliers.size < minimumInliers || !current.cost.isFinite()) return null
        var fitIndices = seed
        var acceptedRefits = 0

        for (round in 0 until maximumRefitRounds) {
            val candidates = fitCandidates(current.model, fitIndices)
            if (candidates.isEmpty()) break
            var bestCandidate: Scored<M>? = null
            for (candidate in candidates) {
                val scored = score(candidate, observationCount, threshold, residual)
                if (scored.inliers.size < minimumInliers || !scored.cost.isFinite()) continue
                val incumbent = bestCandidate
                if (incumbent == null || isBetter(scored, incumbent)) bestCandidate = scored
            }
            val candidate = bestCandidate ?: break
            if (!isBetter(candidate, current)) break

            val previousInliers = current.inliers
            current = candidate
            fitIndices = current.inliers
            acceptedRefits++
            if (fitIndices == previousInliers) break
        }

        return Result(
            model = current.model,
            inlierIndices = current.inliers,
            truncatedSquaredError = current.cost,
            refitRounds = acceptedRefits
        )
    }

    private data class Scored<M>(
        val model: M,
        val inliers: List<Int>,
        val cost: Double
    )

    private fun <M> score(
        model: M,
        observationCount: Int,
        threshold: Double,
        residual: (M, Int) -> Double
    ): Scored<M> {
        val thresholdSquared = threshold * threshold
        val inliers = ArrayList<Int>()
        var cost = 0.0
        for (index in 0 until observationCount) {
            val error = residual(model, index)
            val safeError = if (error.isFinite() && error >= 0.0) error else Double.POSITIVE_INFINITY
            if (safeError <= threshold) inliers += index
            cost += min(safeError * safeError, thresholdSquared)
        }
        return Scored(model, inliers, cost)
    }

    private fun <M> isBetter(candidate: Scored<M>, incumbent: Scored<M>): Boolean {
        if (candidate.inliers.size != incumbent.inliers.size) {
            return candidate.inliers.size > incumbent.inliers.size
        }
        val tolerance = 1e-12 * maxOf(1.0, incumbent.cost)
        return candidate.cost < incumbent.cost - tolerance
    }
}
