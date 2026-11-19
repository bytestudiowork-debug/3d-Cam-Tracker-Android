/*
 * Android adaptation of COLMAP's IncrementalMapper::IterativeLocalRefinement in
 * src/colmap/sfm/incremental_mapper.cc at commit
 * c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
 * Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
 * See app/src/main/assets/licenses/colmap-track-completion-license.txt.
 */
package com.trackstudio.app.tracking.colmap

/** Runs bounded local BA rounds until the changed-observation ratio converges. */
internal object ColmapIterativeLocalRefinement {
    const val DEFAULT_MAX_REFINEMENTS = 2
    const val DEFAULT_MAX_REFINEMENT_CHANGE = 0.001

    enum class LossMode { SOFT_L1, TRIVIAL }

    data class Report(
        val adjustedObservations: Int,
        val mergedObservations: Int = 0,
        val completedObservations: Int = 0,
        val filteredObservations: Int = 0
    ) {
        init {
            require(adjustedObservations >= 0)
            require(mergedObservations >= 0)
            require(completedObservations >= 0)
            require(filteredObservations >= 0)
        }

        val changedObservationRatio: Double
            get() = if (adjustedObservations == 0) 0.0 else
                (mergedObservations.toLong() + completedObservations + filteredObservations).toDouble() /
                    adjustedObservations
    }

    data class Iteration(
        val lossMode: LossMode,
        val report: Report
    )

    data class Result(val iterations: List<Iteration>) {
        val changedObservationRatios: List<Double>
            get() = iterations.map { it.report.changedObservationRatio }
    }

    /** [adjust] returns null when local BA cannot form or solve a valid problem. */
    fun run(
        maxRefinements: Int = DEFAULT_MAX_REFINEMENTS,
        maxRefinementChange: Double = DEFAULT_MAX_REFINEMENT_CHANGE,
        filterAfterUnavailableAdjustment: (() -> Int)? = null,
        adjust: (LossMode) -> Report?
    ): Result {
        require(maxRefinements >= 0)
        require(maxRefinementChange.isFinite() && maxRefinementChange >= 0.0)

        val iterations = ArrayList<Iteration>(maxRefinements)
        for (round in 0 until maxRefinements) {
            val lossMode = if (round == 0) LossMode.SOFT_L1 else LossMode.TRIVIAL
            val report = adjust(lossMode)
            if (report == null) {
                if (iterations.isEmpty()) filterAfterUnavailableAdjustment?.invoke()
                break
            }
            iterations += Iteration(lossMode, report)
            if (report.changedObservationRatio < maxRefinementChange) break
        }
        return Result(iterations)
    }
}
