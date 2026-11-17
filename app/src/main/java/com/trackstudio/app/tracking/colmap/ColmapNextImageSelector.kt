/*
 * Android adaptation of COLMAP IncrementalMapperImpl::FindNextImages and
 * VisibilityPyramid from the pinned COLMAP revision.
 * Sources: src/colmap/sfm/incremental_mapper_impl.cc,
 * src/colmap/sfm/incremental_mapper.h,
 * src/colmap/sfm/observation_manager.cc, and
 * src/colmap/scene/visibility_pyramid.{h,cc} at
 * c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
 * Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
 * See app/src/main/assets/licenses/colmap-image-selection-license.txt.
 */
package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation

/** Ranks eligible image registrations by visible-point count or spatial coverage. */
internal object ColmapNextImageSelector {
    const val DEFAULT_MIN_VISIBLE_POINTS = 30
    const val DEFAULT_MAX_REGISTRATION_TRIALS = 3
    const val DEFAULT_PYRAMID_LEVELS = 6
    // Bounds expensive mobile retries; ranking within the batch remains source-faithful.
    const val ANDROID_MAX_CANDIDATES_PER_BATCH = 4

    enum class Method {
        MAX_VISIBLE_POINTS_NUM,
        MAX_VISIBLE_POINTS_RATIO,
        MIN_UNCERTAINTY
    }

    data class Options(
        val minimumVisiblePoints: Int = DEFAULT_MIN_VISIBLE_POINTS,
        val maximumRegistrationTrials: Int = DEFAULT_MAX_REGISTRATION_TRIALS,
        val method: Method = Method.MIN_UNCERTAINTY,
        val pyramidLevels: Int = DEFAULT_PYRAMID_LEVELS
    )

    data class Candidate(
        val frameIndex: Int,
        val visibleObservations: List<CameraFeatureObservation>,
        val totalObservations: Int,
        val registrationTrials: Int,
        val isRegistered: Boolean = false,
        val isFiltered: Boolean = false
    ) {
        val visiblePointCount: Int
            get() = visibleObservations.size
    }

    /**
     * Mirrors COLMAP's two ranked buckets: never-tried, non-filtered images first;
     * previously tried or filtered images second. Frame index breaks score ties
     * deterministically; COLMAP leaves equal-score order unspecified.
     */
    fun rank(
        candidates: List<Candidate>,
        imageWidth: Int,
        imageHeight: Int,
        options: Options = Options()
    ): List<Candidate> {
        require(imageWidth > 0 && imageHeight > 0)
        require(options.minimumVisiblePoints > 0)
        require(options.maximumRegistrationTrials > 0)
        require(options.pyramidLevels in 1..10)
        candidates.forEach { candidate ->
            require(candidate.frameIndex >= 0)
            require(candidate.totalObservations >= 0)
            require(candidate.registrationTrials >= 0)
        }

        val eligible = candidates.filter { candidate ->
            !candidate.isRegistered &&
                candidate.visiblePointCount >= options.minimumVisiblePoints &&
                candidate.registrationTrials < options.maximumRegistrationTrials
        }
        val fresh = eligible.filter { !it.isFiltered && it.registrationTrials == 0 }
        val previouslyTried = eligible.filter { it.isFiltered || it.registrationTrials > 0 }
        return rankBucket(fresh, imageWidth, imageHeight, options) +
            rankBucket(previouslyTried, imageWidth, imageHeight, options)
    }

    /** Exact occupied-cell score used by COLMAP's fixed six-level pyramid. */
    internal fun visibilityScore(
        observations: List<CameraFeatureObservation>,
        imageWidth: Int,
        imageHeight: Int,
        pyramidLevels: Int = DEFAULT_PYRAMID_LEVELS
    ): Long {
        require(imageWidth > 0 && imageHeight > 0)
        require(pyramidLevels in 1..10)
        val maximumDimension = 1 shl pyramidLevels
        val occupiedCells = Array(pyramidLevels) { HashSet<Int>() }
        var score = 0L

        for (observation in observations) {
            if (!observation.x.isFinite() || !observation.y.isFinite()) continue
            var cellX = (maximumDimension.toDouble() * observation.x / imageWidth)
                .toInt().coerceIn(0, maximumDimension - 1)
            var cellY = (maximumDimension.toDouble() * observation.y / imageHeight)
                .toInt().coerceIn(0, maximumDimension - 1)
            var dimension = maximumDimension
            for (level in pyramidLevels - 1 downTo 0) {
                val cellIndex = cellY * dimension + cellX
                if (occupiedCells[level].add(cellIndex)) {
                    score += dimension.toLong() * dimension
                }
                cellX = cellX ushr 1
                cellY = cellY ushr 1
                dimension = dimension ushr 1
            }
        }
        return score
    }

    private fun rankBucket(
        candidates: List<Candidate>,
        imageWidth: Int,
        imageHeight: Int,
        options: Options
    ): List<Candidate> = candidates.sortedWith(
        compareByDescending<Candidate> { candidate ->
            when (options.method) {
                Method.MAX_VISIBLE_POINTS_NUM -> candidate.visiblePointCount.toDouble()
                Method.MAX_VISIBLE_POINTS_RATIO -> {
                    if (candidate.totalObservations == 0) 0.0
                    else candidate.visiblePointCount.toDouble() / candidate.totalObservations
                }
                Method.MIN_UNCERTAINTY -> visibilityScore(
                    candidate.visibleObservations,
                    imageWidth,
                    imageHeight,
                    options.pyramidLevels
                ).toDouble()
            }
        }.thenBy { it.frameIndex }
    )
}
