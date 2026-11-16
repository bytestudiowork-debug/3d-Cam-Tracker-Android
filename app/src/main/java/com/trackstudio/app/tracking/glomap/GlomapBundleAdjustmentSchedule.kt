/*
 * Copyright (c) 2024, ETH Zurich. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the name of ETH Zurich nor the names of its contributors may
 *       be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Android adaptation of the staged BA schedule in
 * https://github.com/colmap/glomap/blob/0edb1b8435e0f9a594318908b81a31f078a51bf7/glomap/controllers/global_mapper.cc
 */
package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraIntrinsics
import com.trackstudio.app.tracking.SolverObjective
import kotlin.math.max

internal object GlomapBundleAdjustmentSchedule {
    const val DEFAULT_NUM_ITERATIONS = 3
    const val FILTERED_TRACK_STOP_RATIO = 1e-3
    const val MAX_STAGE_COUNT = DEFAULT_NUM_ITERATIONS * 2

    data class State(
        val frameCentersById: Map<Int, DoubleArray>,
        val rotationsByFrame: Map<Int, DoubleArray>,
        val pointsByTrackId: Map<Int, DoubleArray>,
        val tracks: List<CameraFeatureTrack>,
        val intrinsics: CameraIntrinsics
    )

    data class Result(
        val bundleResult: GlomapGlobalBundleAdjuster.Result,
        val tracks: List<CameraFeatureTrack>,
        val completedStages: Int
    )

    fun solve(
        initialState: State,
        stageSolver: (State, optimizeRotations: Boolean, stageIndex: Int) -> GlomapGlobalBundleAdjuster.Result?,
        numIterations: Int = DEFAULT_NUM_ITERATIONS,
        optimizeRotations: Boolean = true,
        filterByReprojection: (State, Double) -> GlomapTrackFilter.Result = { state, threshold ->
            GlomapTrackFilter.filterTracksByReprojection(
                tracks = state.tracks,
                cameraCentersByFrame = state.frameCentersById,
                rotationsByFrame = state.rotationsByFrame,
                pointsByTrackId = state.pointsByTrackId,
                intrinsics = state.intrinsics,
                maxReprojectionError = threshold,
                inNormalizedImage = true
            )
        },
        filterByTriangulationAngle: (State) -> GlomapTrackFilter.Result = { state ->
            GlomapTrackFilter.filterTrackTriangulationAngle(
                tracks = state.tracks,
                cameraCentersByFrame = state.frameCentersById,
                pointsByTrackId = state.pointsByTrackId
            )
        }
    ): Result? {
        require(numIterations > 0)

        var state = initialState
        var lastBundleResult: GlomapGlobalBundleAdjuster.Result? = null
        var initialCost: Double? = null
        var iterationCount = 0
        var stageIndex = 0
        var iteration = 0

        while (iteration < numIterations) {
            val positionResult = stageSolver(state, false, stageIndex++) ?: return null
            if (!SolverObjective.didNotIncrease(positionResult.initialCost, positionResult.finalCost)) return null
            if (initialCost == null) initialCost = positionResult.initialCost
            iterationCount += positionResult.iterations
            state = state.with(positionResult)
            lastBundleResult = positionResult

            if (optimizeRotations) {
                val rotationResult = stageSolver(state, true, stageIndex++) ?: return null
                if (!SolverObjective.didNotIncrease(rotationResult.initialCost, rotationResult.finalCost)) return null
                iterationCount += rotationResult.iterations
                state = state.with(rotationResult)
                lastBundleResult = rotationResult
            }

            var status = true
            var filteredTrackCount = 0
            while (status && iteration < numIterations) {
                val scaling = max(3 - iteration, 1).toDouble()
                val filterResult = filterByReprojection(
                    state,
                    scaling * GlomapTrackFilter.DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED
                )
                filteredTrackCount += filterResult.changedTrackCount
                state = state.copy(tracks = filterResult.tracks)

                if (filteredTrackCount > FILTERED_TRACK_STOP_RATIO * state.tracks.size) {
                    status = false
                } else {
                    iteration++
                }
            }
            if (status) break
            iteration++
        }

        val lastResult = lastBundleResult ?: return null
        val baseThreshold = GlomapTrackFilter.DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED
        state = state.copy(
            tracks = filterByReprojection(state, baseThreshold).tracks
        )
        state = state.copy(
            tracks = filterByTriangulationAngle(state).tracks
        )

        return Result(
            bundleResult = lastResult.copy(
                initialCost = initialCost ?: return null,
                iterations = iterationCount
            ),
            tracks = state.tracks,
            completedStages = stageIndex
        )
    }

    private fun State.with(result: GlomapGlobalBundleAdjuster.Result): State = copy(
        frameCentersById = result.frameCentersById,
        rotationsByFrame = result.rotationsByFrame,
        pointsByTrackId = result.pointsByTrackId,
        intrinsics = result.intrinsics
    )
}
