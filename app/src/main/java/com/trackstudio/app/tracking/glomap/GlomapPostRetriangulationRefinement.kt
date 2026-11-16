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
 * Android adaptation of the post-retriangulation controller sequence in
 * glomap/controllers/global_mapper.cc from GLOMAP commit
 * 0edb1b8435e0f9a594318908b81a31f078a51bf7.
 */
package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraIntrinsics
import com.trackstudio.app.tracking.SolverObjective

internal object GlomapPostRetriangulationRefinement {
    const val DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED = 1e-2
    const val DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES = 1.0

    data class State(
        val frameCentersById: Map<Int, DoubleArray>,
        val rotationsByFrame: Map<Int, DoubleArray>,
        val pointsByTrackId: Map<Int, DoubleArray>,
        val tracks: List<CameraFeatureTrack>,
        val intrinsics: CameraIntrinsics
    )

    data class Result(
        val state: State,
        val firstBundleAdjustment: GlomapGlobalBundleAdjuster.Result?,
        val firstBundleAdjustmentAccepted: Boolean,
        val secondBundleAdjustment: GlomapGlobalBundleAdjuster.Result?,
        val secondBundleAdjustmentAccepted: Boolean
    ) {
        val acceptedBundleAdjustmentCount: Int
            get() = (if (firstBundleAdjustmentAccepted) 1 else 0) +
                (if (secondBundleAdjustmentAccepted) 1 else 0)

        val lastAcceptedBundleAdjustment: GlomapGlobalBundleAdjuster.Result?
            get() = when {
                secondBundleAdjustmentAccepted -> secondBundleAdjustment
                firstBundleAdjustmentAccepted -> firstBundleAdjustment
                else -> null
            }
    }

    fun refine(
        initialState: State,
        solveBundleAdjustment: (State, solveIndex: Int) -> GlomapGlobalBundleAdjuster.Result?,
        filterByReprojection: (State, maxReprojectionError: Double) -> GlomapTrackFilter.Result =
            { state, threshold ->
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
        filterByTriangulationAngle: (State, minTriangulationAngleDegrees: Double) ->
            GlomapTrackFilter.Result = { state, minAngle ->
                GlomapTrackFilter.filterTrackTriangulationAngle(
                    tracks = state.tracks,
                    cameraCentersByFrame = state.frameCentersById,
                    pointsByTrackId = state.pointsByTrackId,
                    minAngleDegrees = minAngle
                )
            }
    ): Result {
        val firstCandidate = solveBundleAdjustment(initialState, 0)
        val firstAccepted = firstCandidate?.takeIf {
            SolverObjective.didNotIncrease(it.initialCost, it.finalCost)
        }

        var state = if (firstAccepted == null) {
            initialState
        } else {
            initialState.with(firstAccepted)
        }

        var secondCandidate: GlomapGlobalBundleAdjuster.Result? = null
        var secondAccepted: GlomapGlobalBundleAdjuster.Result? = null

        if (firstAccepted != null) {
            val afterFirstReprojection = filterByReprojection(
                state,
                DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED
            )
            state = state.copy(tracks = afterFirstReprojection.tracks)

            secondCandidate = solveBundleAdjustment(state, 1)
            secondAccepted = secondCandidate?.takeIf {
                SolverObjective.didNotIncrease(it.initialCost, it.finalCost)
            }
            if (secondAccepted != null) {
                state = state.with(secondAccepted)
            }
        }

        val finalReprojection = filterByReprojection(
            state,
            DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED
        )
        state = state.copy(tracks = finalReprojection.tracks)

        val finalAngle = filterByTriangulationAngle(
            state,
            DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES
        )
        state = state.copy(tracks = finalAngle.tracks)

        return Result(
            state = state,
            firstBundleAdjustment = firstCandidate,
            firstBundleAdjustmentAccepted = firstAccepted != null,
            secondBundleAdjustment = secondCandidate,
            secondBundleAdjustmentAccepted = secondAccepted != null
        )
    }

    private fun State.with(result: GlomapGlobalBundleAdjuster.Result): State = copy(
        frameCentersById = frameCentersById + result.frameCentersById,
        rotationsByFrame = rotationsByFrame + result.rotationsByFrame,
        pointsByTrackId = pointsByTrackId + result.pointsByTrackId,
        intrinsics = result.intrinsics
    )
}
