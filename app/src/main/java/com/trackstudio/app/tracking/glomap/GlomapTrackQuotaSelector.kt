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
 * Android adaptation of GLOMAP TrackEngine::FindTracksForProblem at commit
 * 0edb1b8435e0f9a594318908b81a31f078a51bf7. The existing Android track builder
 * already supplies candidates ordered by descending length and stable
 * representative ID, so this selector retains that source-equivalent priority.
 * See app/src/main/assets/licenses/glomap-track-selection-license.txt.
 */
package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureTrack

/** Selects full tracks for the global optimization problem using GLOMAP quotas. */
internal object GlomapTrackQuotaSelector {
    const val DEFAULT_MIN_TRACKS_PER_VIEW = -1
    const val MIN_VIEWS_PER_TRACK = 3
    const val MAX_VIEWS_PER_TRACK = 100
    const val DEFAULT_MAX_TRACKS = 10_000_000

    data class Result(
        val selectedTracks: List<CameraFeatureTrack>,
        /** The selector's per-view counters, including zero-count registered frames. */
        val quotaCounterByFrame: Map<Int, Int>,
        val lengthEligibleTrackCount: Int,
        val stoppedWhenViewsReachedQuota: Boolean,
        val stoppedAfterExceedingMaxTracks: Boolean
    )

    /**
     * `priorityOrderedTracks` must be ordered longest-first, with stable source
     * IDs descending on ties. GlobalTrackGraphBuilder already provides this order.
     *
     * The pinned header defaults the quota to -1, but its literal `count > -1`
     * guard skips every track. Android treats any negative quota as disabled so
     * the default pipeline retains its existing tracks; nonnegative quotas follow
     * the pinned loop, including its post-increment `>` boundary.
     */
    fun select(
        priorityOrderedTracks: List<CameraFeatureTrack>,
        registeredFrameIds: Set<Int>,
        minTracksPerView: Int = DEFAULT_MIN_TRACKS_PER_VIEW,
        maxNumTracks: Int = DEFAULT_MAX_TRACKS
    ): Result {
        require(maxNumTracks >= 0)

        val tracksPerView = LinkedHashMap<Int, Int>(registeredFrameIds.size)
        registeredFrameIds.sorted().forEach { tracksPerView[it] = 0 }
        var camerasLeft = registeredFrameIds.size
        var lengthEligibleTrackCount = 0
        var stoppedWhenViewsReachedQuota = false
        var stoppedAfterExceedingMaxTracks = false
        val selectedTracks = ArrayList<CameraFeatureTrack>()

        for (track in priorityOrderedTracks) {
            if (track.observations.size !in MIN_VIEWS_PER_TRACK..MAX_VIEWS_PER_TRACK) continue
            lengthEligibleTrackCount++

            val registeredObservations = track.observations.filter {
                it.frameIndex in registeredFrameIds
            }
            val distinctRegisteredViews = registeredObservations
                .mapTo(HashSet()) { it.frameIndex }
            if (distinctRegisteredViews.size < MIN_VIEWS_PER_TRACK) continue

            val selectedTrack = track.copy(observations = registeredObservations)
            var added = false
            for (observation in registeredObservations) {
                val tracksInView = tracksPerView[observation.frameIndex] ?: continue
                if (minTracksPerView >= 0 && tracksInView > minTracksPerView) continue

                val updatedCount = tracksInView + 1
                tracksPerView[observation.frameIndex] = updatedCount
                if (minTracksPerView >= 0 && updatedCount > minTracksPerView) {
                    camerasLeft--
                }
                if (!added) {
                    selectedTracks += selectedTrack
                    added = true
                }
            }

            if (minTracksPerView >= 0 && camerasLeft == 0) {
                stoppedWhenViewsReachedQuota = true
                break
            }
            // The pinned implementation checks `>` after inserting a track.
            if (selectedTracks.size > maxNumTracks) {
                stoppedAfterExceedingMaxTracks = true
                break
            }
        }

        return Result(
            selectedTracks = selectedTracks,
            quotaCounterByFrame = tracksPerView,
            lengthEligibleTrackCount = lengthEligibleTrackCount,
            stoppedWhenViewsReachedQuota = stoppedWhenViewsReachedQuota,
            stoppedAfterExceedingMaxTracks = stoppedAfterExceedingMaxTracks
        )
    }
}
