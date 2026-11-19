package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureTrack

internal object GlobalBundleAdjustmentInput {
    const val MINIMUM_TRACK_VIEWS = 3

    data class Selection(
        val frameIds: List<Int>,
        val tracks: List<CameraFeatureTrack>
    )

    fun select(
        frameCentersById: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        tracks: List<CameraFeatureTrack>
    ): Selection {
        val selectedTracks = tracks.mapNotNull { track ->
            if (track.id !in pointsByTrackId) return@mapNotNull null
            val observations = track.observations.filter { observation ->
                observation.frameIndex in frameCentersById &&
                    observation.frameIndex in rotationsByFrame
            }
            if (observations.asSequence().map { it.frameIndex }.distinct().count() <
                MINIMUM_TRACK_VIEWS
            ) {
                null
            } else {
                track.copy(observations = observations)
            }
        }
        val frameIds = selectedTracks.asSequence()
            .flatMap { it.observations.asSequence().map { observation -> observation.frameIndex } }
            .distinct()
            .sorted()
            .toList()
        return Selection(frameIds = frameIds, tracks = selectedTracks)
    }
}
