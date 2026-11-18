package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureTrack

/**
 * Selects tracks with enough support inside a COLMAP local-bundle image set.
 * The pinned COLMAP local adjuster admits valid points with a global track length
 * greater than one; it does not require three observations inside the local images.
 * This Android-bounded policy keeps two-observation local tracks while excluding
 * single-residual points that cannot constrain a local refinement.
 * Source: COLMAP c5f9cefc87e5dd596b638e4cee0ff543c7d14755,
 * IncrementalMapper::AdjustLocalBundle and DefaultBundleAdjuster::AddImageWithTrivialFrame.
 */
internal object ColmapLocalBundleTrackSelector {
    const val MIN_LOCAL_OBSERVATION_COUNT = 2
    const val MIN_LOCAL_BUNDLE_FRAMES = 2
    const val MIN_LOCAL_BUNDLE_TRACKS = 20
    const val MIN_LOCAL_BUNDLE_OBSERVATIONS = 60

    data class BundleSelection(
        val frameIds: Set<Int>,
        val tracks: List<CameraFeatureTrack>,
        val localObservationCount: Int
    )

    fun select(
        tracks: List<CameraFeatureTrack>,
        localFrameIds: Set<Int>
    ): List<CameraFeatureTrack> = tracks.filter { hasLocalSupport(it, localFrameIds) }

    fun selectBundle(
        tracks: List<CameraFeatureTrack>,
        localFrameIds: Set<Int>
    ): BundleSelection? {
        val selectedTracks = select(tracks, localFrameIds)
        if (!meetsBundleAdjustmentMinimum(localFrameIds, selectedTracks)) return null
        val localObservationCount = selectedTracks.sumOf { track ->
            track.observations.count { it.frameIndex in localFrameIds }
        }
        return BundleSelection(localFrameIds.toSet(), selectedTracks, localObservationCount)
    }

    fun hasLocalSupport(track: CameraFeatureTrack, localFrameIds: Set<Int>): Boolean =
        track.observations.count { it.frameIndex in localFrameIds } >= MIN_LOCAL_OBSERVATION_COUNT

    fun meetsBundleAdjustmentMinimum(
        localFrameIds: Set<Int>,
        localTracks: List<CameraFeatureTrack>
    ): Boolean {
        if (localFrameIds.size < MIN_LOCAL_BUNDLE_FRAMES ||
            localTracks.size < MIN_LOCAL_BUNDLE_TRACKS
        ) return false
        val localObservationCount = localTracks.sumOf { track ->
            track.observations.count { it.frameIndex in localFrameIds }
        }
        return localObservationCount >= MIN_LOCAL_BUNDLE_OBSERVATIONS
    }
}
