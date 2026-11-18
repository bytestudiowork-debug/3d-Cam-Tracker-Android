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

    fun select(
        tracks: List<CameraFeatureTrack>,
        localFrameIds: Set<Int>
    ): List<CameraFeatureTrack> = tracks.filter { hasLocalSupport(it, localFrameIds) }

    fun hasLocalSupport(track: CameraFeatureTrack, localFrameIds: Set<Int>): Boolean =
        track.observations.count { it.frameIndex in localFrameIds } >= MIN_LOCAL_OBSERVATION_COUNT
}
