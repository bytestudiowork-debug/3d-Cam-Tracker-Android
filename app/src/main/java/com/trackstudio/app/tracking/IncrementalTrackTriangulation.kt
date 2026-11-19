package com.trackstudio.app.tracking

import com.trackstudio.app.tracking.colmap.ColmapTrackRetriangulator

internal data class IncrementalTrackTriangulationResult(
    val triangulation: TriangulationResult,
    val changedTrackIds: Set<Int>,
    val addedObservations: Int,
    val createdTrackCount: Int
)

internal object IncrementalTrackTriangulation {
    fun finalize(
        featureTracks: List<CameraFeatureTrack>,
        landmarks: List<CameraLandmark>,
        pairMatches: List<ColmapTrackRetriangulator.PairMatches>,
        poses: List<CameraPose>,
        intrinsics: PinholeIntrinsics,
        featureAt: (frameIndex: Int, featureIndex: Int) -> CameraFeatureObservation?
    ): IncrementalTrackTriangulationResult {
        val tracksById = featureTracks.associateBy { it.id }
        require(tracksById.size == featureTracks.size)

        val retriangulation = ColmapTrackRetriangulator.retriangulate(
            tracks = featureTracks,
            landmarks = landmarks,
            pairMatches = pairMatches,
            poses = poses,
            intrinsics = intrinsics,
            featureAt = featureAt
        )
        val changedTrackIds = retriangulation.tracks.asSequence()
            .filter { tracksById[it.id] != it }
            .mapTo(linkedSetOf()) { it.id }
        val triangulation = SfMTriangulator.triangulate(
            featureTracks = retriangulation.tracks,
            poses = poses,
            intrinsics = intrinsics
        )
        return IncrementalTrackTriangulationResult(
            triangulation = triangulation,
            changedTrackIds = changedTrackIds,
            addedObservations = retriangulation.addedObservations,
            createdTrackCount = retriangulation.createdTracks
        )
    }
}
