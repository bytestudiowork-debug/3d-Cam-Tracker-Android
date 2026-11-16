/*
 * Android adaptation of COLMAP IncrementalTriangulator::Retriangulate,
 * Create, and Continue in src/colmap/sfm/incremental_triangulator.cc at
 * c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
 * Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
 * See app/src/main/assets/licenses/colmap-track-completion-license.txt.
 */
package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.MultiViewTriangulationMath
import com.trackstudio.app.tracking.PinholeIntrinsics
import com.trackstudio.app.tracking.TriangulationView

internal object ColmapTrackRetriangulator {
    const val DEFAULT_MINIMUM_TRIANGULATED_RATIO = 0.2
    const val DEFAULT_MAXIMUM_ANGLE_ERROR_DEGREES = 5.0
    const val DEFAULT_CREATE_MAXIMUM_ANGLE_ERROR_DEGREES = 2.0
    const val DEFAULT_MINIMUM_PARALLAX_DEGREES = 1.5
    const val DEFAULT_IGNORE_TWO_VIEW_TRACKS = true

    data class PairMatches(
        val firstFrame: Int,
        val secondFrame: Int,
        val matches: List<Pair<Int, Int>>
    )

    data class Result(
        val tracks: List<CameraFeatureTrack>,
        val addedObservations: Int,
        val createdTracks: Int,
        val underReconstructedPairCount: Int,
        val processedPairCount: Int
    )

    private data class PointState(
        var track: CameraFeatureTrack,
        val pointWorld: DoubleArray
    )

    fun retriangulate(
        tracks: List<CameraFeatureTrack>,
        landmarks: List<CameraLandmark>,
        pairMatches: List<PairMatches>,
        poses: List<CameraPose>,
        intrinsics: PinholeIntrinsics,
        featureAt: (frameIndex: Int, featureIndex: Int) -> CameraFeatureObservation?,
        minimumTriangulatedRatio: Double = DEFAULT_MINIMUM_TRIANGULATED_RATIO,
        maximumAngleErrorDegrees: Double = DEFAULT_MAXIMUM_ANGLE_ERROR_DEGREES,
        createMaximumAngleErrorDegrees: Double = DEFAULT_CREATE_MAXIMUM_ANGLE_ERROR_DEGREES,
        minimumParallaxDegrees: Double = DEFAULT_MINIMUM_PARALLAX_DEGREES,
        ignoreTwoViewTracks: Boolean = DEFAULT_IGNORE_TWO_VIEW_TRACKS
    ): Result {
        require(minimumTriangulatedRatio.isFinite() && minimumTriangulatedRatio in 0.0..1.0)
        require(maximumAngleErrorDegrees.isFinite() && maximumAngleErrorDegrees > 0.0)
        require(createMaximumAngleErrorDegrees.isFinite() && createMaximumAngleErrorDegrees > 0.0)
        require(minimumParallaxDegrees.isFinite() && minimumParallaxDegrees >= 1e-3)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        require(intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite())
        require(tracks.map { it.id }.distinct().size == tracks.size)
        require(landmarks.map { it.trackId }.distinct().size == landmarks.size)

        if (pairMatches.isEmpty()) return Result(tracks, 0, 0, 0, 0)

        val poseByFrame = poses.asSequence()
            .filter {
                it.status != CameraPoseStatus.HELD_LAST_VALID &&
                    it.status != CameraPoseStatus.UNREGISTERED
            }
            .associateBy { it.frameIndex }
        val tracksById = LinkedHashMap<Int, CameraFeatureTrack>()
        tracks.forEach { tracksById[it.id] = it }

        val pointByTrackId = landmarks.associate { landmark ->
            landmark.trackId to doubleArrayOf(
                landmark.position.x,
                landmark.position.y,
                landmark.position.z
            )
        }
        val pointStates = LinkedHashMap<Int, PointState>()
        val anyTrackOwnerByNode = HashMap<Long, Int>()
        val pointOwnerByNode = HashMap<Long, Int>()
        for (track in tracks) {
            val point = pointByTrackId[track.id]
            if (point != null && point.all(Double::isFinite) && track.observations.size >= 2) {
                pointStates[track.id] = PointState(track, point)
            }
            for (observation in track.observations) {
                if (observation.frameIndex < 0 || observation.featureIndex < 0) continue
                val featureNode = node(observation.frameIndex, observation.featureIndex)
                anyTrackOwnerByNode.putIfAbsent(featureNode, track.id)
                if (track.id in pointStates) pointOwnerByNode.putIfAbsent(featureNode, track.id)
            }
        }

        val neighbors = HashMap<Long, MutableSet<Long>>()
        pairMatches.forEach { pair ->
            require(pair.firstFrame >= 0 && pair.secondFrame >= 0 && pair.firstFrame != pair.secondFrame)
            pair.matches.forEach matchLoop@{ (firstFeature, secondFeature) ->
                if (firstFeature < 0 || secondFeature < 0) return@matchLoop
                val firstNode = node(pair.firstFrame, firstFeature)
                val secondNode = node(pair.secondFrame, secondFeature)
                neighbors.getOrPut(firstNode) { LinkedHashSet() }.add(secondNode)
                neighbors.getOrPut(secondNode) { LinkedHashSet() }.add(firstNode)
            }
        }

        var nextTrackId: Int? = tracks.maxOfOrNull { it.id }?.let { maxId ->
            if (maxId == Int.MAX_VALUE) null else maxId + 1
        } ?: if (tracks.isEmpty()) 0 else null
        val createdTrackIds = ArrayList<Int>()
        var addedObservations = 0
        var underReconstructedPairCount = 0
        var processedPairCount = 0

        for (pair in pairMatches) {
            if (pair.matches.isEmpty()) continue
            val triangulatedMatches = pair.matches.count { (firstFeature, secondFeature) ->
                if (firstFeature < 0 || secondFeature < 0) return@count false
                val firstOwner = pointOwnerByNode[node(pair.firstFrame, firstFeature)]
                firstOwner != null && firstOwner == pointOwnerByNode[node(pair.secondFrame, secondFeature)]
            }
            val ratio = triangulatedMatches.toDouble() / pair.matches.size
            if (ratio >= minimumTriangulatedRatio) continue
            underReconstructedPairCount++

            if (pair.firstFrame !in poseByFrame || pair.secondFrame !in poseByFrame) continue
            processedPairCount++

            for ((firstFeature, secondFeature) in pair.matches) {
                if (firstFeature < 0 || secondFeature < 0) continue
                val firstNode = node(pair.firstFrame, firstFeature)
                val secondNode = node(pair.secondFrame, secondFeature)
                val firstOwner = pointOwnerByNode[firstNode]
                val secondOwner = pointOwnerByNode[secondNode]

                if (firstOwner != null && secondOwner != null) continue

                if (firstOwner != null || secondOwner != null) {
                    val pointOwner = firstOwner ?: secondOwner ?: continue
                    val unownedNode = if (firstOwner == null) firstNode else secondNode
                    if (unownedNode in anyTrackOwnerByNode) continue
                    val state = pointStates[pointOwner] ?: continue
                    val candidateFrame = frameOf(unownedNode)
                    val candidateFeature = featureOf(unownedNode)
                    if (state.track.observations.any { it.frameIndex == candidateFrame }) continue
                    val pose = poseByFrame[candidateFrame] ?: continue
                    val detected = featureAt(candidateFrame, candidateFeature)
                        ?.takeIf { it.x.isFinite() && it.y.isFinite() }
                        ?.copy(frameIndex = candidateFrame, featureIndex = candidateFeature)
                        ?: continue
                    val angleError = MultiViewTriangulationMath.angularErrorDegrees(
                        state.pointWorld,
                        TriangulationView(detected.x, detected.y, pose),
                        intrinsics
                    )
                    if (!angleError.isFinite() || angleError > maximumAngleErrorDegrees) continue

                    val updatedTrack = state.track.copy(
                        observations = (state.track.observations + detected).sortedBy { it.frameIndex }
                    )
                    state.track = updatedTrack
                    tracksById[pointOwner] = updatedTrack
                    anyTrackOwnerByNode[unownedNode] = pointOwner
                    pointOwnerByNode[unownedNode] = pointOwner
                    addedObservations++
                    continue
                }

                if (firstNode in anyTrackOwnerByNode || secondNode in anyTrackOwnerByNode) continue
                if (ignoreTwoViewTracks && isTwoViewObservation(firstNode, neighbors)) continue
                val newTrackId = nextTrackId ?: continue
                val firstPose = poseByFrame[pair.firstFrame] ?: continue
                val secondPose = poseByFrame[pair.secondFrame] ?: continue
                val firstObservation = featureAt(pair.firstFrame, firstFeature)
                    ?.takeIf { it.x.isFinite() && it.y.isFinite() }
                    ?.copy(frameIndex = pair.firstFrame, featureIndex = firstFeature)
                    ?: continue
                val secondObservation = featureAt(pair.secondFrame, secondFeature)
                    ?.takeIf { it.x.isFinite() && it.y.isFinite() }
                    ?.copy(frameIndex = pair.secondFrame, featureIndex = secondFeature)
                    ?: continue

                val estimate = MultiViewTriangulationMath.estimate(
                    observations = listOf(
                        TriangulationView(firstObservation.x, firstObservation.y, firstPose),
                        TriangulationView(secondObservation.x, secondObservation.y, secondPose)
                    ),
                    intrinsics = intrinsics,
                    minimumParallaxDegrees = minimumParallaxDegrees,
                    maximumAngularErrorDegrees = createMaximumAngleErrorDegrees,
                    minimumInlierRatio = 1.0,
                    maximumReprojectionErrorPx = null
                ) ?: continue
                if (estimate.inlierIndices.size != 2) continue

                val newTrack = CameraFeatureTrack(
                    newTrackId,
                    listOf(firstObservation, secondObservation).sortedBy { it.frameIndex }
                )
                tracksById[newTrackId] = newTrack
                createdTrackIds += newTrackId
                pointStates[newTrackId] = PointState(newTrack, estimate.pointWorld)
                anyTrackOwnerByNode[firstNode] = newTrackId
                anyTrackOwnerByNode[secondNode] = newTrackId
                pointOwnerByNode[firstNode] = newTrackId
                pointOwnerByNode[secondNode] = newTrackId
                addedObservations += 2
                nextTrackId = if (newTrackId == Int.MAX_VALUE) null else newTrackId + 1
            }
        }

        val outputTracks = tracks.map { tracksById.getValue(it.id) } +
            createdTrackIds.mapNotNull(tracksById::get)
        return Result(
            tracks = outputTracks,
            addedObservations = addedObservations,
            createdTracks = createdTrackIds.size,
            underReconstructedPairCount = underReconstructedPairCount,
            processedPairCount = processedPairCount
        )
    }

    private fun isTwoViewObservation(
        referenceNode: Long,
        neighbors: Map<Long, Set<Long>>
    ): Boolean {
        val referenceNeighbors = neighbors[referenceNode].orEmpty()
        if (referenceNeighbors.size != 1) return false
        val otherNode = referenceNeighbors.first()
        return neighbors[otherNode].orEmpty().size == 1
    }

    private fun node(frame: Int, feature: Int): Long =
        (frame.toLong() shl 32) or (feature.toLong() and 0xffffffffL)

    private fun frameOf(node: Long): Int = (node ushr 32).toInt()

    private fun featureOf(node: Long): Int = node.toInt()
}
