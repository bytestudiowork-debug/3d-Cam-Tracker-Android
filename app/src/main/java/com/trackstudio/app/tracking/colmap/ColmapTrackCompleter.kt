/*
 * Adapted from COLMAP's IncrementalTriangulator::Complete in
 * src/colmap/sfm/incremental_triangulator.cc at commit
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
import com.trackstudio.app.tracking.CameraProjectionMath
import com.trackstudio.app.tracking.PinholeIntrinsics

/** COLMAP IncrementalTriangulator::Complete over the accepted in-memory view graph. */
internal object ColmapTrackCompleter {
    const val DEFAULT_MAX_REPROJECTION_ERROR_PX = 15.0
    const val DEFAULT_MAX_TRANSITIVITY = 5
    const val DEFAULT_MIN_PAIR_MATCHES = 15

    data class PairMatches(
        val firstFrame: Int,
        val secondFrame: Int,
        val matches: List<Pair<Int, Int>>
    )

    data class Result(
        val tracks: List<CameraFeatureTrack>,
        val addedObservations: Int
    )

    /**
     * Adds unassigned, verified feature observations to existing 3D tracks using
     * COLMAP's breadth-first correspondence traversal and squared reprojection test.
     *
     * The app has one calibrated pinhole camera per frame and already-filtered
     * inlier matches, so camera-model validation and database-cache loading are
     * handled at the engine boundary. Frames remain limited to the solved graph.
     */
    fun completeTracks(
        tracks: List<CameraFeatureTrack>,
        landmarks: List<CameraLandmark>,
        pairMatches: List<PairMatches>,
        activeFrames: Set<Int>,
        poses: List<CameraPose>,
        intrinsics: PinholeIntrinsics,
        featureAt: (frameIndex: Int, featureIndex: Int) -> CameraFeatureObservation?,
        maximumReprojectionErrorPx: Double = DEFAULT_MAX_REPROJECTION_ERROR_PX,
        maximumTransitivity: Int = DEFAULT_MAX_TRANSITIVITY,
        minimumPairMatches: Int = DEFAULT_MIN_PAIR_MATCHES
    ): Result {
        require(maximumReprojectionErrorPx.isFinite() && maximumReprojectionErrorPx > 0.0)
        require(maximumTransitivity >= 0)
        require(minimumPairMatches >= 0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        require(intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite())

        if (tracks.isEmpty() || landmarks.isEmpty() || maximumTransitivity == 0) {
            return Result(tracks, 0)
        }

        val poseByFrame = poses.asSequence()
            .filter { it.status != CameraPoseStatus.HELD_LAST_VALID && it.status != CameraPoseStatus.UNREGISTERED }
            .associateBy { it.frameIndex }
        val pointByTrackId = landmarks.associate { landmark ->
            landmark.trackId to doubleArrayOf(
                landmark.position.x,
                landmark.position.y,
                landmark.position.z
            )
        }
        val activePoseFrames = activeFrames.intersect(poseByFrame.keys)
        if (pointByTrackId.isEmpty() || activePoseFrames.isEmpty()) return Result(tracks, 0)

        val adjacency = HashMap<Long, MutableSet<Long>>()
        for (pair in pairMatches) {
            if (pair.matches.size < minimumPairMatches) continue
            require(pair.firstFrame >= 0 && pair.secondFrame >= 0 && pair.firstFrame != pair.secondFrame)
            for ((firstFeature, secondFeature) in pair.matches) {
                if (firstFeature < 0 || secondFeature < 0) continue
                val first = node(pair.firstFrame, firstFeature)
                val second = node(pair.secondFrame, secondFeature)
                adjacency.getOrPut(first) { LinkedHashSet() }.add(second)
                adjacency.getOrPut(second) { LinkedHashSet() }.add(first)
            }
        }
        if (adjacency.isEmpty()) return Result(tracks, 0)
        val orderedAdjacency = adjacency.mapValues { (_, neighbors) -> neighbors.sorted() }

        // A COLMAP observation with an existing 3D association cannot be reused
        // by another point. Keep the same ownership rule in this in-memory graph.
        val ownerByNode = HashMap<Long, Int>()
        tracks.asSequence()
            .filter { it.id in pointByTrackId }
            .sortedBy { it.id }
            .forEach { track ->
                track.observations.forEach { observation ->
                    if (observation.frameIndex >= 0 && observation.featureIndex >= 0) {
                        val featureNode = node(observation.frameIndex, observation.featureIndex)
                        if (featureNode !in ownerByNode) ownerByNode[featureNode] = track.id
                    }
                }
            }

        var addedTotal = 0
        val completedTracks = tracks.map { track ->
            val point = pointByTrackId[track.id]
                ?: return@map track
            if (point.any { !it.isFinite() }) return@map track

            val observations = track.observations.toMutableList()
            val framesInTrack = observations.mapTo(HashSet()) { it.frameIndex }
            var currentQueue = ArrayDeque<Long>()
            observations.asSequence()
                .filter { it.frameIndex in activePoseFrames && it.featureIndex >= 0 }
                .map { node(it.frameIndex, it.featureIndex) }
                .distinct()
                .forEach(currentQueue::addLast)
            if (currentQueue.isEmpty()) return@map track

            var trackAdded = 0
            val maximumSquaredError = maximumReprojectionErrorPx * maximumReprojectionErrorPx
            for (transitivity in 1..maximumTransitivity) {
                val nextQueue = ArrayDeque<Long>()
                while (currentQueue.isNotEmpty()) {
                    val currentNode = currentQueue.removeLast()
                    for (candidateNode in orderedAdjacency[currentNode].orEmpty()) {
                        val frame = frameOf(candidateNode)
                        val feature = featureOf(candidateNode)
                        if (frame !in activePoseFrames || frame in framesInTrack) continue
                        val owner = ownerByNode[candidateNode]
                        if (owner != null) continue
                        val pose = poseByFrame[frame] ?: continue
                        val detected = featureAt(frame, feature) ?: continue
                        if (!detected.x.isFinite() || !detected.y.isFinite()) continue
                        val observation = detected.copy(frameIndex = frame, featureIndex = feature)
                        val errorSquared = squaredReprojectionError(
                            pointWorld = point,
                            pose = pose,
                            observation = observation,
                            intrinsics = intrinsics
                        )
                        if (!errorSquared.isFinite() || errorSquared > maximumSquaredError) continue

                        observations += observation
                        framesInTrack += frame
                        ownerByNode[candidateNode] = track.id
                        trackAdded++
                        if (transitivity < maximumTransitivity) nextQueue.addLast(candidateNode)
                    }
                }
                if (nextQueue.isEmpty()) break
                currentQueue = nextQueue
            }

            if (trackAdded == 0) {
                track
            } else {
                addedTotal += trackAdded
                track.copy(observations = observations.sortedBy { it.frameIndex })
            }
        }
        return Result(completedTracks, addedTotal)
    }

    private fun squaredReprojectionError(
        pointWorld: DoubleArray,
        pose: CameraPose,
        observation: CameraFeatureObservation,
        intrinsics: PinholeIntrinsics
    ): Double {
        val rotation = pose.rotation
        val translation = pose.translation
        if (rotation.size != 9 || translation.size != 3 ||
            rotation.any { !it.isFinite() } || translation.any { !it.isFinite() }
        ) return Double.POSITIVE_INFINITY
        val cameraX = rotation[0] * pointWorld[0] + rotation[1] * pointWorld[1] +
            rotation[2] * pointWorld[2] + translation[0]
        val cameraY = rotation[3] * pointWorld[0] + rotation[4] * pointWorld[1] +
            rotation[5] * pointWorld[2] + translation[1]
        val cameraZ = rotation[6] * pointWorld[0] + rotation[7] * pointWorld[1] +
            rotation[8] * pointWorld[2] + translation[2]
        val reprojectionError = CameraProjectionMath.reprojectionError(
            cameraX = cameraX,
            cameraY = cameraY,
            cameraZ = cameraZ,
            observedX = observation.x,
            observedY = observation.y,
            intrinsics = intrinsics
        ) ?: return Double.POSITIVE_INFINITY
        return reprojectionError * reprojectionError
    }

    private fun node(frame: Int, feature: Int): Long =
        (frame.toLong() shl 32) or (feature.toLong() and 0xffffffffL)

    private fun frameOf(node: Long): Int = (node ushr 32).toInt()

    private fun featureOf(node: Long): Int = node.toInt()
}
