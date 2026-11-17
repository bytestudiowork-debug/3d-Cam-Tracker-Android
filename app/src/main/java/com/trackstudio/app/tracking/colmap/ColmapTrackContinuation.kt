/*
 * Android adaptation of COLMAP IncrementalTriangulator::TriangulateImage,
 * Find, and Continue in src/colmap/sfm/incremental_triangulator.cc at commit
 * c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
 * Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
 * See app/src/main/assets/licenses/colmap-triangulation-license.txt.
 */
package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.MultiViewTriangulationMath
import com.trackstudio.app.tracking.PinholeIntrinsics
import com.trackstudio.app.tracking.TriangulationView

/** Runs Continue on unassociated observations before residual tracks reach Create. */
internal object ColmapTrackContinuation {
    const val DEFAULT_MAXIMUM_TRANSITIVITY = 1
    const val DEFAULT_MAXIMUM_ANGULAR_ERROR_DEGREES = 2.0

    data class PairMatches(
        val firstFrame: Int,
        val secondFrame: Int,
        val matches: List<Pair<Int, Int>>
    )

    data class Result(
        val tracks: List<CameraFeatureTrack>,
        val addedObservations: Int
    )

    private data class Candidate(
        val trackId: Int,
        val angularErrorDegrees: Double
    )

    fun continueTracks(
        tracks: List<CameraFeatureTrack>,
        pointsByTrackId: Map<Int, DoubleArray>,
        pairMatches: List<PairMatches>,
        poses: List<CameraPose>,
        intrinsics: PinholeIntrinsics,
        maximumTransitivity: Int = DEFAULT_MAXIMUM_TRANSITIVITY,
        maximumAngularErrorDegrees: Double = DEFAULT_MAXIMUM_ANGULAR_ERROR_DEGREES
    ): Result {
        require(maximumTransitivity >= 0)
        require(maximumAngularErrorDegrees.isFinite() && maximumAngularErrorDegrees > 0.0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        require(intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite())

        if (tracks.isEmpty() || pointsByTrackId.isEmpty() || pairMatches.isEmpty() ||
            maximumTransitivity == 0
        ) return Result(tracks, 0)

        val tracksById = LinkedHashMap<Int, CameraFeatureTrack>(tracks.size)
        tracks.forEach { tracksById[it.id] = it }
        val pointByTrackId = pointsByTrackId.filter { (trackId, point) ->
            trackId in tracksById && point.size == 3 && point.all { it.isFinite() }
        }
        if (pointByTrackId.isEmpty()) return Result(tracks, 0)

        val poseByFrame = poses.asSequence()
            .filter {
                it.status != CameraPoseStatus.HELD_LAST_VALID &&
                    it.status != CameraPoseStatus.UNREGISTERED
            }
            .associateBy { it.frameIndex }
        if (poseByFrame.isEmpty()) return Result(tracks, 0)

        val adjacency = HashMap<Long, MutableSet<Long>>()
        for (pair in pairMatches) {
            require(pair.firstFrame >= 0 && pair.secondFrame >= 0 && pair.firstFrame != pair.secondFrame)
            for ((firstFeature, secondFeature) in pair.matches) {
                if (firstFeature < 0 || secondFeature < 0) continue
                val firstNode = node(pair.firstFrame, firstFeature)
                val secondNode = node(pair.secondFrame, secondFeature)
                adjacency.getOrPut(firstNode) { LinkedHashSet() }.add(secondNode)
                adjacency.getOrPut(secondNode) { LinkedHashSet() }.add(firstNode)
            }
        }
        if (adjacency.isEmpty()) return Result(tracks, 0)

        val ownerByNode = HashMap<Long, Int>()
        val ambiguousNodes = HashSet<Long>()
        for (track in tracks) {
            if (track.id !in pointByTrackId) continue
            for (observation in track.observations) {
                if (observation.featureIndex < 0 || observation.frameIndex !in poseByFrame) continue
                val key = node(observation.frameIndex, observation.featureIndex)
                if (key in ambiguousNodes) continue
                val oldOwner = ownerByNode.putIfAbsent(key, track.id)
                if (oldOwner != null && oldOwner != track.id) {
                    ownerByNode.remove(key)
                    ambiguousNodes += key
                }
            }
        }

        var addedObservations = 0
        for (sourceTrack in tracks) {
            if (sourceTrack.id in pointByTrackId) continue
            val residual = ArrayList<CameraFeatureObservation>(sourceTrack.observations.size)
            for (observation in sourceTrack.observations) {
                val pose = poseByFrame[observation.frameIndex]
                if (observation.featureIndex < 0 || pose == null ||
                    !observation.x.isFinite() || !observation.y.isFinite()
                ) {
                    residual += observation
                    continue
                }

                val observationNode = node(observation.frameIndex, observation.featureIndex)
                if (observationNode in ambiguousNodes) {
                    residual += observation
                    continue
                }
                if (observationNode in ownerByNode) continue

                var bestCandidate: Candidate? = null
                for (candidateNode in findTransitiveNodes(
                    observationNode, adjacency, maximumTransitivity
                )) {
                    if (candidateNode in ambiguousNodes) continue
                    val candidateFrame = frameOf(candidateNode)
                    if (candidateFrame !in poseByFrame) continue
                    val candidateTrackId = ownerByNode[candidateNode] ?: continue
                    val targetTrack = tracksById[candidateTrackId] ?: continue
                    if (targetTrack.observations.any { it.frameIndex == observation.frameIndex }) continue
                    val point = pointByTrackId[candidateTrackId] ?: continue
                    val angularError = MultiViewTriangulationMath.angularErrorDegrees(
                        point,
                        TriangulationView(observation.x, observation.y, pose),
                        intrinsics
                    )
                    if (!angularError.isFinite()) continue
                    val best = bestCandidate
                    if (best == null || angularError < best.angularErrorDegrees) {
                        bestCandidate = Candidate(candidateTrackId, angularError)
                    }
                }

                val best = bestCandidate
                if (best == null || best.angularErrorDegrees > maximumAngularErrorDegrees) {
                    residual += observation
                    continue
                }
                val targetTrack = tracksById[best.trackId]
                if (targetTrack == null) {
                    residual += observation
                    continue
                }
                tracksById[best.trackId] = targetTrack.copy(
                    observations = (targetTrack.observations + observation)
                        .sortedWith(compareBy<CameraFeatureObservation> { it.frameIndex }
                            .thenBy { it.featureIndex })
                )
                ownerByNode[observationNode] = best.trackId
                addedObservations++
            }

            if (residual.size == sourceTrack.observations.size) continue
            if (residual.isEmpty()) {
                tracksById.remove(sourceTrack.id)
            } else {
                tracksById[sourceTrack.id] = sourceTrack.copy(observations = residual)
            }
        }

        return Result(
            tracks = tracks.mapNotNull { tracksById[it.id] },
            addedObservations = addedObservations
        )
    }

    private fun findTransitiveNodes(
        seed: Long,
        adjacency: Map<Long, Set<Long>>,
        maximumTransitivity: Int
    ): List<Long> {
        val visited = HashSet<Long>()
        visited += seed
        var frontier = listOf(seed)
        repeat(maximumTransitivity) {
            val next = linkedSetOf<Long>()
            frontier.sorted().forEach { current ->
                adjacency[current].orEmpty().sorted().forEach { neighbor ->
                    if (visited.add(neighbor)) next += neighbor
                }
            }
            if (next.isEmpty()) {
                return visited.asSequence().filter { it != seed }.sorted().toList()
            }
            frontier = next.toList()
        }
        return visited.asSequence().filter { it != seed }.sorted().toList()
    }

    private fun node(frame: Int, feature: Int): Long =
        (frame.toLong() shl 32) or (feature.toLong() and 0xffffffffL)

    private fun frameOf(node: Long): Int = (node ushr 32).toInt()
}
