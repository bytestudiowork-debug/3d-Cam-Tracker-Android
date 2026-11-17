/*
 * Android adaptation of COLMAP's IncrementalTriangulator::Merge from
 * src/colmap/sfm/incremental_triangulator.cc at commit
 * b6b7b54eca6078070f73a3f0a084f79c629a6f10.
 * Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
 * See app/src/main/assets/licenses/colmap-track-completion-license.txt.
 */
package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.PinholeIntrinsics
import org.opencv.core.Point3
import java.util.ArrayDeque

internal object ColmapTrackMerger {
    const val DEFAULT_MAX_REPROJECTION_ERROR_PX = 4.0

    data class PairMatches(
        val firstFrame: Int,
        val secondFrame: Int,
        val matches: List<Pair<Int, Int>>
    )

    data class Result(
        val tracks: List<CameraFeatureTrack>,
        val landmarks: List<CameraLandmark>,
        val mergedLandmarks: List<CameraLandmark>,
        val removedTrackIds: Set<Int>,
        val mergedTrackCount: Int,
        val mergedObservationCount: Int = 0
    )

    private data class PointState(
        val track: CameraFeatureTrack,
        val landmark: CameraLandmark,
        val point: DoubleArray,
        val mergeToken: Long,
        val wasMerged: Boolean
    )

    fun mergeTracks(
        tracks: List<CameraFeatureTrack>,
        landmarks: List<CameraLandmark>,
        pairMatches: List<PairMatches>,
        poses: List<CameraPose>,
        intrinsics: PinholeIntrinsics,
        maximumReprojectionErrorPx: Double = DEFAULT_MAX_REPROJECTION_ERROR_PX,
        candidateTrackIds: Set<Int>? = null
    ): Result {
        require(maximumReprojectionErrorPx.isFinite() && maximumReprojectionErrorPx > 0.0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        require(intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite())
        require(tracks.map { it.id }.distinct().size == tracks.size)
        require(landmarks.map { it.trackId }.distinct().size == landmarks.size)

        if (tracks.isEmpty() || landmarks.isEmpty() || pairMatches.isEmpty()) {
            return Result(tracks, landmarks, emptyList(), emptySet(), 0)
        }

        val poseByFrame = poses.asSequence()
            .filter { it.status != CameraPoseStatus.HELD_LAST_VALID && it.status != CameraPoseStatus.UNREGISTERED }
            .associateBy { it.frameIndex }
        val trackById = tracks.associateBy { it.id }
        val states = linkedMapOf<Int, PointState>()
        var nextMergeToken = 0L
        for (landmark in landmarks) {
            val track = trackById[landmark.trackId] ?: continue
            val observations = track.observations
            if (observations.size < 2 || observations.any { observation ->
                    observation.frameIndex < 0 || observation.featureIndex < 0 ||
                        !observation.x.isFinite() || !observation.y.isFinite()
                } || observations.map { it.frameIndex }.distinct().size != observations.size
            ) continue
            val point = doubleArrayOf(landmark.position.x, landmark.position.y, landmark.position.z)
            if (point.any { !it.isFinite() }) continue
            states[track.id] = PointState(track, landmark, point, nextMergeToken++, wasMerged = false)
        }
        if (states.size < 2) return Result(tracks, landmarks, emptyList(), emptySet(), 0)

        val neighbors = HashMap<Long, MutableSet<Long>>()
        for (pair in pairMatches) {
            require(pair.firstFrame >= 0 && pair.secondFrame >= 0 && pair.firstFrame != pair.secondFrame)
            for ((firstFeature, secondFeature) in pair.matches) {
                if (firstFeature < 0 || secondFeature < 0) continue
                val first = node(pair.firstFrame, firstFeature)
                val second = node(pair.secondFrame, secondFeature)
                neighbors.getOrPut(first) { HashSet() }.add(second)
                neighbors.getOrPut(second) { HashSet() }.add(first)
            }
        }
        if (neighbors.isEmpty()) return Result(tracks, landmarks, emptyList(), emptySet(), 0)

        val ownerByNode = HashMap<Long, Int>()
        for (state in states.values) {
            for (observation in state.track.observations) {
                val key = node(observation.frameIndex, observation.featureIndex)
                val previous = ownerByNode.putIfAbsent(key, state.track.id)
                require(previous == null || previous == state.track.id) {
                    "A feature observation cannot belong to multiple triangulated tracks."
                }
            }
        }

        val orderedNeighbors = neighbors.mapValues { (_, values) -> values.sorted() }
        val pending = ArrayDeque<Int>()
        states.keys.asSequence()
            .filter { candidateTrackIds == null || it in candidateTrackIds }
            .sorted()
            .forEach(pending::addLast)
        val triedPairs = HashSet<Pair<Long, Long>>()
        val removedTrackIds = linkedSetOf<Int>()
        var mergedTrackCount = 0
        val maximumSquaredError = maximumReprojectionErrorPx * maximumReprojectionErrorPx

        while (pending.isNotEmpty()) {
            val sourceId = pending.removeFirst()
            val source = states[sourceId] ?: continue
            val candidateIds = source.track.observations.asSequence()
                .flatMap { observation ->
                    orderedNeighbors[node(observation.frameIndex, observation.featureIndex)].orEmpty().asSequence()
                }
                .mapNotNull(ownerByNode::get)
                .filter { it != sourceId && it in states }
                .distinct()
                .sorted()
                .toList()

            for (candidateId in candidateIds) {
                val currentSource = states[sourceId] ?: break
                val candidate = states[candidateId] ?: continue
                val key = pairKey(currentSource.mergeToken, candidate.mergeToken)
                if (!triedPairs.add(key)) continue

                val firstFrames = currentSource.track.observations.mapTo(HashSet()) { it.frameIndex }
                if (candidate.track.observations.any { it.frameIndex in firstFrames }) continue

                val totalLength = currentSource.track.length.toDouble() + candidate.track.length.toDouble()
                if (!totalLength.isFinite() || totalLength <= 0.0) continue
                val point = DoubleArray(3) { axis ->
                    (currentSource.track.length * currentSource.point[axis] +
                        candidate.track.length * candidate.point[axis]) / totalLength
                }
                if (point.any { !it.isFinite() }) continue

                val mergedObservations = (currentSource.track.observations + candidate.track.observations)
                    .sortedWith(compareBy<CameraFeatureObservation>({ it.frameIndex }, { it.featureIndex }))
                val reprojectionErrors = reprojectionErrors(
                    point,
                    mergedObservations,
                    poseByFrame,
                    intrinsics,
                    maximumSquaredError
                ) ?: continue

                val survivingId = sourceId
                val removedId = candidateId
                val mergedTrack = CameraFeatureTrack(survivingId, mergedObservations)
                val mergedLandmark = CameraLandmark(
                    trackId = survivingId,
                    position = Point3(point[0], point[1], point[2]),
                    observationCount = mergedObservations.size,
                    meanReprojectionErrorPx = reprojectionErrors.average(),
                    triangulationAngleDegrees = maxOf(
                        currentSource.landmark.triangulationAngleDegrees,
                        candidate.landmark.triangulationAngleDegrees
                    )
                )

                states.remove(sourceId)
                states.remove(candidateId)
                states[survivingId] = PointState(
                    mergedTrack,
                    mergedLandmark,
                    point,
                    nextMergeToken++,
                    wasMerged = true
                )
                removedTrackIds += removedId
                for (observation in mergedObservations) {
                    ownerByNode[node(observation.frameIndex, observation.featureIndex)] = survivingId
                }
                mergedTrackCount++
                pending.addLast(survivingId)
                break
            }
        }

        val outputTracks = tracks.mapNotNull { track ->
            if (track.id in removedTrackIds) null else states[track.id]?.track ?: track
        }
        val outputLandmarks = landmarks.mapNotNull { landmark ->
            if (landmark.trackId in removedTrackIds) null else states[landmark.trackId]?.landmark ?: landmark
        }
        val mergedLandmarks = states.values.asSequence()
            .filter { it.wasMerged }
            .map { it.landmark }
            .sortedBy { it.trackId }
            .toList()
        val mergedObservationCount = states.values.asSequence()
            .filter { it.wasMerged }
            .sumOf { it.track.length }

        return Result(
            tracks = outputTracks,
            landmarks = outputLandmarks,
            mergedLandmarks = mergedLandmarks,
            removedTrackIds = removedTrackIds,
            mergedTrackCount = mergedTrackCount,
            mergedObservationCount = mergedObservationCount
        )
    }

    private fun reprojectionErrors(
        pointWorld: DoubleArray,
        observations: List<CameraFeatureObservation>,
        poseByFrame: Map<Int, CameraPose>,
        intrinsics: PinholeIntrinsics,
        maximumSquaredError: Double
    ): List<Double>? {
        val errors = ArrayList<Double>(observations.size)
        for (observation in observations) {
            val pose = poseByFrame[observation.frameIndex] ?: return null
            val rotation = pose.rotation
            val translation = pose.translation
            if (rotation.size != 9 || translation.size != 3 ||
                rotation.any { !it.isFinite() } || translation.any { !it.isFinite() }
            ) return null

            val cameraX = rotation[0] * pointWorld[0] + rotation[1] * pointWorld[1] +
                rotation[2] * pointWorld[2] + translation[0]
            val cameraY = rotation[3] * pointWorld[0] + rotation[4] * pointWorld[1] +
                rotation[5] * pointWorld[2] + translation[1]
            val cameraZ = rotation[6] * pointWorld[0] + rotation[7] * pointWorld[1] +
                rotation[8] * pointWorld[2] + translation[2]
            if (!cameraX.isFinite() || !cameraY.isFinite() || !cameraZ.isFinite() || cameraZ <= 1e-6) {
                return null
            }

            val dx = (intrinsics.fx * cameraX + intrinsics.skew * cameraY) / cameraZ +
                intrinsics.cx - observation.x
            val dy = intrinsics.fy * cameraY / cameraZ + intrinsics.cy - observation.y
            val squaredError = dx * dx + dy * dy
            if (!squaredError.isFinite() || squaredError > maximumSquaredError) return null
            errors += kotlin.math.sqrt(squaredError)
        }
        return errors.takeIf { it.isNotEmpty() }
    }

    private fun node(frame: Int, feature: Int): Long =
        (frame.toLong() shl 32) or (feature.toLong() and 0xffffffffL)

    private fun pairKey(first: Long, second: Long): Pair<Long, Long> {
        val a = minOf(first, second)
        val b = maxOf(first, second)
        return a to b
    }
}
