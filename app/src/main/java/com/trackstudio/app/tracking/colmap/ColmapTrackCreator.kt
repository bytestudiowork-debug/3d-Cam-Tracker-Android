/*
 * Android adaptation of COLMAP IncrementalTriangulator::Create in
 * src/colmap/sfm/incremental_triangulator.cc at commit
 * c5f9cefc87e5dd596b638e4cee0ff543c7d14755.
 * Copyright (c), ETH Zurich and UNC Chapel Hill. All rights reserved.
 * See app/src/main/assets/licenses/colmap-triangulation-license.txt.
 */
package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.CameraProjectionMath
import com.trackstudio.app.tracking.MultiViewTriangulationMath
import com.trackstudio.app.tracking.PinholeIntrinsics
import com.trackstudio.app.tracking.TriangulationStats
import com.trackstudio.app.tracking.TriangulationView
import org.opencv.core.Point3
import kotlin.math.sqrt

/** COLMAP Create with bounded Android hypotheses and recursive residual-track retry. */
internal object ColmapTrackCreator {
    const val DEFAULT_MAXIMUM_ANGULAR_ERROR_DEGREES = 2.0
    const val DEFAULT_MINIMUM_TRIANGULATION_ANGLE_DEGREES = 1.5
    const val MINIMUM_RECURSIVE_RESIDUAL_OBSERVATIONS = 3

    data class Result(
        val tracks: List<CameraFeatureTrack>,
        val landmarks: List<CameraLandmark>,
        val stats: TriangulationStats,
        val additionalTrackCount: Int,
        val unassignedTracks: List<CameraFeatureTrack> = emptyList()
    )

    fun create(
        tracks: List<CameraFeatureTrack>,
        poses: List<CameraPose>,
        intrinsics: PinholeIntrinsics,
        maximumAngularErrorDegrees: Double = DEFAULT_MAXIMUM_ANGULAR_ERROR_DEGREES,
        minimumTriangulationAngleDegrees: Double = DEFAULT_MINIMUM_TRIANGULATION_ANGLE_DEGREES,
        onProgress: (Int, Int) -> Unit = { _, _ -> }
    ): Result {
        require(maximumAngularErrorDegrees.isFinite() && maximumAngularErrorDegrees > 0.0)
        require(minimumTriangulationAngleDegrees.isFinite() && minimumTriangulationAngleDegrees > 0.0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        require(intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite())

        val poseByFrame = poses.asSequence()
            .filter {
                it.status != CameraPoseStatus.HELD_LAST_VALID &&
                    it.status != CameraPoseStatus.UNREGISTERED
            }
            .associateBy { it.frameIndex }
        val outputTracks = ArrayList<CameraFeatureTrack>()
        val outputUnassignedTracks = ArrayList<CameraFeatureTrack>()
        val outputLandmarks = ArrayList<CameraLandmark>()
        var rejectedShort = 0
        var rejectedMissingPose = 0
        var rejectedGeometry = 0
        var rejectedProjection = 0
        var additionalTrackCount = 0
        var nextTrackId = (tracks.maxOfOrNull { it.id }?.toLong() ?: -1L) + 1L
        fun allocateTrackId(): Int? {
            if (nextTrackId > Int.MAX_VALUE.toLong()) return null
            return nextTrackId.toInt().also { nextTrackId++ }
        }

        for ((trackIndex, track) in tracks.withIndex()) {
            if (trackIndex % 16 == 0 || trackIndex == tracks.lastIndex) {
                onProgress(trackIndex + 1, tracks.size)
            }
            if (track.observations.size < 2) {
                rejectedShort++
                outputUnassignedTracks += track
                continue
            }

            var pending = track.observations.mapNotNull { observation ->
                poseByFrame[observation.frameIndex]?.let { observation to it }
            }
            if (pending.size < 2) {
                rejectedMissingPose++
                outputUnassignedTracks += track
                continue
            }
            if (pending.map { it.first.frameIndex }.distinct().size != pending.size) {
                rejectedGeometry++
                outputUnassignedTracks += track
                continue
            }

            var componentCount = 0
            val assignedObservations = HashSet<CameraFeatureObservation>()
            while (pending.size >= 2) {
                val estimate = MultiViewTriangulationMath.estimate(
                    observations = pending.map { (observation, pose) ->
                        TriangulationView(observation.x, observation.y, pose)
                    },
                    intrinsics = intrinsics,
                    minimumParallaxDegrees = minimumTriangulationAngleDegrees,
                    maximumAngularErrorDegrees = maximumAngularErrorDegrees,
                    minimumInlierRatio = 0.0,
                    maximumReprojectionErrorPx = null
                ) ?: break

                val inlierIndices = estimate.inlierIndices.asSequence()
                    .filter { it in pending.indices }
                    .distinct()
                    .sorted()
                    .toList()
                if (inlierIndices.size < 2 || estimate.pointWorld.size != 3 ||
                    estimate.pointWorld.any { !it.isFinite() }
                ) break

                val inlierObservations = inlierIndices.map { pending[it] }
                val errors = inlierObservations.mapNotNull { (observation, pose) ->
                    reprojectionError(estimate.pointWorld, pose, observation, intrinsics)
                }
                if (errors.size < 2) {
                    rejectedProjection++
                    break
                }

                val trackId = if (componentCount == 0) {
                    track.id
                } else {
                    allocateTrackId() ?: break
                }
                val componentTrack = CameraFeatureTrack(
                    trackId,
                    inlierObservations.map { it.first }.sortedBy { it.frameIndex }
                )
                outputTracks += componentTrack
                outputLandmarks += CameraLandmark(
                    trackId = trackId,
                    position = Point3(
                        estimate.pointWorld[0], estimate.pointWorld[1], estimate.pointWorld[2]
                    ),
                    observationCount = errors.size,
                    meanReprojectionErrorPx = errors.average(),
                    triangulationAngleDegrees = estimate.triangulationAngleDegrees
                )
                if (componentCount > 0) additionalTrackCount++
                componentCount++
                assignedObservations.addAll(inlierObservations.map { it.first })

                val inlierSet = inlierIndices.toHashSet()
                val residual = pending.filterIndexed { index, _ -> index !in inlierSet }
                pending = residual
                if (residual.size < MINIMUM_RECURSIVE_RESIDUAL_OBSERVATIONS) break
            }
            if (componentCount == 0) {
                rejectedGeometry++
            }
            val unassignedObservations = track.observations.filterNot { it in assignedObservations }
            if (unassignedObservations.isNotEmpty()) {
                val unassignedTrackId = if (componentCount == 0) track.id else allocateTrackId()
                if (unassignedTrackId != null) {
                    outputUnassignedTracks += CameraFeatureTrack(
                        unassignedTrackId,
                        unassignedObservations.sortedBy { it.frameIndex }
                    )
                }
            }
        }

        val stats = TriangulationStats(
            inputTracks = tracks.size,
            triangulatedLandmarks = outputLandmarks.size,
            rejectedShortTracks = rejectedShort,
            rejectedMissingPoses = rejectedMissingPose,
            rejectedInvalidGeometry = rejectedGeometry,
            rejectedReprojection = rejectedProjection,
            meanReprojectionErrorPx = outputLandmarks.map { it.meanReprojectionErrorPx }
                .takeIf { it.isNotEmpty() }?.average() ?: 0.0,
            meanTriangulationAngleDegrees = outputLandmarks.map { it.triangulationAngleDegrees }
                .takeIf { it.isNotEmpty() }?.average() ?: 0.0
        )
        return Result(
            outputTracks,
            outputLandmarks,
            stats,
            additionalTrackCount,
            outputUnassignedTracks
        )
    }

    fun updatePointEstimates(
        existingPointsByTrackId: Map<Int, DoubleArray>,
        retriangulatedLandmarks: List<CameraLandmark>
    ): Map<Int, DoubleArray> {
        val updatedPoints = existingPointsByTrackId.toMutableMap()
        retriangulatedLandmarks.forEach { landmark ->
            val point = doubleArrayOf(
                landmark.position.x,
                landmark.position.y,
                landmark.position.z
            )
            if (point.all(Double::isFinite)) updatedPoints[landmark.trackId] = point
        }
        return updatedPoints
    }

    private fun reprojectionError(
        pointWorld: DoubleArray,
        pose: CameraPose,
        observation: CameraFeatureObservation,
        intrinsics: PinholeIntrinsics
    ): Double? {
        val rotation = pose.rotation
        val translation = pose.translation
        if (rotation.size != 9 || translation.size != 3 ||
            rotation.any { !it.isFinite() } || translation.any { !it.isFinite() }
        ) return null
        val cameraPoint = doubleArrayOf(
            rotation[0] * pointWorld[0] + rotation[1] * pointWorld[1] + rotation[2] * pointWorld[2] + translation[0],
            rotation[3] * pointWorld[0] + rotation[4] * pointWorld[1] + rotation[5] * pointWorld[2] + translation[1],
            rotation[6] * pointWorld[0] + rotation[7] * pointWorld[1] + rotation[8] * pointWorld[2] + translation[2]
        )
        if (cameraPoint.any { !it.isFinite() } || cameraPoint[2] <= 1e-9) return null
        val predicted = CameraProjectionMath.project(cameraPoint, intrinsics)
        val dx = predicted[0] - observation.x
        val dy = predicted[1] - observation.y
        val error = sqrt(dx * dx + dy * dy)
        return error.takeIf { it.isFinite() }
    }
}
