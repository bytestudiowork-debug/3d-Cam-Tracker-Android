package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.PinholeIntrinsics
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Incremental point/observation filtering adapted from COLMAP's ObservationManager.
 * Reprojection outliers are removed before triangulation-angle stability is checked.
 */
internal object ColmapObservationFilter {
    const val DEFAULT_MAX_REPROJECTION_ERROR_PX = 4.0
    const val DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES = 1.5

    data class Result(
        val tracksById: Map<Int, CameraFeatureTrack>,
        val landmarksByTrackId: Map<Int, CameraLandmark>,
        val removedObservationsByTrack: Map<Int, List<CameraFeatureObservation>>,
        val deletedTrackIds: Set<Int>,
        val filteredObservationCount: Int
    )

    fun filter(
        tracksById: Map<Int, CameraFeatureTrack>,
        landmarksByTrackId: Map<Int, CameraLandmark>,
        posesByFrame: Map<Int, CameraPose>,
        intrinsics: PinholeIntrinsics,
        affectedFrameIds: Set<Int> = emptySet(),
        changedTrackIds: Set<Int> = emptySet(),
        maxReprojectionErrorPx: Double = DEFAULT_MAX_REPROJECTION_ERROR_PX,
        minTriangulationAngleDegrees: Double = DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES
    ): Result {
        require(maxReprojectionErrorPx.isFinite() && maxReprojectionErrorPx > 0.0)
        require(minTriangulationAngleDegrees.isFinite() && minTriangulationAngleDegrees >= 0.0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        require(intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite())

        val registeredPoses = posesByFrame.filterValues { pose ->
            pose.status != CameraPoseStatus.HELD_LAST_VALID &&
                pose.status != CameraPoseStatus.UNREGISTERED
        }
        val candidates = linkedSetOf<Int>()
        if (affectedFrameIds.isNotEmpty()) {
            tracksById.values.asSequence()
                .filter { track -> track.observations.any { it.frameIndex in affectedFrameIds } }
                .mapTo(candidates) { it.id }
        }
        candidates.addAll(changedTrackIds)

        val filteredTracks = tracksById.toMutableMap()
        val filteredLandmarks = landmarksByTrackId.toMutableMap()
        val removedByTrack = linkedMapOf<Int, List<CameraFeatureObservation>>()
        val deletedTrackIds = linkedSetOf<Int>()
        val maximumSquaredError = maxReprojectionErrorPx * maxReprojectionErrorPx
        var filteredObservationCount = 0

        for (trackId in candidates) {
            val track = tracksById[trackId] ?: continue
            val landmark = landmarksByTrackId[trackId] ?: continue
            val point = doubleArrayOf(
                landmark.position.x,
                landmark.position.y,
                landmark.position.z
            )
            val retained = ArrayList<CameraFeatureObservation>(track.observations.size)
            val rejected = ArrayList<CameraFeatureObservation>()
            val errorsByObservation = HashMap<CameraFeatureObservation, Double>()

            for (observation in track.observations) {
                val pose = registeredPoses[observation.frameIndex]
                if (pose == null) {
                    // A non-registered timeline observation is not part of this 3D track yet.
                    retained += observation
                    continue
                }
                val errorSquared = squaredReprojectionError(point, pose, observation, intrinsics)
                if (errorSquared <= maximumSquaredError) {
                    retained += observation
                    errorsByObservation[observation] = sqrt(errorSquared)
                } else {
                    rejected += observation
                }
            }

            if (rejected.isNotEmpty()) {
                removedByTrack[trackId] = rejected
                filteredObservationCount += rejected.size
            }

            val registeredRetained = retained.filter { it.frameIndex in registeredPoses }
            if (registeredRetained.size <= 1) {
                filteredTracks.remove(trackId)
                filteredLandmarks.remove(trackId)
                deletedTrackIds += trackId
                filteredObservationCount += retained.size
                continue
            }

            val maximumAngle = maximumTriangulationAngleDegrees(
                point = point,
                observations = registeredRetained,
                posesByFrame = registeredPoses
            )
            if (maximumAngle < minTriangulationAngleDegrees) {
                filteredTracks.remove(trackId)
                filteredLandmarks.remove(trackId)
                deletedTrackIds += trackId
                filteredObservationCount += retained.size
                continue
            }

            if (rejected.isNotEmpty()) {
                filteredTracks[trackId] = track.copy(observations = retained)
            }
            val meanError = registeredRetained.mapNotNull(errorsByObservation::get)
                .takeIf { it.isNotEmpty() }?.average()
                ?: landmark.meanReprojectionErrorPx
            filteredLandmarks[trackId] = landmark.copy(
                observationCount = registeredRetained.size,
                meanReprojectionErrorPx = meanError,
                triangulationAngleDegrees = maximumAngle
            )
        }

        return Result(
            tracksById = filteredTracks,
            landmarksByTrackId = filteredLandmarks,
            removedObservationsByTrack = removedByTrack,
            deletedTrackIds = deletedTrackIds,
            filteredObservationCount = filteredObservationCount
        )
    }

    private fun squaredReprojectionError(
        point: DoubleArray,
        pose: CameraPose,
        observation: CameraFeatureObservation,
        intrinsics: PinholeIntrinsics
    ): Double {
        if (point.size != 3 || pose.rotation.size != 9 || pose.translation.size != 3 ||
            point.any { !it.isFinite() } || pose.rotation.any { !it.isFinite() } ||
            pose.translation.any { !it.isFinite() } ||
            !observation.x.isFinite() || !observation.y.isFinite()
        ) return Double.POSITIVE_INFINITY

        val r = pose.rotation
        val t = pose.translation
        val cameraX = r[0] * point[0] + r[1] * point[1] + r[2] * point[2] + t[0]
        val cameraY = r[3] * point[0] + r[4] * point[1] + r[5] * point[2] + t[1]
        val cameraZ = r[6] * point[0] + r[7] * point[1] + r[8] * point[2] + t[2]
        if (!cameraX.isFinite() || !cameraY.isFinite() || !cameraZ.isFinite() ||
            cameraZ <= 1e-12
        ) return Double.POSITIVE_INFINITY

        val dx = (intrinsics.fx * cameraX + intrinsics.skew * cameraY) / cameraZ +
            intrinsics.cx - observation.x
        val dy = intrinsics.fy * cameraY / cameraZ + intrinsics.cy - observation.y
        val errorSquared = dx * dx + dy * dy
        return errorSquared.takeIf { it.isFinite() } ?: Double.POSITIVE_INFINITY
    }

    private fun maximumTriangulationAngleDegrees(
        point: DoubleArray,
        observations: List<CameraFeatureObservation>,
        posesByFrame: Map<Int, CameraPose>
    ): Double {
        var maximumRadians = 0.0
        for (firstIndex in observations.indices) {
            val firstPose = posesByFrame[observations[firstIndex].frameIndex] ?: continue
            val firstRay = rayFromCameraCenter(point, firstPose) ?: continue
            for (secondIndex in firstIndex + 1 until observations.size) {
                val secondPose = posesByFrame[observations[secondIndex].frameIndex] ?: continue
                val secondRay = rayFromCameraCenter(point, secondPose) ?: continue
                val cosine = (firstRay[0] * secondRay[0] +
                    firstRay[1] * secondRay[1] + firstRay[2] * secondRay[2])
                    .coerceIn(-1.0, 1.0)
                maximumRadians = maxOf(maximumRadians, acos(cosine))
            }
        }
        return Math.toDegrees(maximumRadians)
    }

    private fun rayFromCameraCenter(point: DoubleArray, pose: CameraPose): DoubleArray? {
        if (point.size != 3 || pose.rotation.size != 9 || pose.translation.size != 3 ||
            point.any { !it.isFinite() } || pose.rotation.any { !it.isFinite() } ||
            pose.translation.any { !it.isFinite() }
        ) return null
        val r = pose.rotation
        val t = pose.translation
        val centerX = -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2])
        val centerY = -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2])
        val centerZ = -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])
        val x = point[0] - centerX
        val y = point[1] - centerY
        val z = point[2] - centerZ
        val norm = sqrt(x * x + y * y + z * z)
        if (!norm.isFinite() || norm <= 1e-12) return null
        return doubleArrayOf(x / norm, y / norm, z / norm)
    }
}
