/*
 * Copyright (c) 2024, ETH Zurich. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the name of ETH Zurich nor the names of its contributors may
 *       be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Android adaptation of glomap/processors/track_filter.cc from
 * colmap/glomap commit 99806d0869f802fad218516a2e027793e7ca687d.
 */
package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraIntrinsics
import kotlin.math.cos
import kotlin.math.sqrt

/** Source-faithful track observation filters for GLOMAP's global reconstruction path. */
internal object GlomapTrackFilter {
    const val DEFAULT_MAX_ANGLE_ERROR_DEGREES = 1.0
    const val DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED = 1e-2
    const val DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES = 1.0
    private const val EPS = 1e-12

    data class Result(
        val tracks: List<CameraFeatureTrack>,
        val changedTrackCount: Int,
        val removedObservationCount: Int
    )

    /** Mirrors TrackFilter::FilterTracksByReprojection for normalized pinhole observations. */
    fun filterTracksByReprojection(
        tracks: List<CameraFeatureTrack>,
        cameraCentersByFrame: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        intrinsics: CameraIntrinsics,
        maxReprojectionError: Double = DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED,
        inNormalizedImage: Boolean = true
    ): Result {
        require(maxReprojectionError.isFinite() && maxReprojectionError > 0.0)
        requireValidIntrinsics(intrinsics)
        return filterObservations(tracks) { track, observation ->
            val point = pointsByTrackId[track.id] ?: return@filterObservations true
            val cameraPoint = cameraPoint(
                observation.frameIndex, point, cameraCentersByFrame, rotationsByFrame
            ) ?: return@filterObservations true
            if (!cameraPoint.all { it.isFinite() } || cameraPoint[2] < EPS) {
                return@filterObservations false
            }

            val normalized = normalizedFeature(observation, intrinsics)
            val x = cameraPoint[0] / cameraPoint[2]
            val y = cameraPoint[1] / cameraPoint[2]
            val error = if (inNormalizedImage) {
                val dx = x - normalized[0] / (normalized[2] + EPS)
                val dy = y - normalized[1] / (normalized[2] + EPS)
                sqrt(dx * dx + dy * dy)
            } else {
                val predictedX = intrinsics.fx * x + intrinsics.skew * y + intrinsics.cx
                val predictedY = intrinsics.fy * y + intrinsics.cy
                val dx = predictedX - observation.x
                val dy = predictedY - observation.y
                sqrt(dx * dx + dy * dy)
            }
            error < maxReprojectionError
        }
    }

    /** Mirrors TrackFilter::FilterTracksByAngle, including the doubled uncalibrated threshold. */
    fun filterTracksByAngle(
        tracks: List<CameraFeatureTrack>,
        cameraCentersByFrame: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        intrinsics: CameraIntrinsics,
        maxAngleErrorDegrees: Double = DEFAULT_MAX_ANGLE_ERROR_DEGREES,
        hasPriorFocalLength: Boolean = true
    ): Result {
        require(maxAngleErrorDegrees.isFinite() && maxAngleErrorDegrees in 0.0..180.0)
        requireValidIntrinsics(intrinsics)
        val thresholdDegrees = maxAngleErrorDegrees * if (hasPriorFocalLength) 1.0 else 2.0
        val threshold = cos(Math.toRadians(thresholdDegrees))
        return filterObservations(tracks) { track, observation ->
            val point = pointsByTrackId[track.id] ?: return@filterObservations true
            val cameraPoint = cameraPoint(
                observation.frameIndex, point, cameraCentersByFrame, rotationsByFrame
            ) ?: return@filterObservations true
            if (!cameraPoint.all { it.isFinite() } || cameraPoint[2] < EPS) {
                return@filterObservations false
            }
            val pointRay = normalized(cameraPoint) ?: return@filterObservations false
            val featureRay = normalized(normalizedFeature(observation, intrinsics))
                ?: return@filterObservations false
            dot(pointRay, featureRay) > threshold
        }
    }

    /** Mirrors TrackFilter::FilterTrackTriangulationAngle. */
    fun filterTrackTriangulationAngle(
        tracks: List<CameraFeatureTrack>,
        cameraCentersByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        minAngleDegrees: Double = DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES
    ): Result {
        require(minAngleDegrees.isFinite() && minAngleDegrees in 0.0..180.0)
        val threshold = cos(Math.toRadians(minAngleDegrees))
        var changedTracks = 0
        var removedObservations = 0
        val filteredTracks = tracks.map { track ->
            val point = pointsByTrackId[track.id]
            val directions = if (point == null || !isVector3(point)) {
                null
            } else {
                track.observations.map { observation ->
                    val center = cameraCentersByFrame[observation.frameIndex]
                    if (center == null || !isVector3(center)) null
                    else normalized(doubleArrayOf(
                        point[0] - center[0],
                        point[1] - center[1],
                        point[2] - center[2]
                    ))
                }
            }
            val hasCompleteGeometry = directions != null && directions.all { it != null }
            val hasSufficientAngle = if (hasCompleteGeometry) {
                var found = false
                val rays = directions!!.filterNotNull()
                for (first in rays.indices) {
                    for (second in first + 1 until rays.size) {
                        if (dot(rays[first], rays[second]) < threshold) {
                            found = true
                            break
                        }
                    }
                    if (found) break
                }
                found
            } else {
                true
            }
            val observations = if (hasSufficientAngle) track.observations else emptyList()
            if (observations.size != track.observations.size) {
                changedTracks++
                removedObservations += track.observations.size - observations.size
            }
            track.copy(observations = observations)
        }
        return Result(filteredTracks, changedTracks, removedObservations)
    }

    /** GLOMAP's post-positioning sequence and source defaults. */
    fun filterAfterPositioning(
        tracks: List<CameraFeatureTrack>,
        cameraCentersByFrame: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        intrinsics: CameraIntrinsics,
        hasPriorFocalLength: Boolean = true
    ): List<CameraFeatureTrack> {
        var filtered = filterTracksByAngle(
            tracks, cameraCentersByFrame, rotationsByFrame, pointsByTrackId,
            intrinsics, DEFAULT_MAX_ANGLE_ERROR_DEGREES, hasPriorFocalLength
        ).tracks
        filtered = filterTrackTriangulationAngle(
            filtered, cameraCentersByFrame, pointsByTrackId,
            DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES
        ).tracks
        filtered = filterTracksByReprojection(
            filtered, cameraCentersByFrame, rotationsByFrame, pointsByTrackId,
            intrinsics, 10.0 * DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED,
            inNormalizedImage = true
        ).tracks
        return filtered
    }

    /** GLOMAP's final post-bundle-adjustment filtering sequence. */
    fun filterAfterBundleAdjustment(
        tracks: List<CameraFeatureTrack>,
        cameraCentersByFrame: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        intrinsics: CameraIntrinsics
    ): List<CameraFeatureTrack> {
        var filtered = filterTracksByReprojection(
            tracks, cameraCentersByFrame, rotationsByFrame, pointsByTrackId,
            intrinsics, DEFAULT_MAX_REPROJECTION_ERROR_NORMALIZED,
            inNormalizedImage = true
        ).tracks
        filtered = filterTrackTriangulationAngle(
            filtered, cameraCentersByFrame, pointsByTrackId,
            DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES
        ).tracks
        return filtered
    }

    private fun filterObservations(
        tracks: List<CameraFeatureTrack>,
        keepObservation: (CameraFeatureTrack, CameraFeatureObservation) -> Boolean
    ): Result {
        var changedTracks = 0
        var removedObservations = 0
        val filteredTracks = tracks.map { track ->
            val observations = track.observations.filter { keepObservation(track, it) }
            if (observations.size != track.observations.size) {
                changedTracks++
                removedObservations += track.observations.size - observations.size
            }
            track.copy(observations = observations)
        }
        return Result(filteredTracks, changedTracks, removedObservations)
    }

    private fun cameraPoint(
        frameIndex: Int,
        point: DoubleArray,
        cameraCentersByFrame: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>
    ): DoubleArray? {
        val center = cameraCentersByFrame[frameIndex] ?: return null
        val rotation = rotationsByFrame[frameIndex] ?: return null
        if (!isVector3(point) || !isVector3(center) || rotation.size != 9 ||
            rotation.any { !it.isFinite() }
        ) return null
        val x = point[0] - center[0]
        val y = point[1] - center[1]
        val z = point[2] - center[2]
        return doubleArrayOf(
            rotation[0] * x + rotation[1] * y + rotation[2] * z,
            rotation[3] * x + rotation[4] * y + rotation[5] * z,
            rotation[6] * x + rotation[7] * y + rotation[8] * z
        )
    }

    private fun normalizedFeature(
        observation: CameraFeatureObservation,
        intrinsics: CameraIntrinsics
    ): DoubleArray {
        val y = (observation.y - intrinsics.cy) / intrinsics.fy
        val x = (observation.x - intrinsics.cx - intrinsics.skew * y) / intrinsics.fx
        return doubleArrayOf(x, y, 1.0)
    }

    private fun normalized(vector: DoubleArray): DoubleArray? {
        if (!isVector3(vector)) return null
        val norm = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
        if (!norm.isFinite() || norm <= 0.0) return null
        return DoubleArray(3) { vector[it] / norm }
    }

    private fun dot(first: DoubleArray, second: DoubleArray): Double =
        first[0] * second[0] + first[1] * second[1] + first[2] * second[2]

    private fun isVector3(value: DoubleArray): Boolean =
        value.size == 3 && value.all { it.isFinite() }

    private fun requireValidIntrinsics(intrinsics: CameraIntrinsics) {
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0 &&
            intrinsics.fy.isFinite() && intrinsics.fy > 0.0 &&
            intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite()
        )
    }
}
