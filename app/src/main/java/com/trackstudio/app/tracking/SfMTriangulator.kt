package com.trackstudio.app.tracking

import org.opencv.core.Mat
import org.opencv.core.Point3
import kotlin.math.sqrt

/** Sparse calibrated multi-view triangulation with bounded Android-side consensus. */
internal object SfMTriangulator {

    fun triangulate(
        featureTracks: List<CameraFeatureTrack>,
        poses: List<CameraPose>,
        cameraMatrix: Mat,
        maximumReprojectionErrorPx: Double = 4.0,
        minimumParallaxDegrees: Double = 1.5,
        minimumObservationInlierRatio: Double = 0.70,
        onProgress: (Int, Int) -> Unit = { _, _ -> }
    ): TriangulationResult {
        require(cameraMatrix.rows() == 3 && cameraMatrix.cols() == 3 && !cameraMatrix.empty()) {
            "Camera matrix must be a non-empty 3x3 matrix."
        }
        val intrinsics = PinholeIntrinsics(
            fx = cameraMatrix.get(0, 0)?.firstOrNull() ?: Double.NaN,
            skew = cameraMatrix.get(0, 1)?.firstOrNull() ?: Double.NaN,
            cx = cameraMatrix.get(0, 2)?.firstOrNull() ?: Double.NaN,
            fy = cameraMatrix.get(1, 1)?.firstOrNull() ?: Double.NaN,
            cy = cameraMatrix.get(1, 2)?.firstOrNull() ?: Double.NaN
        )
        return triangulate(
            featureTracks,
            poses,
            intrinsics,
            maximumReprojectionErrorPx,
            minimumParallaxDegrees,
            minimumObservationInlierRatio,
            onProgress
        )
    }

    fun triangulate(
        featureTracks: List<CameraFeatureTrack>,
        poses: List<CameraPose>,
        intrinsics: PinholeIntrinsics,
        maximumReprojectionErrorPx: Double = 4.0,
        minimumParallaxDegrees: Double = 1.5,
        minimumObservationInlierRatio: Double = 0.70,
        onProgress: (Int, Int) -> Unit = { _, _ -> }
    ): TriangulationResult {
        require(maximumReprojectionErrorPx.isFinite() && maximumReprojectionErrorPx > 0.0)
        require(minimumParallaxDegrees.isFinite() && minimumParallaxDegrees >= 1e-3)
        require(minimumObservationInlierRatio.isFinite() && minimumObservationInlierRatio in 0.0..1.0)
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0 &&
            intrinsics.fy.isFinite() && intrinsics.fy > 0.0 &&
            intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite()
        ) { "Camera intrinsics must be finite and have positive focal lengths." }

        val poseByFrame = poses.associateBy { it.frameIndex }
        val landmarks = mutableListOf<CameraLandmark>()
        val inlierTracks = mutableListOf<CameraFeatureTrack>()
        var rejectedShort = 0
        var rejectedMissingPose = 0
        var rejectedGeometry = 0
        var rejectedReprojection = 0

        for ((trackIndex, track) in featureTracks.withIndex()) {
            if (trackIndex % 16 == 0 || trackIndex == featureTracks.lastIndex) {
                onProgress(trackIndex + 1, featureTracks.size)
            }
            if (track.observations.size < 2) {
                rejectedShort++
                continue
            }
            // Held and unregistered frames retain timeline alignment but are not geometric views.
            val validObservations = track.observations.mapNotNull { observation ->
                poseByFrame[observation.frameIndex]
                    ?.takeIf {
                        it.status != CameraPoseStatus.HELD_LAST_VALID &&
                            it.status != CameraPoseStatus.UNREGISTERED
                    }
                    ?.let { observation to it }
            }
            if (validObservations.size < 2) {
                rejectedMissingPose++
                continue
            }

            val robustEstimate = MultiViewTriangulationMath.estimate(
                observations = validObservations.map { (observation, pose) ->
                    TriangulationView(observation.x, observation.y, pose)
                },
                intrinsics = intrinsics,
                minimumParallaxDegrees = minimumParallaxDegrees,
                minimumInlierRatio = minimumObservationInlierRatio,
                maximumReprojectionErrorPx = maximumReprojectionErrorPx
            )
            if (robustEstimate == null) {
                rejectedGeometry++
                continue
            }

            val point = robustEstimate.pointWorld
            val inlierErrors = mutableListOf<Double>()
            val inlierObservations = mutableListOf<CameraFeatureObservation>()
            val robustInlierIndices = robustEstimate.inlierIndices.toHashSet()
            for ((index, pair) in validObservations.withIndex()) {
                if (index !in robustInlierIndices) continue
                val (observation, pose) = pair
                val cameraPoint = transform(pose, point)
                // Behind-camera views are not reprojection inliers and remain in the ratio denominator.
                if (cameraPoint[2] <= 1e-6) continue
                val error = reprojectionError(cameraPoint, observation, intrinsics)
                if (error.isFinite() && error <= maximumReprojectionErrorPx) {
                    inlierErrors += error
                    inlierObservations += observation
                }
            }
            val inlierRatio = inlierErrors.size.toDouble() / validObservations.size
            if (inlierErrors.size < 2 || inlierRatio < minimumObservationInlierRatio) {
                rejectedReprojection++
                continue
            }

            val meanInlierError = inlierErrors.average()
            if (!meanInlierError.isFinite()) {
                rejectedReprojection++
                continue
            }
            landmarks += CameraLandmark(
                trackId = track.id,
                position = Point3(point[0], point[1], point[2]),
                observationCount = inlierErrors.size,
                meanReprojectionErrorPx = meanInlierError,
                triangulationAngleDegrees = robustEstimate.triangulationAngleDegrees
            )
            inlierTracks += track.copy(observations = inlierObservations)
        }

        return TriangulationResult(
            landmarks = landmarks,
            inlierTracks = inlierTracks,
            stats = TriangulationStats(
                inputTracks = featureTracks.size,
                triangulatedLandmarks = landmarks.size,
                rejectedShortTracks = rejectedShort,
                rejectedMissingPoses = rejectedMissingPose,
                rejectedInvalidGeometry = rejectedGeometry,
                rejectedReprojection = rejectedReprojection,
                meanReprojectionErrorPx = if (landmarks.isEmpty()) 0.0
                    else landmarks.map { it.meanReprojectionErrorPx }.average(),
                meanTriangulationAngleDegrees = if (landmarks.isEmpty()) 0.0
                    else landmarks.map { it.triangulationAngleDegrees }.average()
            )
        )
    }

    private fun transform(pose: CameraPose, point: DoubleArray): DoubleArray {
        val r = pose.rotation
        val t = pose.translation
        return doubleArrayOf(
            r[0] * point[0] + r[1] * point[1] + r[2] * point[2] + t[0],
            r[3] * point[0] + r[4] * point[1] + r[5] * point[2] + t[1],
            r[6] * point[0] + r[7] * point[1] + r[8] * point[2] + t[2]
        )
    }

    private fun reprojectionError(
        cameraPoint: DoubleArray,
        observation: CameraFeatureObservation,
        intrinsics: PinholeIntrinsics
    ): Double {
        val predicted = CameraProjectionMath.project(cameraPoint, intrinsics)
        val dx = predicted[0] - observation.x
        val dy = predicted[1] - observation.y
        return sqrt(dx * dx + dy * dy)
    }
}

internal data class TriangulationResult(
    val landmarks: List<CameraLandmark>,
    val inlierTracks: List<CameraFeatureTrack>,
    val stats: TriangulationStats
)

data class CameraLandmark(
    val trackId: Int,
    val position: Point3,
    val observationCount: Int,
    val meanReprojectionErrorPx: Double,
    val triangulationAngleDegrees: Double = 0.0
)

data class TriangulationStats(
    val inputTracks: Int = 0,
    val triangulatedLandmarks: Int = 0,
    val rejectedShortTracks: Int = 0,
    val rejectedMissingPoses: Int = 0,
    val rejectedInvalidGeometry: Int = 0,
    val rejectedReprojection: Int = 0,
    val meanReprojectionErrorPx: Double = 0.0,
    val meanTriangulationAngleDegrees: Double = 0.0
)

internal fun TriangulationStats.withFinalLandmarks(landmarks: List<CameraLandmark>): TriangulationStats =
    copy(
        triangulatedLandmarks = landmarks.size,
        meanReprojectionErrorPx = landmarks.map { it.meanReprojectionErrorPx }
            .takeIf { it.isNotEmpty() }?.average() ?: 0.0,
        meanTriangulationAngleDegrees = landmarks.map { it.triangulationAngleDegrees }
            .takeIf { it.isNotEmpty() }?.average() ?: 0.0
    )
