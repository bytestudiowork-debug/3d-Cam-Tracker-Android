package com.trackstudio.app.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvException
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Registers one frame against an existing sparse 3D map.
 *
 * This is the absolute-pose step used after a valid two-view reconstruction.
 * Unlike chaining recoverPose translations, PnP estimates each camera in the
 * same world coordinate system and therefore preserves the map's scale.
 *
 * This class is deliberately independent of frame decoding and UI state.
 */
internal object IncrementalCameraRegistrar {

    fun register(
        frameIndex: Int,
        frameWidth: Int,
        frameHeight: Int,
        featureTracks: List<CameraFeatureTrack>,
        landmarks: Collection<CameraLandmark>,
        cameraMatrix: Mat,
        minimumCorrespondences: Int = CameraRegistrationPolicy.MINIMUM_CORRESPONDENCES,
        minimumInlierRatio: Double = CameraRegistrationPolicy.MINIMUM_INLIER_RATIO,
        maximumReprojectionErrorPx: Float = CameraRegistrationPolicy.FINAL_REPROJECTION_ERROR_PX,
        minimumOccupiedGridCells: Int = CameraRegistrationPolicy.MINIMUM_OCCUPIED_GRID_CELLS,
        ransacReprojectionErrorPx: Float = CameraRegistrationPolicy.RANSAC_REPROJECTION_ERROR_PX,
        minimumCheiralityRatio: Double = CameraRegistrationPolicy.MINIMUM_CHEIRALITY_RATIO
    ): CameraRegistrationResult {
        require(frameWidth > 0 && frameHeight > 0)
        require(cameraMatrix.rows() == 3 && cameraMatrix.cols() == 3 && !cameraMatrix.empty())
        require(minimumCorrespondences >= 4)
        require(minimumInlierRatio.isFinite() && minimumInlierRatio in 0.0..1.0)
        require(maximumReprojectionErrorPx.isFinite() && maximumReprojectionErrorPx > 0f)
        require(minimumOccupiedGridCells in 0..16)
        require(ransacReprojectionErrorPx.isFinite() && ransacReprojectionErrorPx > 0f)
        require(minimumCheiralityRatio.isFinite() && minimumCheiralityRatio in 0.0..1.0)

        val intrinsics = PinholeIntrinsics(
            cameraMatrix.get(0, 0)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 2)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 2)?.firstOrNull() ?: Double.NaN
        )
        if (!intrinsics.fx.isFinite() || !intrinsics.fy.isFinite() ||
            !intrinsics.skew.isFinite() || !intrinsics.cx.isFinite() ||
            !intrinsics.cy.isFinite() || intrinsics.fx <= 0.0 || intrinsics.fy <= 0.0
        ) return CameraRegistrationResult.rejected(frameIndex, 0, "Invalid camera intrinsics")

        val observationsByTrack = featureTracks.associate { track ->
            track.id to track.observations.firstOrNull { it.frameIndex == frameIndex }
        }
        val points3d = mutableListOf<Point3>()
        val points2d = mutableListOf<Point>()

        // A landmark is usable only when its source track has an observation
        // in this exact frame. Track IDs provide the 2D-3D association.
        for (landmark in landmarks) {
            val observation = observationsByTrack[landmark.trackId] ?: continue
            val p = landmark.position
            if (!p.x.isFinite() || !p.y.isFinite() || !p.z.isFinite() ||
                !observation.x.isFinite() || !observation.y.isFinite()
            ) continue
            points3d += Point3(p.x, p.y, p.z)
            points2d += Point(observation.x, observation.y)
        }

        val correspondenceCount = points3d.size
        if (correspondenceCount < minimumCorrespondences) {
            return CameraRegistrationResult.rejected(
                frameIndex, correspondenceCount, "Not enough 2D-3D correspondences"
            )
        }

        val objectPoints = MatOfPoint3f()
        val imagePoints = MatOfPoint2f()
        val distortion = MatOfDouble()
        var rotationVector = Mat()
        var translationVector = Mat()
        val inlierIndices = Mat()
        val inlierObjectPoints = MatOfPoint3f()
        val inlierImagePoints = MatOfPoint2f()
        val projectedPoints = MatOfPoint2f()
        val rotationMatrix = Mat()

        return try {
            objectPoints.fromArray(*points3d.toTypedArray())
            imagePoints.fromArray(*points2d.toTypedArray())

            val solved = Calib3d.solvePnPRansac(
                objectPoints,
                imagePoints,
                cameraMatrix,
                distortion,
                rotationVector,
                translationVector,
                false,
                CameraRegistrationPolicy.RANSAC_MAX_ITERATIONS,
                ransacReprojectionErrorPx,
                CameraRegistrationPolicy.RANSAC_CONFIDENCE,
                inlierIndices,
                Calib3d.SOLVEPNP_P3P
            )
            if (!solved || inlierIndices.empty()) {
                return CameraRegistrationResult.rejected(
                    frameIndex, correspondenceCount, "PnP-RANSAC could not estimate a stable pose"
                )
            }

            val normalizedRotationVector = toDoubleVector(rotationVector)
            val normalizedTranslationVector = toDoubleVector(translationVector)
            if (normalizedRotationVector == null || normalizedTranslationVector == null) {
                normalizedRotationVector?.release()
                normalizedTranslationVector?.release()
                return CameraRegistrationResult.rejected(
                    frameIndex, correspondenceCount, "Invalid PnP rotation/translation vector output"
                )
            }
            rotationVector.release()
            translationVector.release()
            rotationVector = normalizedRotationVector
            translationVector = normalizedTranslationVector

            val rawInlierCount = inlierIndices.total().toInt()
            var inlierIds = (0 until rawInlierCount).mapNotNull { index ->
                val coordinate = OpenCvMaskLayout.coordinates(
                    inlierIndices.rows(), inlierIndices.cols(), index, rawInlierCount
                ) ?: return@mapNotNull null
                inlierIndices.get(coordinate.first, coordinate.second)?.firstOrNull()?.toInt()
            }.filter { it in points3d.indices }.distinct()

            var inlierCount = inlierIds.size
            var inlierRatio = inlierCount.toDouble() / correspondenceCount
            if (inlierCount < minimumCorrespondences || inlierRatio < minimumInlierRatio) {
                return CameraRegistrationResult.rejected(
                    frameIndex,
                    correspondenceCount,
                    "PnP pose failed inlier quality gate",
                    inlierCount,
                    inlierRatio
                )
            }

            val initialObjectArray = inlierIds.map { points3d[it] }.toTypedArray()
            val initialImageArray = inlierIds.map { points2d[it] }.toTypedArray()

            // Refine the RANSAC pose against its initial consensus.
            val refinement = CameraPoseRefiner.refine(
                initialObjectArray,
                initialImageArray,
                cameraMatrix,
                rotationVector,
                translationVector
            )
            val ransacRotationVector = DoubleArray(3)
            val ransacTranslation = DoubleArray(3)
            val expectedPoseReadBytes = ransacRotationVector.size * java.lang.Double.BYTES
            val rotationReadBytes = rotationVector.get(0, 0, ransacRotationVector)
            val translationReadBytes = translationVector.get(0, 0, ransacTranslation)
            if (rotationReadBytes != expectedPoseReadBytes ||
                translationReadBytes != expectedPoseReadBytes
            ) {
                return CameraRegistrationResult.rejected(
                    frameIndex,
                    correspondenceCount,
                    "Could not read the RANSAC PnP pose: " +
                        "rvec ${matrixReadSummary(rotationVector, rotationReadBytes, expectedPoseReadBytes)}; " +
                        "tvec ${matrixReadSummary(translationVector, translationReadBytes, expectedPoseReadBytes)}",
                    inlierCount,
                    inlierRatio
                )
            }
            if (refinement != null &&
                (rotationVector.put(0, 0, *refinement.rotationVector) != refinement.rotationVector.size ||
                    translationVector.put(0, 0, *refinement.translation) != refinement.translation.size)
            ) {
                return CameraRegistrationResult.rejected(
                    frameIndex,
                    correspondenceCount,
                    "Could not store the robustly refined PnP pose",
                    inlierCount,
                    inlierRatio
                )
            }

            val initialPose = poseModel(
                refinement?.rotationVector ?: ransacRotationVector,
                refinement?.translation ?: ransacTranslation
            )
                ?: return CameraRegistrationResult.rejected(
                    frameIndex, correspondenceCount, "Invalid PnP pose", inlierCount, inlierRatio
                )
            val polished = LocalConsensusOptimizer.refine(
                initialModel = initialPose,
                observationCount = correspondenceCount,
                initialInliers = inlierIds,
                minimumInliers = minimumCorrespondences,
                threshold = ransacReprojectionErrorPx.toDouble(),
                maximumRefitRounds = CameraRegistrationPolicy.LOCAL_OPTIMIZATION_ROUNDS,
                fitCandidates = { currentPose, consensusIndices ->
                    fitPoseModel(
                        currentPose, consensusIndices, points3d, points2d, cameraMatrix
                    )?.let(::listOf).orEmpty()
                },
                residual = { pose, index ->
                    reprojectionError(pose, points3d[index], points2d[index], intrinsics)
                }
            )
            val finalPose = if (polished != null && polished.inlierIndices.size >= inlierCount) {
                inlierIds = polished.inlierIndices
                inlierCount = inlierIds.size
                inlierRatio = inlierCount.toDouble() / correspondenceCount
                polished.model
            } else {
                initialPose
            }
            if (rotationVector.put(0, 0, *finalPose.rotationVector) != 3 ||
                translationVector.put(0, 0, *finalPose.translation) != 3
            ) {
                return CameraRegistrationResult.rejected(
                    frameIndex, correspondenceCount, "Invalid locally refined pose", inlierCount, inlierRatio
                )
            }
            val inlierObjectArray = inlierIds.map { points3d[it] }.toTypedArray()
            val inlierImageArray = inlierIds.map { points2d[it] }.toTypedArray()
            inlierObjectPoints.fromArray(*inlierObjectArray)
            inlierImagePoints.fromArray(*inlierImageArray)

            Calib3d.Rodrigues(rotationVector, rotationMatrix)
            val rotation = OpenCvMatrixReadback.readCv64F(rotationMatrix, 3, 3)
                ?: return CameraRegistrationResult.rejected(
                    frameIndex,
                    correspondenceCount,
                    "Could not read the refined PnP rotation",
                    inlierCount,
                    inlierRatio
                )
            val translation = OpenCvMatrixReadback.readCv64F(translationVector, 3, 1)
                ?: return CameraRegistrationResult.rejected(
                    frameIndex,
                    correspondenceCount,
                    "Could not read the refined PnP translation",
                    inlierCount,
                    inlierRatio
                )
            val positiveDepthCount = inlierIds.count { index ->
                val point = points3d[index]
                val depth = rotation[6] * point.x + rotation[7] * point.y +
                    rotation[8] * point.z + translation[2]
                depth.isFinite() && depth > 1e-6
            }
            val cheiralityRatio = positiveDepthCount.toDouble() / inlierCount
            if (positiveDepthCount < minimumCorrespondences || cheiralityRatio < minimumCheiralityRatio) {
                return CameraRegistrationResult.rejected(
                    frameIndex,
                    correspondenceCount,
                    "PnP inliers fail positive-depth validation",
                    inlierCount,
                    inlierRatio
                )
            }

            val occupiedCells = occupiedGridCells(
                inlierImageArray, frameWidth, frameHeight
            )
            if (occupiedCells < minimumOccupiedGridCells) {
                return CameraRegistrationResult.rejected(
                    frameIndex,
                    correspondenceCount,
                    "PnP inliers are too spatially concentrated",
                    inlierCount,
                    inlierRatio,
                    occupiedCells
                )
            }

            Calib3d.projectPoints(
                inlierObjectPoints,
                rotationVector,
                translationVector,
                cameraMatrix,
                distortion,
                projectedPoints
            )
            val projected = projectedPoints.toArray()
            if (projected.size != inlierImageArray.size) {
                return CameraRegistrationResult.rejected(
                    frameIndex, correspondenceCount, "Invalid reprojection result",
                    inlierCount, inlierRatio, occupiedCells
                )
            }

            var squaredError = 0.0
            val errors = DoubleArray(projected.size)
            for (i in projected.indices) {
                val dx = projected[i].x - inlierImageArray[i].x
                val dy = projected[i].y - inlierImageArray[i].y
                val error = sqrt(dx * dx + dy * dy)
                errors[i] = error
                squaredError += error * error
            }
            val rms = sqrt(squaredError / errors.size)
            val median = errors.sorted().let { sorted ->
                val middle = sorted.size / 2
                if (sorted.size % 2 == 0) {
                    (sorted[middle - 1] + sorted[middle]) * 0.5
                } else sorted[middle]
            }
            if (!rms.isFinite() || !median.isFinite() ||
                median > maximumReprojectionErrorPx ||
                rms > maximumReprojectionErrorPx * 1.5
            ) {
                return CameraRegistrationResult.rejected(
                    frameIndex,
                    correspondenceCount,
                    "Reprojection error exceeds pose quality limits",
                    inlierCount,
                    inlierRatio,
                    occupiedCells,
                    median,
                    rms
                )
            }

            CameraRegistrationResult(
                frameIndex = frameIndex,
                accepted = true,
                pose = CameraPose(
                    frameIndex,
                    rotation,
                    translation,
                    CameraPoseStatus.PNP_REGISTERED
                ),
                correspondenceCount = correspondenceCount,
                inlierCount = inlierCount,
                inlierRatio = inlierRatio,
                occupiedGridCells = occupiedCells,
                medianReprojectionErrorPx = median,
                rmsReprojectionErrorPx = rms,
                reason = if (refinement == null) {
                    "Registered against existing 3D map; used RANSAC PnP seed because robust refinement returned no candidate"
                } else {
                    "Registered against existing 3D map"
                }
            )
        } finally {
            objectPoints.release()
            imagePoints.release()
            distortion.release()
            rotationVector.release()
            translationVector.release()
            inlierIndices.release()
            inlierObjectPoints.release()
            inlierImagePoints.release()
            projectedPoints.release()
            rotationMatrix.release()
        }
    }

    private fun matrixReadSummary(matrix: Mat, readBytes: Int, expectedBytes: Int): String =
        "${matrix.rows()}x${matrix.cols()}, type=${matrix.type()}, depth=${matrix.depth()}, " +
            "channels=${matrix.channels()}, total=${matrix.total()}, " +
            "continuous=${matrix.isContinuous()}, read=$readBytes/$expectedBytes bytes"

    private fun toDoubleVector(source: Mat): Mat? {
        if (source.empty() || source.total() != 3L || source.channels() != 1) return null
        val converted = Mat()
        return try {
            source.convertTo(converted, CvType.CV_64F)
            if (converted.empty() || converted.total() != 3L ||
                converted.channels() != 1 || converted.type() != CvType.CV_64F
            ) {
                converted.release()
                null
            } else {
                converted
            }
        } catch (_: CvException) {
            converted.release()
            null
        }
    }

    private data class PoseModel(
        val rotationVector: DoubleArray,
        val translation: DoubleArray,
        val rotationMatrix: DoubleArray
    )

    private fun poseModel(rotationVector: DoubleArray, translation: DoubleArray): PoseModel? {
        if (rotationVector.size != 3 || translation.size != 3 ||
            rotationVector.any { !it.isFinite() } || translation.any { !it.isFinite() }
        ) return null
        val vector = Mat(3, 1, CvType.CV_64F)
        val matrix = Mat()
        return try {
            if (vector.put(0, 0, *rotationVector) != 3) return null
            Calib3d.Rodrigues(vector, matrix)
            val rotation = DoubleArray(9)
            val expectedRotationReadBytes = rotation.size * java.lang.Double.BYTES
            if (matrix.get(0, 0, rotation) != expectedRotationReadBytes ||
                rotation.any { !it.isFinite() }
            ) {
                return null
            }
            PoseModel(rotationVector.copyOf(), translation.copyOf(), rotation)
        } catch (_: CvException) {
            null
        } finally {
            vector.release()
            matrix.release()
        }
    }

    private fun fitPoseModel(
        current: PoseModel,
        indices: List<Int>,
        points3d: List<Point3>,
        points2d: List<Point>,
        cameraMatrix: Mat
    ): PoseModel? {
        if (indices.size < 4) return null
        val objectArray = indices.map(points3d::get).toTypedArray()
        val imageArray = indices.map(points2d::get).toTypedArray()
        val objectSubset = MatOfPoint3f()
        val imageSubset = MatOfPoint2f()
        val distortion = MatOfDouble()
        val rotationVector = Mat(3, 1, CvType.CV_64F)
        val translationVector = Mat(3, 1, CvType.CV_64F)
        return try {
            objectSubset.fromArray(*objectArray)
            imageSubset.fromArray(*imageArray)
            if (rotationVector.put(0, 0, *current.rotationVector) != 3 ||
                translationVector.put(0, 0, *current.translation) != 3
            ) return null
            val solved = Calib3d.solvePnP(
                objectSubset,
                imageSubset,
                cameraMatrix,
                distortion,
                rotationVector,
                translationVector,
                true,
                Calib3d.SOLVEPNP_ITERATIVE
            )
            if (!solved) return null
            val refinement = CameraPoseRefiner.refine(
                objectArray, imageArray, cameraMatrix, rotationVector, translationVector
            ) ?: return null
            poseModel(refinement.rotationVector, refinement.translation)
        } catch (_: CvException) {
            null
        } finally {
            objectSubset.release()
            imageSubset.release()
            distortion.release()
            rotationVector.release()
            translationVector.release()
        }
    }

    private fun reprojectionError(
        pose: PoseModel,
        point: Point3,
        observation: Point,
        intrinsics: PinholeIntrinsics
    ): Double {
        val r = pose.rotationMatrix
        val t = pose.translation
        val cameraPoint = doubleArrayOf(
            r[0] * point.x + r[1] * point.y + r[2] * point.z + t[0],
            r[3] * point.x + r[4] * point.y + r[5] * point.z + t[1],
            r[6] * point.x + r[7] * point.y + r[8] * point.z + t[2]
        )
        if (cameraPoint.any { !it.isFinite() } || cameraPoint[2] <= 1e-8) {
            return Double.POSITIVE_INFINITY
        }
        val projected = CameraProjectionMath.project(cameraPoint, intrinsics)
        return hypot(projected[0] - observation.x, projected[1] - observation.y)
    }

    private fun occupiedGridCells(
        points: Array<Point>,
        width: Int,
        height: Int
    ): Int {
        val occupied = BooleanArray(16)
        for (point in points) {
            val column = ((point.x / width) * 4.0).toInt().coerceIn(0, 3)
            val row = ((point.y / height) * 4.0).toInt().coerceIn(0, 3)
            occupied[row * 4 + column] = true
        }
        return occupied.count { it }
    }
}

internal object CameraRegistrationPolicy {
    const val MINIMUM_CORRESPONDENCES = 30
    const val MINIMUM_INLIER_RATIO = 0.25
    const val RANSAC_REPROJECTION_ERROR_PX = 12.0f
    const val RANSAC_MAX_ITERATIONS = 10000
    const val RANSAC_CONFIDENCE = 0.99999
    const val LOCAL_OPTIMIZATION_ROUNDS = 2
    const val FINAL_REPROJECTION_ERROR_PX = 4.0f
    const val MINIMUM_CHEIRALITY_RATIO = 0.90
    const val MINIMUM_OCCUPIED_GRID_CELLS = 4
}

internal data class CameraRegistrationResult(
    val frameIndex: Int,
    val accepted: Boolean,
    val pose: CameraPose?,
    val correspondenceCount: Int,
    val inlierCount: Int,
    val inlierRatio: Double,
    val occupiedGridCells: Int,
    val medianReprojectionErrorPx: Double,
    val rmsReprojectionErrorPx: Double,
    val reason: String
) {
    companion object {
        fun rejected(
            frameIndex: Int,
            correspondenceCount: Int,
            reason: String,
            inlierCount: Int = 0,
            inlierRatio: Double = 0.0,
            occupiedGridCells: Int = 0,
            medianError: Double = Double.POSITIVE_INFINITY,
            rmsError: Double = Double.POSITIVE_INFINITY
        ) = CameraRegistrationResult(
            frameIndex = frameIndex,
            accepted = false,
            pose = null,
            correspondenceCount = correspondenceCount,
            inlierCount = inlierCount,
            inlierRatio = inlierRatio,
            occupiedGridCells = occupiedGridCells,
            medianReprojectionErrorPx = medianError,
            rmsReprojectionErrorPx = rmsError,
            reason = reason
        )
    }
}
