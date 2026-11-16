package com.trackstudio.app.tracking

import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Point3

/** Fixed-intrinsics P3P/EPnP LO-RANSAC estimator used for incremental registration. */
internal object AbsolutePoseLORansacEstimator {
    data class Result(
        val rotation: DoubleArray,
        val translation: DoubleArray,
        val inlierIndices: List<Int>,
        val numTrials: Int,
        val numLocalTrials: Int,
        val inlierResidualSum: Double
    )

    fun estimate(
        objectPoints: Array<Point3>,
        imagePoints: Array<Point>,
        cameraMatrix: Mat,
        maxErrorPx: Float,
        randomSeed: Int = -1
    ): Result? {
        if (objectPoints.size < MINIMAL_SAMPLE_SIZE || objectPoints.size != imagePoints.size ||
            cameraMatrix.empty() || cameraMatrix.rows() != 3 || cameraMatrix.cols() != 3 ||
            !maxErrorPx.isFinite() || maxErrorPx <= 0f || randomSeed < -1
        ) return null

        val world = DoubleArray(objectPoints.size * 3)
        val image = DoubleArray(imagePoints.size * 2)
        objectPoints.forEachIndexed { index, point ->
            world[index * 3] = point.x
            world[index * 3 + 1] = point.y
            world[index * 3 + 2] = point.z
        }
        imagePoints.forEachIndexed { index, point ->
            image[index * 2] = point.x
            image[index * 2 + 1] = point.y
        }
        val intrinsics = doubleArrayOf(
            cameraMatrix.get(0, 0)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 2)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 2)?.firstOrNull() ?: Double.NaN
        )
        if (world.any { !it.isFinite() } || image.any { !it.isFinite() } ||
            intrinsics.any { !it.isFinite() } || intrinsics[0] <= 0.0 || intrinsics[3] <= 0.0
        ) return null

        val options = doubleArrayOf(
            maxErrorPx.toDouble(),
            MIN_INLIER_RATIO,
            CONFIDENCE,
            DYNAMIC_TRIAL_MULTIPLIER,
            MIN_NUM_TRIALS.toDouble(),
            MAX_NUM_TRIALS.toDouble(),
            MAX_LOCAL_TRIALS.toDouble()
        )
        val packed = nativeEstimate(world, image, intrinsics, options, randomSeed) ?: return null
        if (packed.size < RESULT_HEADER_SIZE || packed[0] != 1.0 ||
            packed.take(16).drop(1).any { !it.isFinite() }
        ) return null
        val inlierCountValue = packed[16]
        if (!inlierCountValue.isFinite() || inlierCountValue < MINIMAL_SAMPLE_SIZE ||
            inlierCountValue != inlierCountValue.toInt().toDouble()
        ) return null
        val inlierCount = inlierCountValue.toInt()
        if (packed.size != RESULT_HEADER_SIZE + inlierCount) return null
        val inliers = (0 until inlierCount).map { packed[RESULT_HEADER_SIZE + it].toInt() }
        if (inliers.any { it !in objectPoints.indices } || inliers.distinct().size != inliers.size) {
            return null
        }
        return Result(
            rotation = packed.copyOfRange(4, 13),
            translation = packed.copyOfRange(13, 16),
            inlierIndices = inliers,
            numTrials = packed[1].toInt(),
            numLocalTrials = packed[2].toInt(),
            inlierResidualSum = packed[3]
        )
    }

    private external fun nativeEstimate(
        objectPoints: DoubleArray,
        imagePoints: DoubleArray,
        intrinsics: DoubleArray,
        options: DoubleArray,
        randomSeed: Int
    ): DoubleArray?

    init {
        System.loadLibrary("trackstudio_camera")
    }

    private const val MINIMAL_SAMPLE_SIZE = 3
    private const val MIN_INLIER_RATIO = CameraRegistrationPolicy.RANSAC_MIN_INLIER_RATIO
    private const val CONFIDENCE = CameraRegistrationPolicy.RANSAC_CONFIDENCE
    private const val DYNAMIC_TRIAL_MULTIPLIER = CameraRegistrationPolicy.RANSAC_DYNAMIC_TRIAL_MULTIPLIER
    private const val MIN_NUM_TRIALS = CameraRegistrationPolicy.RANSAC_MIN_NUM_TRIALS
    private const val MAX_NUM_TRIALS = CameraRegistrationPolicy.RANSAC_MAX_ITERATIONS
    private const val MAX_LOCAL_TRIALS = CameraRegistrationPolicy.RANSAC_MAX_LOCAL_TRIALS
    private const val RESULT_HEADER_SIZE = 17
}
