package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.PinholeIntrinsics
import kotlin.math.abs

/** Android bridge for COLMAP's pinned focal-unknown P4PF absolute-pose path. */
internal object ColmapP4pfAbsolutePoseEstimator {
    data class Estimate(
        val rotation: DoubleArray,
        val translation: DoubleArray,
        val inlierIndices: List<Int>,
        val initialFocalLength: Double,
        val refinedFx: Double,
        val refinedFy: Double,
        val initialCost: Double,
        val finalCost: Double,
        val trials: Int,
        val refinementIterations: Int
    )

    fun estimate(
        worldPointsXYZ: DoubleArray,
        imagePointsXY: DoubleArray,
        initialIntrinsics: PinholeIntrinsics,
        maxErrorPx: Double,
        maxTrials: Int = ANDROID_MAX_TRIALS,
        randomSeed: Int = -1
    ): Estimate? {
        if (worldPointsXYZ.size < MINIMUM_SAMPLE_SIZE * 3 || worldPointsXYZ.size % 3 != 0 ||
            imagePointsXY.size != worldPointsXYZ.size / 3 * 2 ||
            worldPointsXYZ.any { !it.isFinite() } || imagePointsXY.any { !it.isFinite() } ||
            !isValid(initialIntrinsics) || !maxErrorPx.isFinite() || maxErrorPx <= 0.0 ||
            maxTrials !in PINNED_MINIMUM_TRIALS..ANDROID_MAX_TRIALS || randomSeed < -1
        ) return null

        val packed = nativeEstimate(
            worldPointsXYZ,
            imagePointsXY,
            doubleArrayOf(
                initialIntrinsics.fx,
                initialIntrinsics.fy,
                initialIntrinsics.skew,
                initialIntrinsics.cx,
                initialIntrinsics.cy
            ),
            doubleArrayOf(
                maxErrorPx,
                PINNED_MINIMUM_INLIER_RATIO,
                PINNED_CONFIDENCE,
                PINNED_DYNAMIC_TRIAL_MULTIPLIER,
                PINNED_MINIMUM_TRIALS.toDouble(),
                maxTrials.toDouble(),
                PINNED_REFINEMENT_LOSS_SCALE,
                PINNED_REFINEMENT_GRADIENT_TOLERANCE
            ),
            randomSeed
        ) ?: return null

        if (packed.size < RESULT_HEADER_SIZE || packed[0] != 1.0 ||
            packed.take(RESULT_HEADER_SIZE).any { !it.isFinite() }
        ) return null
        val trials = packed[1].toIntOrNull() ?: return null
        val refinementIterations = packed[7].toIntOrNull() ?: return null
        val inlierCount = packed[8].toIntOrNull() ?: return null
        val pointCount = worldPointsXYZ.size / 3
        if (packed.size != RESULT_HEADER_SIZE + inlierCount ||
            inlierCount < MINIMUM_SAMPLE_SIZE || inlierCount > pointCount ||
            trials !in PINNED_MINIMUM_TRIALS..maxTrials ||
            refinementIterations !in 0..PINNED_MAX_REFINEMENT_ITERATIONS ||
            packed[2] <= 0.0 || packed[3] <= 0.0 || packed[4] <= 0.0
        ) return null

        val rotation = packed.copyOfRange(9, 18)
        if (!isProperRotation(rotation)) return null
        val translation = packed.copyOfRange(18, 21)
        if (translation.any { !it.isFinite() }) return null
        val inliers = ArrayList<Int>(inlierCount)
        val seen = BooleanArray(pointCount)
        for (index in 0 until inlierCount) {
            val value = packed[RESULT_HEADER_SIZE + index]
            if (!value.isFinite() || value < 0.0 || value % 1.0 != 0.0 || value >= pointCount) {
                return null
            }
            val inlier = value.toInt()
            if (seen[inlier]) return null
            seen[inlier] = true
            inliers += inlier
        }
        return Estimate(
            rotation = rotation,
            translation = translation,
            inlierIndices = inliers,
            initialFocalLength = packed[2],
            refinedFx = packed[3],
            refinedFy = packed[4],
            initialCost = packed[5],
            finalCost = packed[6],
            trials = trials,
            refinementIterations = refinementIterations
        )
    }

    private fun isValid(intrinsics: PinholeIntrinsics): Boolean =
        intrinsics.fx.isFinite() && intrinsics.fx > 0.0 &&
            intrinsics.fy.isFinite() && intrinsics.fy > 0.0 &&
            intrinsics.skew.isFinite() && abs(intrinsics.skew) <= 1e-9 &&
            intrinsics.cx.isFinite() && intrinsics.cy.isFinite()

    private fun isProperRotation(rotation: DoubleArray): Boolean {
        if (rotation.size != 9 || rotation.any { !it.isFinite() }) return false
        for (rowA in 0 until 3) {
            for (rowB in 0 until 3) {
                var dot = 0.0
                for (column in 0 until 3) {
                    dot += rotation[rowA * 3 + column] * rotation[rowB * 3 + column]
                }
                val expected = if (rowA == rowB) 1.0 else 0.0
                if (abs(dot - expected) > ROTATION_ORTHOGONALITY_TOLERANCE) return false
            }
        }
        val determinant =
            rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
                rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
                rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
        return abs(determinant - 1.0) <= ROTATION_ORTHOGONALITY_TOLERANCE
    }

    private fun Double.toIntOrNull(): Int? =
        if (isFinite() && this >= 0.0 && this % 1.0 == 0.0 && this <= Int.MAX_VALUE) {
            toInt()
        } else {
            null
        }

    private external fun nativeEstimate(
        worldPointsXYZ: DoubleArray,
        imagePointsXY: DoubleArray,
        intrinsics: DoubleArray,
        options: DoubleArray,
        randomSeed: Int
    ): DoubleArray?

    private const val RESULT_HEADER_SIZE = 21
    private const val MINIMUM_SAMPLE_SIZE = 4
    private const val PINNED_MINIMUM_TRIALS = 100
    private const val PINNED_MINIMUM_INLIER_RATIO = 0.1
    private const val PINNED_CONFIDENCE = 0.99999
    private const val PINNED_DYNAMIC_TRIAL_MULTIPLIER = 3.0
    private const val PINNED_REFINEMENT_LOSS_SCALE = 1.0
    private const val PINNED_REFINEMENT_GRADIENT_TOLERANCE = 1.0
    private const val PINNED_MAX_REFINEMENT_ITERATIONS = 100
    const val PINNED_MAX_TRIALS = 10_000
    const val ANDROID_MAX_TRIALS = 10_000
    private const val ROTATION_ORTHOGONALITY_TOLERANCE = 1e-5

    init {
        System.loadLibrary("trackstudio_reconstruction")
    }
}
