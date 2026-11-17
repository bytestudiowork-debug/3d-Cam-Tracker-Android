package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraProjectionMath
import com.trackstudio.app.tracking.PinholeIntrinsics
import kotlin.math.cos
import kotlin.math.sqrt

/**
 * Android in-memory port of GLOMAP's calibrated essential-image-pair inlier
 * scoring. Inputs are fixed-pinhole pixel coordinates; lens distortion is not
 * modeled by the Android reconstruction path.
 */
internal object GlomapImagePairInlierScorer {
    data class Result(
        val inlierIndices: List<Int>,
        val score: Double,
        val squaredThreshold: Double
    )

    const val DEFAULT_MAX_EPIPOLAR_ERROR_PX = 1.0
    const val MINIMUM_INLIERS = 30
    const val MINIMUM_INLIER_RATIO = 0.25

    private const val EPS = 1e-12
    private const val MIN_CHEIRALITY_DEPTH = 1e-2
    private const val MAX_CHEIRALITY_DEPTH = 100.0
    private const val TRIANGULATION_ANGLE_DOT_THRESHOLD = 1.0 + 1e-6
    private val EPIPOLE_DOT_THRESHOLD = cos(Math.toRadians(3.0)) + 1e-6

    /**
     * Scores corresponding raw-match rows in their original order. `rotation`
     * and `translation` describe camera2_from_camera1; `translation` is the
     * unit direction returned by relative-pose estimation.
     */
    fun scoreEssential(
        firstPixels: List<Pair<Double, Double>>,
        secondPixels: List<Pair<Double, Double>>,
        firstIntrinsics: PinholeIntrinsics,
        secondIntrinsics: PinholeIntrinsics,
        rotation: DoubleArray,
        translation: DoubleArray,
        maxEpipolarErrorPx: Double = DEFAULT_MAX_EPIPOLAR_ERROR_PX
    ): Result? {
        if (firstPixels.size != secondPixels.size ||
            rotation.size != 9 || translation.size != 3 ||
            rotation.any { !it.isFinite() } || translation.any { !it.isFinite() } ||
            !isRotationMatrix(rotation) ||
            !isValidIntrinsics(firstIntrinsics) || !isValidIntrinsics(secondIntrinsics) ||
            !maxEpipolarErrorPx.isFinite() || maxEpipolarErrorPx <= 0.0 ||
            firstPixels.any { !it.first.isFinite() || !it.second.isFinite() } ||
            secondPixels.any { !it.first.isFinite() || !it.second.isFinite() }
        ) return null

        val focal1 = 0.5 * (firstIntrinsics.fx + firstIntrinsics.fy)
        val focal2 = 0.5 * (secondIntrinsics.fx + secondIntrinsics.fy)
        val threshold = maxEpipolarErrorPx * 0.5 * (1.0 / focal1 + 1.0 / focal2)
        val squaredThreshold = threshold * threshold
        if (!squaredThreshold.isFinite() || squaredThreshold <= 0.0) return null

        val rays1 = firstPixels.map { unitRay(it, firstIntrinsics) ?: return null }
        val rays2 = secondPixels.map { unitRay(it, secondIntrinsics) ?: return null }
        val essential = essentialFromMotion(rotation, translation)

        val epipole12 = translation.copyOf()
        if (epipole12[2] < 0.0) negate(epipole12)
        val epipole21 = transposeMultiply(rotation, translation).map { -it }.toDoubleArray()
        if (epipole21[2] < 0.0) negate(epipole21)

        val inliers = ArrayList<Int>()
        var score = 0.0
        for (index in rays1.indices) {
            val ray1 = rays1[index]
            val ray2 = rays2[index]
            val error = sampsonError(essential, ray1, ray2)
            val accepted = error < squaredThreshold &&
                checkCheirality(rotation, translation, ray1, ray2) &&
                relativeRayCosine(rotation, ray1, ray2) < TRIANGULATION_ANGLE_DOT_THRESHOLD &&
                dot(ray1, epipole21) < EPIPOLE_DOT_THRESHOLD &&
                dot(ray2, epipole12) < EPIPOLE_DOT_THRESHOLD
            if (accepted) {
                inliers += index
                score += error
            } else {
                score += squaredThreshold
            }
        }
        return Result(inliers, score, squaredThreshold)
    }

    private fun unitRay(pixel: Pair<Double, Double>, intrinsics: PinholeIntrinsics): DoubleArray? {
        val ray = CameraProjectionMath.pixelToNormalized(pixel.first, pixel.second, intrinsics)
        val norm = sqrt(dot(ray, ray))
        if (!norm.isFinite() || norm <= 0.0) return null
        return DoubleArray(3) { ray[it] / norm }
    }

    private fun essentialFromMotion(rotation: DoubleArray, translation: DoubleArray): DoubleArray {
        val tx = doubleArrayOf(
            0.0, -translation[2], translation[1],
            translation[2], 0.0, -translation[0],
            -translation[1], translation[0], 0.0
        )
        return DoubleArray(9) { index ->
            val row = index / 3
            val column = index % 3
            tx[row * 3] * rotation[column] +
                tx[row * 3 + 1] * rotation[3 + column] +
                tx[row * 3 + 2] * rotation[6 + column]
        }
    }

    /** Exact pinned GLOMAP 3D-ray Sampson-error form. */
    private fun sampsonError(essential: DoubleArray, ray1: DoubleArray, ray2: DoubleArray): Double {
        val z1 = EPS + ray1[2]
        val z2 = EPS + ray2[2]
        val ex1x = (essential[0] * ray1[0] + essential[1] * ray1[1] + essential[2] * ray1[2]) / z1
        val ex1y = (essential[3] * ray1[0] + essential[4] * ray1[1] + essential[5] * ray1[2]) / z1
        val ex1z = (essential[6] * ray1[0] + essential[7] * ray1[1] + essential[8] * ray1[2]) / z1
        val etx2x = (essential[0] * ray2[0] + essential[3] * ray2[1] + essential[6] * ray2[2]) / z2
        val etx2y = (essential[1] * ray2[0] + essential[4] * ray2[1] + essential[7] * ray2[2]) / z2
        val constraint = ex1x * ray2[0] + ex1y * ray2[1] + ex1z * ray2[2]
        val denominator = ex1x * ex1x + ex1y * ex1y + etx2x * etx2x + etx2y * etx2y
        val error = constraint * constraint / denominator
        return if (error.isFinite()) error else Double.POSITIVE_INFINITY
    }

    /**
     * Port of GLOMAP's PoseLib-derived CheckCheirality for unit feature rays,
     * including its dropped positive factor in the two-depth solve.
     */
    private fun checkCheirality(
        rotation: DoubleArray,
        translation: DoubleArray,
        ray1: DoubleArray,
        ray2: DoubleArray
    ): Boolean {
        val rotatedX = rotation[0] * ray1[0] + rotation[1] * ray1[1] + rotation[2] * ray1[2]
        val rotatedY = rotation[3] * ray1[0] + rotation[4] * ray1[1] + rotation[5] * ray1[2]
        val rotatedZ = rotation[6] * ray1[0] + rotation[7] * ray1[1] + rotation[8] * ray1[2]
        val a = -(rotatedX * ray2[0] + rotatedY * ray2[1] + rotatedZ * ray2[2])
        val b1 = -(rotatedX * translation[0] + rotatedY * translation[1] + rotatedZ * translation[2])
        val b2 = dot(ray2, translation)
        val lambda1 = b1 - a * b2
        val lambda2 = -a * b1 + b2
        val depthScale = 1.0 - a * a
        if (!depthScale.isFinite() || depthScale <= 0.0) return false
        val minDepth = MIN_CHEIRALITY_DEPTH * depthScale
        val maxDepth = MAX_CHEIRALITY_DEPTH * depthScale
        return lambda1.isFinite() && lambda2.isFinite() &&
            lambda1 > minDepth && lambda2 > minDepth &&
            lambda1 < maxDepth && lambda2 < maxDepth
    }

    private fun relativeRayCosine(
        rotation: DoubleArray,
        ray1: DoubleArray,
        ray2: DoubleArray
    ): Double {
        val inverseRotatedX = rotation[0] * ray2[0] + rotation[3] * ray2[1] + rotation[6] * ray2[2]
        val inverseRotatedY = rotation[1] * ray2[0] + rotation[4] * ray2[1] + rotation[7] * ray2[2]
        val inverseRotatedZ = rotation[2] * ray2[0] + rotation[5] * ray2[1] + rotation[8] * ray2[2]
        return ray1[0] * inverseRotatedX + ray1[1] * inverseRotatedY + ray1[2] * inverseRotatedZ
    }

    private fun isValidIntrinsics(intrinsics: PinholeIntrinsics): Boolean =
        intrinsics.fx.isFinite() && intrinsics.fx > 0.0 &&
            intrinsics.fy.isFinite() && intrinsics.fy > 0.0 &&
            intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite()

    private fun isRotationMatrix(rotation: DoubleArray): Boolean {
        for (row in 0..2) for (column in 0..2) {
            var value = 0.0
            for (k in 0..2) value += rotation[row * 3 + k] * rotation[column * 3 + k]
            if (kotlin.math.abs(value - if (row == column) 1.0 else 0.0) > 1e-3) return false
        }
        val determinant =
            rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
                rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
                rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
        return determinant.isFinite() && kotlin.math.abs(determinant - 1.0) <= 1e-3
    }

    private fun transposeMultiply(matrix: DoubleArray, vector: DoubleArray): DoubleArray = DoubleArray(3) { column ->
        matrix[column] * vector[0] + matrix[3 + column] * vector[1] + matrix[6 + column] * vector[2]
    }

    private fun dot(first: DoubleArray, second: DoubleArray): Double =
        first[0] * second[0] + first[1] * second[1] + first[2] * second[2]

    private fun negate(vector: DoubleArray) {
        for (index in vector.indices) vector[index] = -vector[index]
    }
}
