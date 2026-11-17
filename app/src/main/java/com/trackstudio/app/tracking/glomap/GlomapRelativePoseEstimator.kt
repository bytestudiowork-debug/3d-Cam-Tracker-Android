package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.PinholeIntrinsics

/** Android bridge for GLOMAP's pinned PoseLib calibrated relative-pose estimator. */
internal object GlomapRelativePoseEstimator {
    data class Estimate(
        val rotation: DoubleArray,
        val translation: DoubleArray,
        val inlierIndices: List<Int>,
        val iterations: Int
    )

    fun estimate(
        firstPixelsXY: DoubleArray,
        secondPixelsXY: DoubleArray,
        firstWidth: Int,
        firstHeight: Int,
        secondWidth: Int,
        secondHeight: Int,
        firstIntrinsics: PinholeIntrinsics,
        secondIntrinsics: PinholeIntrinsics,
        maxIterations: Int = ANDROID_MAX_ITERATIONS
    ): Estimate? {
        if (firstPixelsXY.size != secondPixelsXY.size ||
            firstPixelsXY.size % 2 != 0 || firstPixelsXY.size < MINIMUM_SAMPLE_SIZE * 2 ||
            firstWidth <= 0 || firstHeight <= 0 || secondWidth <= 0 || secondHeight <= 0 ||
            maxIterations <= 0 || firstPixelsXY.any { !it.isFinite() } ||
            secondPixelsXY.any { !it.isFinite() } ||
            !isValid(firstIntrinsics) || !isValid(secondIntrinsics)
        ) return null

        val packed = nativeEstimate(
            firstPixelsXY,
            secondPixelsXY,
            doubleArrayOf(
                firstIntrinsics.fx, firstIntrinsics.fy, firstIntrinsics.skew,
                firstIntrinsics.cx, firstIntrinsics.cy
            ),
            intArrayOf(firstWidth, firstHeight),
            doubleArrayOf(
                secondIntrinsics.fx, secondIntrinsics.fy, secondIntrinsics.skew,
                secondIntrinsics.cx, secondIntrinsics.cy
            ),
            intArrayOf(secondWidth, secondHeight),
            maxIterations
        ) ?: return null

        val numMatches = firstPixelsXY.size / 2
        if (packed.size != RESULT_HEADER_SIZE + numMatches || packed[0] != 1.0 ||
            packed.take(RESULT_HEADER_SIZE).any { !it.isFinite() }
        ) return null
        val inlierCount = packed[13].toInt()
        val iterations = packed[14].toInt()
        if (inlierCount < MINIMUM_SAMPLE_SIZE || iterations < 0) return null
        val inliers = (0 until numMatches).filter { packed[RESULT_HEADER_SIZE + it] != 0.0 }
        if (inliers.size != inlierCount) return null
        return Estimate(
            rotation = packed.copyOfRange(1, 10),
            translation = packed.copyOfRange(10, 13),
            inlierIndices = inliers,
            iterations = iterations
        )
    }

    private fun isValid(intrinsics: PinholeIntrinsics): Boolean =
        intrinsics.fx.isFinite() && intrinsics.fx > 0.0 &&
            intrinsics.fy.isFinite() && intrinsics.fy > 0.0 &&
            intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite()

    private external fun nativeEstimate(
        firstPixelsXY: DoubleArray,
        secondPixelsXY: DoubleArray,
        camera1Values: DoubleArray,
        camera1Dimensions: IntArray,
        camera2Values: DoubleArray,
        camera2Dimensions: IntArray,
        maxIterations: Int
    ): DoubleArray?

    private const val RESULT_HEADER_SIZE = 15
    private const val MINIMUM_SAMPLE_SIZE = 5
    const val PINNED_MAX_ITERATIONS = 50_000
    const val ANDROID_MAX_ITERATIONS = 30_000

    init {
        System.loadLibrary("trackstudio_reconstruction")
    }
}
