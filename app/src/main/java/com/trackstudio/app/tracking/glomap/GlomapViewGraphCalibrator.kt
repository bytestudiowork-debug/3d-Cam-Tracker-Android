package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraIntrinsics

/**
 * Android in-memory adaptation of GLOMAP's pinned ViewGraphCalibrator.
 * Input fundamentals are row-major pixel-coordinate matrices in pair order.
 */
internal class GlomapViewGraphCalibrator {
    data class Result(
        val intrinsics: CameraIntrinsics,
        val initialFocalLength: Double,
        val estimatedFocalLength: Double,
        val focalLengthUpdated: Boolean,
        val initialCost: Double,
        val finalCost: Double,
        val iterations: Int,
        val acceptedPairIndices: Set<Int>,
        val rejectedPairIndices: Set<Int>
    )

    fun calibrate(
        fundamentals: List<DoubleArray>,
        initialIntrinsics: CameraIntrinsics,
        maxIterations: Int = 100,
        numThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
    ): Result? {
        require(maxIterations > 0 && numThreads > 0)
        if (fundamentals.size < MINIMUM_PAIR_COUNT) return null
        require(fundamentals.all { matrix ->
            matrix.size == 9 && matrix.all { it.isFinite() }
        }) {
            "Each fundamental matrix must contain nine finite row-major values."
        }

        val focal = 0.5 * (initialIntrinsics.fx + initialIntrinsics.fy)
        if (!focal.isFinite() || focal <= 0.0) return null
        val packedFundamentals = DoubleArray(fundamentals.size * 9)
        fundamentals.forEachIndexed { index, matrix ->
            System.arraycopy(matrix, 0, packedFundamentals, index * 9, 9)
        }
        val packed = nativeSolve(
            packedFundamentals,
            initialIntrinsics.cx,
            initialIntrinsics.cy,
            focal,
            maxIterations,
            numThreads
        ) ?: return null
        if (packed.size != RESULT_HEADER_SIZE + fundamentals.size || packed[0] != 1.0 ||
            !packed[1].isFinite() || !packed[2].isFinite() ||
            !packed[4].isFinite() || packed[4] <= 0.0
        ) return null

        val residuals = packed.copyOfRange(RESULT_HEADER_SIZE, packed.size)
        val accepted = residuals.indices.filterTo(LinkedHashSet<Int>()) {
            residuals[it].isFinite() && residuals[it] <= MAX_TWO_VIEW_ERROR
        }
        val rejected = residuals.indices.filterTo(LinkedHashSet<Int>()) { it !in accepted }
        val solvedFocal = packed[4]
        val focalRatio = solvedFocal / focal
        val focalLengthUpdated = focalRatio.isFinite() &&
            focalRatio in MIN_FOCAL_RATIO..MAX_FOCAL_RATIO
        return Result(
            intrinsics = if (focalLengthUpdated) {
                initialIntrinsics.copy(fx = solvedFocal, fy = solvedFocal)
            } else {
                initialIntrinsics
            },
            initialFocalLength = focal,
            estimatedFocalLength = solvedFocal,
            focalLengthUpdated = focalLengthUpdated,
            initialCost = packed[1],
            finalCost = packed[2],
            iterations = packed[3].toInt().coerceAtLeast(0),
            acceptedPairIndices = accepted,
            rejectedPairIndices = rejected
        )
    }

    private external fun nativeSolve(
        fundamentalMatrices: DoubleArray,
        principalX: Double,
        principalY: Double,
        initialFocalLength: Double,
        maxNumIterations: Int,
        numThreads: Int
    ): DoubleArray?

    companion object {
        private const val RESULT_HEADER_SIZE = 5
        private const val MINIMUM_PAIR_COUNT = 1
        private const val MAX_TWO_VIEW_ERROR = 2.0
        private const val MIN_FOCAL_RATIO = 0.1
        private const val MAX_FOCAL_RATIO = 10.0

        init {
            System.loadLibrary("trackstudio_reconstruction")
        }
    }
}
