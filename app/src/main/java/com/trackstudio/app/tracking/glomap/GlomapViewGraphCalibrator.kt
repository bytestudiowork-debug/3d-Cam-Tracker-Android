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

    data class CameraInput(
        val cameraId: Int,
        val intrinsics: CameraIntrinsics,
        val hasPriorFocalLength: Boolean = false
    )

    data class FundamentalPair(
        val fundamental: DoubleArray,
        val firstCameraId: Int,
        val secondCameraId: Int
    )

    data class MultiCameraResult(
        val intrinsicsByCameraId: Map<Int, CameraIntrinsics>,
        val initialFocalLengthsByCameraId: Map<Int, Double>,
        val estimatedFocalLengthsByCameraId: Map<Int, Double>,
        val focalLengthUpdatedCameraIds: Set<Int>,
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

    fun calibrateMultiCamera(
        pairs: List<FundamentalPair>,
        cameras: List<CameraInput>,
        maxIterations: Int = 100,
        numThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
    ): MultiCameraResult? {
        require(maxIterations > 0 && numThreads > 0)
        if (pairs.isEmpty() || cameras.isEmpty()) return null
        require(pairs.size <= Int.MAX_VALUE / 9 && cameras.size <= Int.MAX_VALUE / 2)
        require(cameras.map { it.cameraId }.toSet().size == cameras.size) {
            "Camera IDs must be unique."
        }
        require(pairs.all { pair ->
            pair.fundamental.size == 9 && pair.fundamental.all { it.isFinite() }
        }) {
            "Each fundamental matrix must contain nine finite row-major values."
        }

        val cameraIndexById = cameras.mapIndexed { index, camera -> camera.cameraId to index }.toMap()
        val pairCameraIndices = IntArray(pairs.size * 2)
        val packedFundamentals = DoubleArray(pairs.size * 9)
        val participates = BooleanArray(cameras.size)
        pairs.forEachIndexed { pairIndex, pair ->
            val firstIndex = cameraIndexById[pair.firstCameraId]
                ?: throw IllegalArgumentException("Pair references unknown camera ${pair.firstCameraId}.")
            val secondIndex = cameraIndexById[pair.secondCameraId]
                ?: throw IllegalArgumentException("Pair references unknown camera ${pair.secondCameraId}.")
            pairCameraIndices[pairIndex * 2] = firstIndex
            pairCameraIndices[pairIndex * 2 + 1] = secondIndex
            participates[firstIndex] = true
            participates[secondIndex] = true
            System.arraycopy(pair.fundamental, 0, packedFundamentals, pairIndex * 9, 9)
        }

        val initialFocals = DoubleArray(cameras.size)
        val principalPoints = DoubleArray(cameras.size * 2)
        val hasPriorFocalLength = BooleanArray(cameras.size)
        cameras.forEachIndexed { index, camera ->
            val focal = 0.5 * (camera.intrinsics.fx + camera.intrinsics.fy)
            if (!focal.isFinite() || focal <= 0.0 ||
                !camera.intrinsics.cx.isFinite() || !camera.intrinsics.cy.isFinite()
            ) return null
            initialFocals[index] = focal
            principalPoints[index * 2] = camera.intrinsics.cx
            principalPoints[index * 2 + 1] = camera.intrinsics.cy
            hasPriorFocalLength[index] = camera.hasPriorFocalLength
        }

        if (cameras.indices.none { participates[it] && !hasPriorFocalLength[it] }) {
            val initialById = cameras.associate { camera ->
                camera.cameraId to 0.5 * (camera.intrinsics.fx + camera.intrinsics.fy)
            }
            return MultiCameraResult(
                intrinsicsByCameraId = cameras.associate { it.cameraId to it.intrinsics },
                initialFocalLengthsByCameraId = initialById,
                estimatedFocalLengthsByCameraId = initialById,
                focalLengthUpdatedCameraIds = emptySet(),
                initialCost = 0.0,
                finalCost = 0.0,
                iterations = 0,
                acceptedPairIndices = pairs.indices.toSet(),
                rejectedPairIndices = emptySet()
            )
        }

        val packed = nativeSolveMulti(
            packedFundamentals,
            pairCameraIndices,
            principalPoints,
            initialFocals,
            hasPriorFocalLength,
            maxIterations,
            numThreads
        ) ?: return null
        val expectedSize = MULTI_RESULT_HEADER_SIZE.toLong() + cameras.size + pairs.size
        if (expectedSize > Int.MAX_VALUE || packed.size.toLong() != expectedSize ||
            packed[0] != 1.0 || !packed[1].isFinite() || !packed[2].isFinite() ||
            !packed[3].isFinite() || !packed[4].isFinite() || !packed[5].isFinite() ||
            packed[4].toInt() != cameras.size || packed[5].toInt() != pairs.size
        ) return null

        val estimatedFocals = packed.copyOfRange(
            MULTI_RESULT_HEADER_SIZE,
            MULTI_RESULT_HEADER_SIZE + cameras.size
        )
        if (estimatedFocals.any { !it.isFinite() || it <= 0.0 }) return null
        val initialById = LinkedHashMap<Int, Double>(cameras.size)
        val estimatedById = LinkedHashMap<Int, Double>(cameras.size)
        val intrinsicsById = LinkedHashMap<Int, CameraIntrinsics>(cameras.size)
        val updatedCameraIds = LinkedHashSet<Int>()
        cameras.forEachIndexed { index, camera ->
            val initial = initialFocals[index]
            val estimated = estimatedFocals[index]
            val ratio = estimated / initial
            val withinSourceBounds = ratio.isFinite() && ratio in MIN_FOCAL_RATIO..MAX_FOCAL_RATIO
            initialById[camera.cameraId] = initial
            estimatedById[camera.cameraId] = estimated
            intrinsicsById[camera.cameraId] = if (withinSourceBounds) {
                if (kotlin.math.abs(ratio - 1.0) > FOCAL_UPDATE_EPSILON &&
                    !camera.hasPriorFocalLength
                ) {
                    updatedCameraIds += camera.cameraId
                }
                camera.intrinsics.copy(fx = estimated, fy = estimated)
            } else {
                camera.intrinsics
            }
        }

        val residuals = packed.copyOfRange(MULTI_RESULT_HEADER_SIZE + cameras.size, packed.size)
        val accepted = residuals.indices.filterTo(LinkedHashSet<Int>()) {
            residuals[it].isFinite() && residuals[it] <= MAX_TWO_VIEW_ERROR
        }
        val rejected = residuals.indices.filterTo(LinkedHashSet<Int>()) { it !in accepted }
        return MultiCameraResult(
            intrinsicsByCameraId = intrinsicsById,
            initialFocalLengthsByCameraId = initialById,
            estimatedFocalLengthsByCameraId = estimatedById,
            focalLengthUpdatedCameraIds = updatedCameraIds,
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

    private external fun nativeSolveMulti(
        fundamentalMatrices: DoubleArray,
        pairCameraIndices: IntArray,
        principalPoints: DoubleArray,
        initialFocalLengths: DoubleArray,
        hasPriorFocalLength: BooleanArray,
        maxNumIterations: Int,
        numThreads: Int
    ): DoubleArray?

    companion object {
        private const val RESULT_HEADER_SIZE = 5
        private const val MULTI_RESULT_HEADER_SIZE = 6
        private const val MINIMUM_PAIR_COUNT = 1
        private const val MAX_TWO_VIEW_ERROR = 2.0
        private const val MIN_FOCAL_RATIO = 0.1
        private const val MAX_FOCAL_RATIO = 10.0
        private const val FOCAL_UPDATE_EPSILON = 1e-12

        init {
            System.loadLibrary("trackstudio_reconstruction")
        }
    }
}
