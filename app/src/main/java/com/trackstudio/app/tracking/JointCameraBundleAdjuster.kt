package com.trackstudio.app.tracking

import org.opencv.core.Mat

/** Joint robust reprojection refinement for a connected incremental camera map. */
internal class JointCameraBundleAdjuster {
    data class Result(
        val frameCentersById: Map<Int, DoubleArray>,
        val rotationsByFrame: Map<Int, DoubleArray>,
        val pointsByTrackId: Map<Int, DoubleArray>,
        val intrinsics: CameraIntrinsics,
        val initialCost: Double,
        val finalCost: Double,
        val positiveDepthRatio: Double,
        val iterations: Int,
        val principalPointRefinementEnabled: Boolean
    )

    private data class ObservationRef(
        val frameIndex: Int,
        val trackIndex: Int,
        val observation: CameraFeatureObservation
    )

    fun solve(
        frameCentersById: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        tracks: List<CameraFeatureTrack>,
        cameraMatrix: Mat,
        imageWidth: Int,
        imageHeight: Int,
        maxIterations: Int = 45,
        numThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
        optimizeFocalLength: Boolean = true,
        optimizePointTrackIds: Set<Int>? = null
    ): Result? {
        require(maxIterations > 0 && numThreads > 0)
        if (cameraMatrix.empty() || cameraMatrix.rows() != 3 || cameraMatrix.cols() != 3) return null

        val observedFrames = tracks.asSequence()
            .flatMap { track -> track.observations.asSequence().map { it.frameIndex } }
            .toSet()
        val frameIds = frameCentersById.keys
            .filter { rotationsByFrame.containsKey(it) && it in observedFrames }
            .sorted()
        val activeFrameSet = frameIds.toSet()
        val trackIds = tracks.asSequence()
            .filter { pointsByTrackId.containsKey(it.id) }
            .filter { track -> track.observations.count { it.frameIndex in activeFrameSet } >= 3 }
            .map { it.id }
            .distinct()
            .sorted()
            .toList()
        if (frameIds.size < 3 || trackIds.size < 20) return null

        val frameIndexById = frameIds.withIndex().associate { it.value to it.index }
        val trackIndexById = trackIds.withIndex().associate { it.value to it.index }
        if (optimizePointTrackIds != null && !trackIndexById.keys.containsAll(optimizePointTrackIds)) {
            return null
        }
        val observations = tracks.asSequence()
            .filter { it.id in trackIndexById }
            .flatMap { track ->
                track.observations.asSequence().mapNotNull { observation ->
                    val frameIndex = frameIndexById[observation.frameIndex] ?: return@mapNotNull null
                    ObservationRef(frameIndex, trackIndexById.getValue(track.id), observation)
                }
            }
            .toList()
        if (observations.size < 60) return null

        val centers = DoubleArray(frameIds.size * 3)
        frameIds.forEachIndexed { index, id ->
            val value = frameCentersById.getValue(id)
            if (!isVector3(value)) return null
            System.arraycopy(value, 0, centers, index * 3, 3)
        }
        val rotations = DoubleArray(frameIds.size * 9)
        frameIds.forEachIndexed { index, id ->
            val value = rotationsByFrame.getValue(id)
            if (!isRotationMatrix(value)) return null
            System.arraycopy(value, 0, rotations, index * 9, 9)
        }
        val points = DoubleArray(trackIds.size * 3)
        trackIds.forEachIndexed { index, id ->
            val value = pointsByTrackId.getValue(id)
            if (!isVector3(value)) return null
            System.arraycopy(value, 0, points, index * 3, 3)
        }
        val frameIndices = IntArray(observations.size) { observations[it].frameIndex }
        val trackIndices = IntArray(observations.size) { observations[it].trackIndex }
        val imagePoints = DoubleArray(observations.size * 2) { index ->
            val observation = observations[index / 2].observation
            if (index % 2 == 0) observation.x else observation.y
        }
        val intrinsics = doubleArrayOf(
            cameraMatrix.get(0, 0)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 2)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 2)?.firstOrNull() ?: Double.NaN
        )
        if (intrinsics.any { !it.isFinite() } || intrinsics[0] <= 0.0 || intrinsics[3] <= 0.0) return null
        val pointOptimizationMask = if (optimizePointTrackIds == null) null else {
            IntArray(trackIds.size) { index ->
                if (trackIds[index] in optimizePointTrackIds) 1 else 0
            }
        }

        val refinePrincipalPoint = optimizeFocalLength && PrincipalPointRefinementPolicy.isObservable(
            PrincipalPointRefinementPolicy.assess(
                tracks = tracks,
                frameCentersById = frameCentersById,
                rotationsByFrame = rotationsByFrame,
                pointsByTrackId = pointsByTrackId,
                imageWidth = imageWidth,
                imageHeight = imageHeight
            )
        )
        val packed = nativeSolve(
            centers, rotations, points, frameIndices, trackIndices, imagePoints,
            intrinsics, refinePrincipalPoint, imageWidth, imageHeight,
            maxIterations, numThreads, optimizeFocalLength, null, pointOptimizationMask
        ) ?: return null
        val expectedSize = 10L + centers.size + rotations.size + points.size
        if (packed.size.toLong() != expectedSize || packed[0] != 1.0 ||
            !packed[1].isFinite() || !packed[2].isFinite() ||
            !packed[3].isFinite() || packed[3] < 0.0 || packed[3] > Int.MAX_VALUE.toDouble() ||
            packed[3] != packed[3].toInt().toDouble() ||
            !packed[4].isFinite() || packed[4] !in 0.9..1.0
        ) return null

        var offset = 5
        val solvedCenters = LinkedHashMap<Int, DoubleArray>(frameIds.size)
        frameIds.forEach { id ->
            solvedCenters[id] = packed.copyOfRange(offset, offset + 3)
            offset += 3
        }
        val solvedRotations = LinkedHashMap<Int, DoubleArray>(frameIds.size)
        frameIds.forEach { id ->
            val rotation = packed.copyOfRange(offset, offset + 9)
            if (!isRotationMatrix(rotation)) return null
            solvedRotations[id] = rotation
            offset += 9
        }
        val solvedPoints = LinkedHashMap<Int, DoubleArray>(trackIds.size)
        trackIds.forEach { id ->
            solvedPoints[id] = packed.copyOfRange(offset, offset + 3)
            offset += 3
        }
        val solvedIntrinsicValues = packed.copyOfRange(offset, offset + 5)
        val principalPointWithinBounds = !refinePrincipalPoint || (
            kotlin.math.abs(solvedIntrinsicValues[2] - intrinsics[2]) <= 0.05 * imageWidth + 1e-8 &&
                kotlin.math.abs(solvedIntrinsicValues[4] - intrinsics[4]) <= 0.05 * imageHeight + 1e-8
            )
        if (solvedIntrinsicValues.any { !it.isFinite() } ||
            solvedIntrinsicValues[0] <= 0.0 || solvedIntrinsicValues[3] <= 0.0 ||
            solvedIntrinsicValues[1] != intrinsics[1] ||
            (!optimizeFocalLength &&
                (solvedIntrinsicValues[0] != intrinsics[0] || solvedIntrinsicValues[3] != intrinsics[3])) ||
            (!refinePrincipalPoint && solvedIntrinsicValues[2] != intrinsics[2]) ||
            (!refinePrincipalPoint && solvedIntrinsicValues[4] != intrinsics[4]) ||
            !principalPointWithinBounds ||
            solvedCenters.values.any { it.any { value -> !value.isFinite() } } ||
            solvedPoints.values.any { it.any { value -> !value.isFinite() } }
        ) return null

        return Result(
            frameCentersById = solvedCenters,
            rotationsByFrame = solvedRotations,
            pointsByTrackId = solvedPoints,
            intrinsics = CameraIntrinsics(
                fx = solvedIntrinsicValues[0],
                skew = solvedIntrinsicValues[1],
                cx = solvedIntrinsicValues[2],
                fy = solvedIntrinsicValues[3],
                cy = solvedIntrinsicValues[4]
            ),
            initialCost = packed[1],
            finalCost = packed[2],
            positiveDepthRatio = packed[4],
            iterations = packed[3].toInt(),
            principalPointRefinementEnabled = refinePrincipalPoint
        )
    }

    private external fun nativeSolve(
        cameraCenters: DoubleArray,
        rotations: DoubleArray,
        points: DoubleArray,
        frameIndices: IntArray,
        trackIndices: IntArray,
        imagePoints: DoubleArray,
        intrinsics: DoubleArray,
        refinePrincipalPoint: Boolean,
        imageWidth: Int,
        imageHeight: Int,
        maxIterations: Int,
        numThreads: Int,
        optimizeFocalLength: Boolean,
        progressListener: Any?,
        pointOptimizationMask: IntArray?
    ): DoubleArray?

    private fun isVector3(value: DoubleArray): Boolean =
        value.size == 3 && value.all { it.isFinite() }

    private fun isRotationMatrix(value: DoubleArray): Boolean {
        if (value.size != 9 || value.any { !it.isFinite() }) return false
        for (row in 0..2) for (column in 0..2) {
            var product = 0.0
            for (k in 0..2) product += value[row * 3 + k] * value[column * 3 + k]
            if (kotlin.math.abs(product - if (row == column) 1.0 else 0.0) > 1e-3) return false
        }
        val determinant =
            value[0] * (value[4] * value[8] - value[5] * value[7]) -
                value[1] * (value[3] * value[8] - value[5] * value[6]) +
                value[2] * (value[3] * value[7] - value[4] * value[6])
        return determinant.isFinite() && kotlin.math.abs(determinant - 1.0) <= 1e-3
    }

    companion object {
        init {
            System.loadLibrary("trackstudio_reconstruction")
        }
    }
}
