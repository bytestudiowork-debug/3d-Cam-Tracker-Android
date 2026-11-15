package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureTrack
import org.opencv.core.Mat

/**
 * Joint reprojection bundle adjustment for the independent global pipeline.
 * Camera rotations are held at the robust view-graph solution; camera centres
 * and 3D points are jointly optimized in the same sparse Schur problem.
 */
internal class GlomapGlobalBundleAdjuster {
    data class Result(
        val frameCentersById: Map<Int, DoubleArray>,
        val pointsByTrackId: Map<Int, DoubleArray>,
        val initialCost: Double,
        val finalCost: Double,
        val positiveDepthRatio: Double,
        val iterations: Int
    )

    fun solve(
        frameCentersById: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        tracks: List<CameraFeatureTrack>,
        cameraMatrix: Mat,
        maxIterations: Int = 60,
        numThreads: Int = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
        onIteration: (Int, Int) -> Unit = { _, _ -> }
    ): Result? {
        require(maxIterations > 0 && numThreads > 0)
        if (cameraMatrix.empty() || cameraMatrix.rows() != 3 || cameraMatrix.cols() != 3) return null

        val observedFrames = tracks.asSequence()
            .flatMap { it.observations.asSequence().map { observation -> observation.frameIndex } }
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

        val frameIndex = frameIds.withIndex().associate { it.value to it.index }
        val trackIndex = trackIds.withIndex().associate { it.value to it.index }
        val observations = tracks.asSequence()
            .filter { it.id in trackIndex }
            .flatMap { track ->
                track.observations.asSequence().mapNotNull { observation ->
                    val f = frameIndex[observation.frameIndex] ?: return@mapNotNull null
                    f to (trackIndex.getValue(track.id) to observation)
                }
            }
            .toList()
        if (observations.size < 60) return null

        val centers = frameIds.flatMap { id ->
            frameCentersById.getValue(id).also { if (!isVector3(it)) return null }.toList()
        }.toDoubleArray()
        val rotations = frameIds.flatMap { id ->
            rotationsByFrame.getValue(id).also { if (!isRotationMatrix(it)) return null }.toList()
        }.toDoubleArray()
        val points = trackIds.flatMap { id ->
            pointsByTrackId.getValue(id).also { if (!isVector3(it)) return null }.toList()
        }.toDoubleArray()
        val frameIndices = IntArray(observations.size) { observations[it].first }
        val trackIndices = IntArray(observations.size) { observations[it].second.first }
        val imagePoints = DoubleArray(observations.size * 2) { index ->
            val observation = observations[index / 2].second.second
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

        val packed = nativeSolve(
            centers, rotations, points, frameIndices, trackIndices,
            imagePoints, intrinsics, maxIterations, numThreads,
            GlomapSolverProgressListener(onIteration)
        ) ?: return null
        val expected = 5 + centers.size + points.size
        if (packed.size != expected || packed[0] != 1.0 ||
            !packed[1].isFinite() || !packed[2].isFinite() ||
            !packed[4].isFinite() || packed[4] < 0.9 || packed[4] > 1.0
        ) return null

        var offset = 5
        val solvedCenters = LinkedHashMap<Int, DoubleArray>(frameIds.size)
        for (id in frameIds) {
            solvedCenters[id] = packed.copyOfRange(offset, offset + 3)
            offset += 3
        }
        val solvedPoints = LinkedHashMap<Int, DoubleArray>(trackIds.size)
        for (id in trackIds) {
            solvedPoints[id] = packed.copyOfRange(offset, offset + 3)
            offset += 3
        }
        if (solvedCenters.values.any { it.any { value -> !value.isFinite() } } ||
            solvedPoints.values.any { it.any { value -> !value.isFinite() } }
        ) return null

        return Result(
            frameCentersById = solvedCenters,
            pointsByTrackId = solvedPoints,
            initialCost = packed[1],
            finalCost = packed[2],
            positiveDepthRatio = packed[4],
            iterations = packed[3].toInt()
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
        maxIterations: Int,
        numThreads: Int,
        progressListener: GlomapSolverProgressListener
    ): DoubleArray?

    private fun isVector3(value: DoubleArray): Boolean {
        return value.size == 3 && value.all { it.isFinite() }
    }

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
            System.loadLibrary("trackstudio_glomap")
        }
    }
}
