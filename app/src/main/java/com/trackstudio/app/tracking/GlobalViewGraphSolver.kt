package com.trackstudio.app.tracking

/**
 * Runs the two rotation-averaging solves used by GLOMAP's global-mapper
 * controller: a first solve for edge filtering, then a final solve on the
 * filtered largest connected component.
 *
 * Important current limitation: the input is still the custom solver's
 * accepted initial-pair evaluations, not GLOMAP's full image-pair view graph.
 * This stage therefore establishes the controller boundary/order only; it is
 * not yet a complete GLOMAP relative-pose or view-graph implementation.
 */
internal object GlobalViewGraphSolver {
    data class Result(
        val rotationsByFrame: Map<Int, DoubleArray>,
        val inputEdgeCount: Int,
        val acceptedEdgeCount: Int,
        val iterations: Int
    )

    fun solve(
        evaluations: List<InitialPairEvaluation>,
        policy: InitialPairPolicy = InitialPairPolicy(),
        rotationOutlierDegrees: Double = 10.0
    ): Result? {
        val edges = evaluations.asSequence()
            .filter { it.accepted }
            .map { it.candidate }
            .filter {
                it.inlierCount >= policy.minimumInliers &&
                    it.inlierRatio >= policy.minimumInlierRatio &&
                    it.relativeRotation?.let { rotation ->
                        rotation.size == 9 && rotation.all { value -> value.isFinite() }
                    } == true
            }
            .map {
                GlobalRotationAverager.Edge(
                    firstFrame = it.firstFrame,
                    secondFrame = it.secondFrame,
                    relativeRotation = it.relativeRotation!!.copyOf(),
                    inlierCount = it.inlierCount
                )
            }
            .toList()
        if (edges.isEmpty()) return null

        // Upstream global_mapper.cc: first solve is for filtering.
        val filteringPass = GlobalRotationAverager.solve(
            frameIndices = edges.flatMap { listOf(it.firstFrame, it.secondFrame) }.distinct(),
            edges = edges,
            rotationOutlierDegrees = rotationOutlierDegrees
        ) ?: return null
        if (filteringPass.acceptedEdges.isEmpty()) return null

        // GlobalRotationAverager applies rotation-residual filtering and
        // largest-component extraction after each solve. The second call is
        // GLOMAP's final-estimation pass; there is no third solve in the
        // executable controller body at the pinned upstream revision.
        val finalPass = GlobalRotationAverager.solve(
            frameIndices = filteringPass.rotationsByFrame.keys.toList(),
            edges = filteringPass.acceptedEdges,
            rotationOutlierDegrees = rotationOutlierDegrees
        ) ?: return null

        return Result(
            rotationsByFrame = finalPass.rotationsByFrame,
            inputEdgeCount = edges.size,
            acceptedEdgeCount = finalPass.acceptedEdges.size,
            iterations = filteringPass.iterations + finalPass.iterations
        )
    }

    /** Aligns the graph's arbitrary world-rotation gauge to one existing camera pose. */
    internal fun alignRotationsToPoseGauge(
        rotationsByFrame: Map<Int, DoubleArray>,
        anchorFrameId: Int,
        targetAnchorRotation: DoubleArray
    ): Map<Int, DoubleArray>? {
        val graphAnchorRotation = rotationsByFrame[anchorFrameId] ?: return null
        if (!isProperRotation(graphAnchorRotation) || !isProperRotation(targetAnchorRotation)) {
            return null
        }
        val worldGaugeChange = multiply(transpose(graphAnchorRotation), targetAnchorRotation)
        val aligned = LinkedHashMap<Int, DoubleArray>(rotationsByFrame.size)
        for ((frameId, rotation) in rotationsByFrame) {
            if (!isProperRotation(rotation)) return null
            aligned[frameId] = multiply(rotation, worldGaugeChange)
        }
        return aligned
    }

    private fun isProperRotation(rotation: DoubleArray): Boolean {
        if (rotation.size != 9 || rotation.any { !it.isFinite() }) return false
        for (firstRow in 0..2) for (secondRow in firstRow..2) {
            var dot = 0.0
            for (column in 0..2) {
                dot += rotation[firstRow * 3 + column] * rotation[secondRow * 3 + column]
            }
            val expected = if (firstRow == secondRow) 1.0 else 0.0
            if (kotlin.math.abs(dot - expected) > 0.02) return false
        }
        val determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
            rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
            rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
        return determinant.isFinite() && determinant in 0.95..1.05
    }

    private fun transpose(rotation: DoubleArray): DoubleArray = doubleArrayOf(
        rotation[0], rotation[3], rotation[6],
        rotation[1], rotation[4], rotation[7],
        rotation[2], rotation[5], rotation[8]
    )

    private fun multiply(first: DoubleArray, second: DoubleArray): DoubleArray {
        val result = DoubleArray(9)
        for (row in 0..2) for (column in 0..2) for (inner in 0..2) {
            result[row * 3 + column] += first[row * 3 + inner] * second[inner * 3 + column]
        }
        return result
    }
}
