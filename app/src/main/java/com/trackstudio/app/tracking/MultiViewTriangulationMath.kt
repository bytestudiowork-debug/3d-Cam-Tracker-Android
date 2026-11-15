package com.trackstudio.app.tracking

import kotlin.math.acos
import kotlin.math.abs
import kotlin.math.sqrt

internal data class TriangulationView(
    val pixelX: Double,
    val pixelY: Double,
    val pose: CameraPose
)

internal data class MultiViewTriangulationEstimate(
    val pointWorld: DoubleArray,
    val inlierIndices: List<Int>,
    val triangulationAngleDegrees: Double
)

/** Pure calibrated DLT and bounded deterministic consensus for mobile reconstruction. */
internal object MultiViewTriangulationMath {
    fun estimate(
        observations: List<TriangulationView>,
        intrinsics: PinholeIntrinsics,
        minimumParallaxDegrees: Double = 1.5,
        maximumAngularErrorDegrees: Double = 2.0,
        minimumInlierRatio: Double = 0.70,
        maximumReprojectionErrorPx: Double? = null,
        maximumPairObservations: Int = 32,
        maximumScoringObservations: Int = 64,
        maximumHypotheses: Int = 24
    ): MultiViewTriangulationEstimate? {
        require(intrinsics.fx.isFinite() && intrinsics.fx > 0.0)
        require(intrinsics.fy.isFinite() && intrinsics.fy > 0.0)
        require(intrinsics.skew.isFinite() && intrinsics.cx.isFinite() && intrinsics.cy.isFinite())
        require(minimumParallaxDegrees.isFinite() && minimumParallaxDegrees >= 1e-3)
        require(maximumAngularErrorDegrees.isFinite() && maximumAngularErrorDegrees > 0.0)
        require(minimumInlierRatio.isFinite() && minimumInlierRatio in 0.0..1.0)
        require(maximumReprojectionErrorPx == null ||
            (maximumReprojectionErrorPx.isFinite() && maximumReprojectionErrorPx > 0.0))
        require(maximumPairObservations >= 2 && maximumScoringObservations >= 2)
        require(maximumHypotheses > 0)
        if (observations.size < 2 || observations.any { !isValid(it) }) return null

        val pairIndices = sampleIndices(observations.size, maximumPairObservations)
        val scoringIndices = sampleIndices(observations.size, maximumScoringObservations)
        val pairHypotheses = mutableListOf<PairHypothesis>()
        for (firstSlot in 0 until pairIndices.lastIndex) {
            for (secondSlot in firstSlot + 1 until pairIndices.size) {
                val firstIndex = pairIndices[firstSlot]
                val secondIndex = pairIndices[secondSlot]
                val angle = rayAngleDegrees(observations[firstIndex], observations[secondIndex], intrinsics)
                if (!angle.isFinite() || angle < minimumParallaxDegrees || angle >= 179.0 ||
                    cameraCentersCoincide(observations[firstIndex].pose, observations[secondIndex].pose)
                ) continue
                pairHypotheses += PairHypothesis(firstIndex, secondIndex, angle)
            }
        }
        if (pairHypotheses.isEmpty()) return null
        pairHypotheses.sortWith(
            compareByDescending<PairHypothesis> { it.angleDegrees }
                .thenBy { it.firstIndex }
                .thenBy { it.secondIndex }
        )

        val boundedHypotheses = sampleValues(pairHypotheses, maximumHypotheses)
        var bestPoint: DoubleArray? = null
        var bestAngle = 0.0
        var bestSupport = -1
        var bestResidualSum = Double.POSITIVE_INFINITY
        for (hypothesis in boundedHypotheses) {
            val point = triangulateDlt(
                listOf(observations[hypothesis.firstIndex], observations[hypothesis.secondIndex]),
                intrinsics
            ) ?: continue
            val firstCameraPoint = worldToCamera(observations[hypothesis.firstIndex].pose, point)
            val secondCameraPoint = worldToCamera(observations[hypothesis.secondIndex].pose, point)
            if (firstCameraPoint[2] <= 1e-9 || secondCameraPoint[2] <= 1e-9) continue

            var support = 0
            var residualSum = 0.0
            for (index in scoringIndices) {
                val observation = observations[index]
                val error = angularErrorDegrees(point, observation, intrinsics)
                if (error <= maximumAngularErrorDegrees &&
                    (maximumReprojectionErrorPx == null ||
                        reprojectionErrorPixels(point, observation, intrinsics) <= maximumReprojectionErrorPx)
                ) {
                    support++
                    residualSum += error
                }
            }
            if (support > bestSupport || (support == bestSupport && residualSum < bestResidualSum)) {
                bestPoint = point
                bestAngle = hypothesis.angleDegrees
                bestSupport = support
                bestResidualSum = residualSum
            }
        }

        var point = bestPoint ?: return null
        var inliers = consensusInliers(
            point, observations, intrinsics, maximumAngularErrorDegrees, maximumReprojectionErrorPx
        )
        if (inliers.size < 2 || inliers.size.toDouble() / observations.size < minimumInlierRatio) return null

        point = triangulateDlt(inliers.map { observations[it] }, intrinsics) ?: return null
        inliers = consensusInliers(
            point, observations, intrinsics, maximumAngularErrorDegrees, maximumReprojectionErrorPx
        )
        if (inliers.size < 2 || inliers.size.toDouble() / observations.size < minimumInlierRatio) return null

        // One final all-inlier solve keeps the returned point tied to the final consensus set.
        point = triangulateDlt(inliers.map { observations[it] }, intrinsics) ?: return null
        inliers = consensusInliers(
            point, observations, intrinsics, maximumAngularErrorDegrees, maximumReprojectionErrorPx
        )
        if (inliers.size < 2 || inliers.size.toDouble() / observations.size < minimumInlierRatio) return null
        return MultiViewTriangulationEstimate(point, inliers, bestAngle)
    }

    /** Linear homogeneous two-view DLT; for three or more views uses bearing-ray projectors. */
    fun triangulateDlt(
        observations: List<TriangulationView>,
        intrinsics: PinholeIntrinsics
    ): DoubleArray? {
        if (observations.size < 2 || observations.any { !isValid(it) }) return null
        val homogeneous = if (observations.size == 2) {
            twoViewRightSingularVector(observations, intrinsics)
        } else {
            multiViewSmallestEigenvector(observations, intrinsics)
        } ?: return null
        val w = homogeneous[3]
        if (!w.isFinite() || abs(w) < 1e-10) return null
        val point = doubleArrayOf(homogeneous[0] / w, homogeneous[1] / w, homogeneous[2] / w)
        return point.takeIf { values -> values.all { it.isFinite() } }
    }

    fun angularErrorDegrees(
        pointWorld: DoubleArray,
        observation: TriangulationView,
        intrinsics: PinholeIntrinsics
    ): Double {
        val cameraPoint = worldToCamera(observation.pose, pointWorld)
        if (cameraPoint[2] <= 1e-9 || !cameraPoint.all(Double::isFinite)) return Double.POSITIVE_INFINITY
        val measured = normalizedRay(observation, intrinsics)
        val predictedLength = sqrt(cameraPoint.sumOf { it * it })
        if (!predictedLength.isFinite() || predictedLength <= 1e-12) return Double.POSITIVE_INFINITY
        val dot = ((measured[0] * cameraPoint[0] + measured[1] * cameraPoint[1] +
            measured[2] * cameraPoint[2]) / predictedLength).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(dot))
    }

    fun rayAngleDegrees(
        first: TriangulationView,
        second: TriangulationView,
        intrinsics: PinholeIntrinsics
    ): Double {
        val a = cameraRayToWorld(first.pose, normalizedRay(first, intrinsics))
        val b = cameraRayToWorld(second.pose, normalizedRay(second, intrinsics))
        val dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(dot))
    }

    private data class PairHypothesis(
        val firstIndex: Int,
        val secondIndex: Int,
        val angleDegrees: Double
    )

    private fun consensusInliers(
        point: DoubleArray,
        observations: List<TriangulationView>,
        intrinsics: PinholeIntrinsics,
        maximumAngularErrorDegrees: Double,
        maximumReprojectionErrorPx: Double?
    ): List<Int> = observations.indices.filter { index ->
        isConsensusInlier(
            point, observations[index], intrinsics, maximumAngularErrorDegrees,
            maximumReprojectionErrorPx
        )
    }

    private fun isConsensusInlier(
        point: DoubleArray,
        observation: TriangulationView,
        intrinsics: PinholeIntrinsics,
        maximumAngularErrorDegrees: Double,
        maximumReprojectionErrorPx: Double?
    ): Boolean {
        if (angularErrorDegrees(point, observation, intrinsics) > maximumAngularErrorDegrees) return false
        return maximumReprojectionErrorPx == null ||
            reprojectionErrorPixels(point, observation, intrinsics) <= maximumReprojectionErrorPx
    }

    private fun reprojectionErrorPixels(
        point: DoubleArray,
        observation: TriangulationView,
        intrinsics: PinholeIntrinsics
    ): Double {
        val cameraPoint = worldToCamera(observation.pose, point)
        if (cameraPoint[2] <= 1e-9 || !cameraPoint.all(Double::isFinite)) {
            return Double.POSITIVE_INFINITY
        }
        val projected = CameraProjectionMath.project(cameraPoint, intrinsics)
        val dx = projected[0] - observation.pixelX
        val dy = projected[1] - observation.pixelY
        return sqrt(dx * dx + dy * dy)
    }

    private fun twoViewRightSingularVector(
        observations: List<TriangulationView>,
        intrinsics: PinholeIntrinsics
    ): DoubleArray? {
        val design = Array(4) { DoubleArray(4) }
        observations.forEachIndexed { index, observation ->
            val normalized = CameraProjectionMath.pixelToNormalized(
                observation.pixelX, observation.pixelY, intrinsics
            )
            val p = projectionRows(observation.pose)
            for (column in 0..3) {
                design[index * 2][column] = normalized[0] * p[2][column] - p[0][column]
                design[index * 2 + 1][column] = normalized[1] * p[2][column] - p[1][column]
            }
        }
        return rightSingularVectorSmallest(design)
    }

    private fun multiViewSmallestEigenvector(
        observations: List<TriangulationView>,
        intrinsics: PinholeIntrinsics
    ): DoubleArray? {
        val normal = Array(4) { DoubleArray(4) }
        for (observation in observations) {
            val bearing = normalizedRay(observation, intrinsics)
            val p = projectionRows(observation.pose)
            val bearingTimesP = DoubleArray(4) { column ->
                bearing[0] * p[0][column] + bearing[1] * p[1][column] + bearing[2] * p[2][column]
            }
            val term = Array(3) { row -> DoubleArray(4) { column ->
                p[row][column] - bearing[row] * bearingTimesP[column]
            } }
            for (row in 0..3) {
                for (column in row..3) {
                    var value = 0.0
                    for (axis in 0..2) value += term[axis][row] * term[axis][column]
                    normal[row][column] += value
                    if (row != column) normal[column][row] += value
                }
            }
        }
        return smallestSymmetricEigenvector(normal)
    }

    private fun rightSingularVectorSmallest(matrix: Array<DoubleArray>): DoubleArray? {
        val rows = matrix.size
        if (rows == 0 || matrix.any { it.size != 4 || it.any { value -> !value.isFinite() } }) return null
        val columns = Array(4) { column -> DoubleArray(rows) { row -> matrix[row][column] } }
        val vectors = identity4()
        for (sweep in 0 until 80) {
            var changed = false
            for (p in 0..2) {
                for (q in p + 1..3) {
                    var alpha = 0.0
                    var beta = 0.0
                    var gamma = 0.0
                    for (row in 0 until rows) {
                        alpha += columns[p][row] * columns[p][row]
                        beta += columns[q][row] * columns[q][row]
                        gamma += columns[p][row] * columns[q][row]
                    }
                    if (abs(gamma) <= 1e-14 * sqrt(alpha * beta)) continue
                    val tau = (beta - alpha) / (2.0 * gamma)
                    val t = if (tau >= 0.0) 1.0 / (tau + sqrt(1.0 + tau * tau))
                    else -1.0 / (-tau + sqrt(1.0 + tau * tau))
                    val c = 1.0 / sqrt(1.0 + t * t)
                    val s = t * c
                    for (row in 0 until rows) {
                        val cp = columns[p][row]
                        val cq = columns[q][row]
                        columns[p][row] = c * cp - s * cq
                        columns[q][row] = s * cp + c * cq
                    }
                    for (row in 0..3) {
                        val vp = vectors[row][p]
                        val vq = vectors[row][q]
                        vectors[row][p] = c * vp - s * vq
                        vectors[row][q] = s * vp + c * vq
                    }
                    changed = true
                }
            }
            if (!changed) break
        }
        val norms = DoubleArray(4) { column -> columns[column].sumOf { it * it } }
        val smallest = norms.indices.minByOrNull { norms[it] } ?: return null
        return DoubleArray(4) { row -> vectors[row][smallest] }.normalizedOrNull()
    }

    private fun smallestSymmetricEigenvector(matrix: Array<DoubleArray>): DoubleArray? {
        if (matrix.size != 4 || matrix.any { it.size != 4 || it.any { value -> !value.isFinite() } }) return null
        val a = Array(4) { matrix[it].copyOf() }
        val vectors = identity4()
        for (iteration in 0 until 100) {
            var p = 0
            var q = 1
            var maximum = abs(a[p][q])
            for (row in 0..3) {
                for (column in row + 1..3) {
                    val value = abs(a[row][column])
                    if (value > maximum) {
                        maximum = value
                        p = row
                        q = column
                    }
                }
            }
            val scale = (0..3).maxOf { abs(a[it][it]) }.coerceAtLeast(1.0)
            if (maximum <= 1e-14 * scale) break
            val app = a[p][p]
            val aqq = a[q][q]
            val apq = a[p][q]
            val tau = (aqq - app) / (2.0 * apq)
            val t = if (tau >= 0.0) 1.0 / (tau + sqrt(1.0 + tau * tau))
            else -1.0 / (-tau + sqrt(1.0 + tau * tau))
            val c = 1.0 / sqrt(1.0 + t * t)
            val s = t * c
            a[p][p] = app - t * apq
            a[q][q] = aqq + t * apq
            a[p][q] = 0.0
            a[q][p] = 0.0
            for (index in 0..3) {
                if (index != p && index != q) {
                    val aip = a[index][p]
                    val aiq = a[index][q]
                    a[index][p] = c * aip - s * aiq
                    a[p][index] = a[index][p]
                    a[index][q] = s * aip + c * aiq
                    a[q][index] = a[index][q]
                }
                val vip = vectors[index][p]
                val viq = vectors[index][q]
                vectors[index][p] = c * vip - s * viq
                vectors[index][q] = s * vip + c * viq
            }
        }
        val smallest = (0..3).minByOrNull { a[it][it] } ?: return null
        return DoubleArray(4) { row -> vectors[row][smallest] }.normalizedOrNull()
    }

    private fun projectionRows(pose: CameraPose): Array<DoubleArray> {
        require(pose.rotation.size == 9 && pose.translation.size == 3)
        val r = pose.rotation
        val t = pose.translation
        require(r.all { it.isFinite() } && t.all { it.isFinite() })
        return arrayOf(
            doubleArrayOf(r[0], r[1], r[2], t[0]),
            doubleArrayOf(r[3], r[4], r[5], t[1]),
            doubleArrayOf(r[6], r[7], r[8], t[2])
        )
    }

    private fun normalizedRay(view: TriangulationView, intrinsics: PinholeIntrinsics): DoubleArray {
        val pinholeRay = CameraProjectionMath.pixelToNormalized(view.pixelX, view.pixelY, intrinsics)
        val length = sqrt(pinholeRay.sumOf { it * it })
        return DoubleArray(3) { pinholeRay[it] / length }
    }

    private fun cameraRayToWorld(pose: CameraPose, cameraRay: DoubleArray): DoubleArray {
        val vector = CameraPoseGeometry.cameraVectorToWorld(
            pose.rotation, cameraRay[0], cameraRay[1], cameraRay[2]
        )
        val length = sqrt(vector.sumOf { it * it })
        return DoubleArray(3) { vector[it] / length }
    }

    private fun worldToCamera(pose: CameraPose, point: DoubleArray): DoubleArray {
        val r = pose.rotation
        val t = pose.translation
        require(r.size == 9 && t.size == 3)
        return doubleArrayOf(
            r[0] * point[0] + r[1] * point[1] + r[2] * point[2] + t[0],
            r[3] * point[0] + r[4] * point[1] + r[5] * point[2] + t[1],
            r[6] * point[0] + r[7] * point[1] + r[8] * point[2] + t[2]
        )
    }

    private fun cameraCentersCoincide(first: CameraPose, second: CameraPose): Boolean {
        val a = CameraPoseGeometry.cameraCenterWorld(first)
        val b = CameraPoseGeometry.cameraCenterWorld(second)
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        val distance = sqrt(dx * dx + dy * dy + dz * dz)
        val scale = maxOf(sqrt(a.sumOf { it * it }), sqrt(b.sumOf { it * it }))
        return distance == 0.0 || (scale > 0.0 && distance <= 1e-12 * scale)
    }

    private fun sampleIndices(size: Int, maximum: Int): List<Int> {
        if (size <= maximum) return (0 until size).toList()
        return (0 until maximum).map { index -> index * (size - 1) / (maximum - 1) }
    }

    private fun <T> sampleValues(values: List<T>, maximum: Int): List<T> {
        if (values.size <= maximum) return values
        val selected = mutableListOf(values.first())
        if (maximum == 1) return selected
        val candidates = (1 until values.size).toMutableList()
        var state = values.size xor 0x5EED1234
        val count = minOf(maximum - 1, candidates.size)
        for (index in 0 until count) {
            state = state * 1664525 + 1013904223
            val offset = ((state.toLong() and 0xffffffffL) % (candidates.size - index)).toInt()
            val chosen = index + offset
            val swap = candidates[index]
            candidates[index] = candidates[chosen]
            candidates[chosen] = swap
            selected += values[candidates[index]]
        }
        return selected
    }

    private fun isValid(view: TriangulationView): Boolean =
        view.pixelX.isFinite() && view.pixelY.isFinite() &&
            view.pose.rotation.size == 9 && view.pose.translation.size == 3 &&
            view.pose.rotation.all { it.isFinite() } && view.pose.translation.all { it.isFinite() } &&
            isProperRotation(view.pose.rotation)

    private fun isProperRotation(rotation: DoubleArray): Boolean {
        for (firstRow in 0..2) {
            for (secondRow in firstRow..2) {
                var dot = 0.0
                for (column in 0..2) {
                    dot += rotation[firstRow * 3 + column] * rotation[secondRow * 3 + column]
                }
                val expected = if (firstRow == secondRow) 1.0 else 0.0
                if (abs(dot - expected) > 0.02) return false
            }
        }
        val determinant = rotation[0] * (rotation[4] * rotation[8] - rotation[5] * rotation[7]) -
            rotation[1] * (rotation[3] * rotation[8] - rotation[5] * rotation[6]) +
            rotation[2] * (rotation[3] * rotation[7] - rotation[4] * rotation[6])
        return determinant.isFinite() && determinant in 0.95..1.05
    }

    private fun DoubleArray.normalizedOrNull(): DoubleArray? {
        val length = sqrt(sumOf { it * it })
        if (!length.isFinite() || length <= 1e-14) return null
        return DoubleArray(size) { this[it] / length }
    }

    private fun identity4() = Array(4) { row -> DoubleArray(4) { column -> if (row == column) 1.0 else 0.0 } }
}
