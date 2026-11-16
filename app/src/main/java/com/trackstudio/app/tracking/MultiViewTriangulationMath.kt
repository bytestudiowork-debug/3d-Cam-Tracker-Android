package com.trackstudio.app.tracking

import kotlin.math.acos
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.ln
import kotlin.math.sqrt

internal data class TriangulationView(
    val pixelX: Double,
    val pixelY: Double,
    val pose: CameraPose
)

internal data class MultiViewTriangulationEstimate(
    val pointWorld: DoubleArray,
    val inlierIndices: List<Int>,
    val triangulationAngleDegrees: Double,
    val ransacTrials: Int = 0,
    val localOptimizationPasses: Int = 0
)

/** Calibrated DLT with COLMAP-derived locally optimized consensus for mobile reconstruction. */
internal object MultiViewTriangulationMath {
    /*
     * The estimator and local-optimization loop adapt COLMAP's
     * TriangulationEstimator/EstimateTriangulation and LORANSAC at
     * c5f9cefc87e5dd596b638e4cee0ff543c7d14755. The small-track exhaustive
     * policy follows IncrementalTriangulator::TriangulateTrack. Mobile
     * hypothesis and observation limits remain Android-specific.
     * See app/src/main/assets/licenses/colmap-triangulation-license.txt.
     */
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

        val exhaustiveSampling = observations.size <= EXHAUSTIVE_SAMPLING_THRESHOLD
        val pairIndices = sampleIndices(
            observations.size,
            if (exhaustiveSampling) observations.size else maximumPairObservations
        )
        val scoringIndices = sampleIndices(observations.size, maximumScoringObservations)
        val pairHypotheses = mutableListOf<PairHypothesis>()
        for (firstSlot in 0 until pairIndices.lastIndex) {
            for (secondSlot in firstSlot + 1 until pairIndices.size) {
                pairHypotheses += PairHypothesis(pairIndices[firstSlot], pairIndices[secondSlot])
            }
        }
        if (pairHypotheses.isEmpty()) return null
        val hypotheses = if (exhaustiveSampling) pairHypotheses else sampleValues(pairHypotheses, maximumHypotheses)
        val minimumTrials = if (exhaustiveSampling) hypotheses.size else 0
        val maximumTrials = hypotheses.size
        val maximumAngularResidual = maximumAngularErrorDegrees * maximumAngularErrorDegrees
        var dynamicMaximumTrials = maximumTrials
        var bestModel: DoubleArray? = null
        var bestSupport = InlierSupport(0, Double.POSITIVE_INFINITY)
        var ransacTrials = 0
        var localOptimizationPasses = 0

        for ((trial, hypothesis) in hypotheses.withIndex()) {
            ransacTrials++
            val model = estimateModel(
                listOf(observations[hypothesis.firstIndex], observations[hypothesis.secondIndex]),
                intrinsics,
                minimumParallaxDegrees
            )
            if (model != null) {
                val support = evaluateSupport(
                    model,
                    scoringIndices,
                    observations,
                    intrinsics,
                    maximumAngularResidual,
                    maximumReprojectionErrorPx
                )
                if (isLeftSupportBetter(support, bestSupport)) {
                    bestModel = model
                    bestSupport = support
                    var inlierIndices = consensusInliers(
                        model,
                        observations,
                        intrinsics,
                        maximumAngularErrorDegrees,
                        maximumReprojectionErrorPx
                    )
                    if (support.numInliers > MIN_NUM_SAMPLES && inlierIndices.size > MIN_NUM_SAMPLES) {
                        for (localTrial in 0 until MAX_LOCAL_TRIALS) {
                            val inlierViews = inlierIndices.map(observations::get)
                            if (inlierViews.size <= MIN_NUM_SAMPLES) break
                            val previousBestInliers = bestSupport.numInliers
                            val localModel = estimateModel(inlierViews, intrinsics, minimumParallaxDegrees)
                                ?: break
                            localOptimizationPasses++
                            val localSupport = evaluateSupport(
                                localModel,
                                scoringIndices,
                                observations,
                                intrinsics,
                                maximumAngularResidual,
                                maximumReprojectionErrorPx
                            )
                            if (isLeftSupportBetter(localSupport, bestSupport)) {
                                bestModel = localModel
                                bestSupport = localSupport
                            }
                            if (bestSupport.numInliers <= previousBestInliers) break
                            inlierIndices = consensusInliers(
                                bestModel!!,
                                observations,
                                intrinsics,
                                maximumAngularErrorDegrees,
                                maximumReprojectionErrorPx
                            )
                        }
                    }
                    dynamicMaximumTrials = minOf(
                        maximumTrials,
                        computeRansacTrialLimit(bestSupport.numInliers, scoringIndices.size)
                    )
                }
            }
            if (trial >= dynamicMaximumTrials && trial >= minimumTrials) break
        }

        val point = bestModel ?: return null
        val inliers = consensusInliers(
            point, observations, intrinsics, maximumAngularErrorDegrees, maximumReprojectionErrorPx
        )
        if (inliers.size < MIN_NUM_SAMPLES || inliers.size.toDouble() / observations.size < minimumInlierRatio) {
            return null
        }
        val finalAngle = maximumTriangulationAngleDegrees(point, inliers.map(observations::get))
        return MultiViewTriangulationEstimate(
            point,
            inliers,
            finalAngle,
            ransacTrials,
            localOptimizationPasses
        )
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

    private data class PairHypothesis(val firstIndex: Int, val secondIndex: Int)

    private data class InlierSupport(val numInliers: Int, val residualSum: Double)

    private fun estimateModel(
        observations: List<TriangulationView>,
        intrinsics: PinholeIntrinsics,
        minimumParallaxDegrees: Double
    ): DoubleArray? {
        if (observations.size < MIN_NUM_SAMPLES) return null
        val point = triangulateDlt(observations, intrinsics) ?: return null
        if (observations.any { worldToCamera(it.pose, point)[2] <= 1e-9 }) return null
        val angle = maximumTriangulationAngleDegrees(point, observations)
        if (angle < minimumParallaxDegrees) return null
        return point
    }

    private fun maximumTriangulationAngleDegrees(
        pointWorld: DoubleArray,
        observations: List<TriangulationView>
    ): Double {
        var maximumAngle = 0.0
        for (firstIndex in 0 until observations.lastIndex) {
            for (secondIndex in firstIndex + 1 until observations.size) {
                maximumAngle = maxOf(
                    maximumAngle,
                    triangulationAngleDegrees(
                        pointWorld,
                        observations[firstIndex].pose,
                        observations[secondIndex].pose
                    )
                )
            }
        }
        return maximumAngle
    }

    private fun triangulationAngleDegrees(
        pointWorld: DoubleArray,
        firstPose: CameraPose,
        secondPose: CameraPose
    ): Double {
        val firstCenter = CameraPoseGeometry.cameraCenterWorld(firstPose)
        val secondCenter = CameraPoseGeometry.cameraCenterWorld(secondPose)
        val firstDx = pointWorld[0] - firstCenter[0]
        val firstDy = pointWorld[1] - firstCenter[1]
        val firstDz = pointWorld[2] - firstCenter[2]
        val secondDx = pointWorld[0] - secondCenter[0]
        val secondDy = pointWorld[1] - secondCenter[1]
        val secondDz = pointWorld[2] - secondCenter[2]
        val firstLengthSquared = firstDx * firstDx + firstDy * firstDy + firstDz * firstDz
        val secondLengthSquared = secondDx * secondDx + secondDy * secondDy + secondDz * secondDz
        val baselineX = firstCenter[0] - secondCenter[0]
        val baselineY = firstCenter[1] - secondCenter[1]
        val baselineZ = firstCenter[2] - secondCenter[2]
        val baselineLengthSquared = baselineX * baselineX + baselineY * baselineY + baselineZ * baselineZ
        val denominator = 2.0 * sqrt(firstLengthSquared * secondLengthSquared)
        if (!denominator.isFinite() || denominator == 0.0) return 0.0
        val cosine = ((firstLengthSquared + secondLengthSquared - baselineLengthSquared) / denominator)
            .coerceIn(-1.0, 1.0)
        val angle = acos(cosine)
        return Math.toDegrees(minOf(angle, Math.PI - angle))
    }

    private fun evaluateSupport(
        pointWorld: DoubleArray,
        scoringIndices: List<Int>,
        observations: List<TriangulationView>,
        intrinsics: PinholeIntrinsics,
        maximumAngularResidual: Double,
        maximumReprojectionErrorPx: Double?
    ): InlierSupport {
        var numInliers = 0
        var residualSum = 0.0
        for (index in scoringIndices) {
            val observation = observations[index]
            val angularError = angularErrorDegrees(pointWorld, observation, intrinsics)
            val residual = angularError * angularError
            if (residual <= maximumAngularResidual &&
                (maximumReprojectionErrorPx == null ||
                    reprojectionErrorPixels(pointWorld, observation, intrinsics) <= maximumReprojectionErrorPx)
            ) {
                numInliers++
                residualSum += residual
            }
        }
        return InlierSupport(numInliers, residualSum)
    }

    private fun isLeftSupportBetter(left: InlierSupport, right: InlierSupport): Boolean =
        left.numInliers > right.numInliers ||
            (left.numInliers == right.numInliers && left.residualSum < right.residualSum)

    private fun computeRansacTrialLimit(numInliers: Int, numSamples: Int): Int {
        if (numSamples < MIN_NUM_SAMPLES || numInliers < MIN_NUM_SAMPLES) return Int.MAX_VALUE
        var probabilityOfInlierSample = 1.0
        for (sampleIndex in 0 until MIN_NUM_SAMPLES) {
            probabilityOfInlierSample *=
                (numInliers - sampleIndex).toDouble() / (numSamples - sampleIndex).toDouble()
        }
        val probabilityOfOutlierSample = 1.0 - probabilityOfInlierSample
        if (probabilityOfOutlierSample <= 0.0) return 1
        if (probabilityOfOutlierSample >= 1.0) return Int.MAX_VALUE
        val requiredTrials = ceil(
            ln(1.0 - RANSAC_CONFIDENCE) / ln(probabilityOfOutlierSample) * DYNAMIC_TRIALS_MULTIPLIER
        )
        if (!requiredTrials.isFinite() || requiredTrials >= Int.MAX_VALUE) return Int.MAX_VALUE
        return requiredTrials.toInt().coerceAtLeast(1)
    }

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

    private const val MIN_NUM_SAMPLES = 2
    private const val EXHAUSTIVE_SAMPLING_THRESHOLD = 15
    private const val MAX_LOCAL_TRIALS = 10
    private const val RANSAC_CONFIDENCE = 0.9999
    private const val DYNAMIC_TRIALS_MULTIPLIER = 3.0
}
