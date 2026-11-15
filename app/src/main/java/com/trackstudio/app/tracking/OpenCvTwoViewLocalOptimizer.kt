package com.trackstudio.app.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvException
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import kotlin.math.min

internal object OpenCvTwoViewLocalOptimizer {
    data class Result(
        val essentialMatrix: DoubleArray,
        val rotation: DoubleArray,
        val translation: DoubleArray,
        val epipolarInlierCount: Int,
        val inlierIndices: List<Int>,
        val refitRounds: Int
    )

    private data class EssentialModel(
        val essential: DoubleArray,
        val fundamental: DoubleArray
    )

    fun refine(
        firstPoints: Array<Point>,
        secondPoints: Array<Point>,
        cameraMatrix: Mat,
        initialEssential: DoubleArray,
        initialRansacInliers: List<Int>,
        thresholdPx: Double,
        confidence: Double,
        maximumIterations: Int,
        maximumRefitRounds: Int = 2
    ): Result? {
        if (firstPoints.size != secondPoints.size || firstPoints.size < 8 ||
            cameraMatrix.empty() || cameraMatrix.rows() != 3 || cameraMatrix.cols() != 3 ||
            !thresholdPx.isFinite() || thresholdPx <= 0.0 ||
            !confidence.isFinite() || confidence !in 0.0..1.0 || maximumIterations <= 0
        ) return null
        val k = DoubleArray(9)
        if (cameraMatrix.get(0, 0, k) != k.size) return null
        val initialFundamental = EpipolarGeometryMath.fundamentalFromEssential(initialEssential, k)
            ?: return null
        val initialModel = EssentialModel(initialEssential.copyOf(), initialFundamental)

        fun fitEssentialCandidates(indices: List<Int>): List<EssentialModel> {
            if (indices.size < 5) return emptyList()
            val subsetFirst = MatOfPoint2f()
            val subsetSecond = MatOfPoint2f()
            val mask = Mat()
            var estimate: Mat? = null
            try {
                subsetFirst.fromArray(*indices.map(firstPoints::get).toTypedArray())
                subsetSecond.fromArray(*indices.map(secondPoints::get).toTypedArray())
                val estimated = Calib3d.findEssentialMat(
                    subsetFirst,
                    subsetSecond,
                    cameraMatrix,
                    Calib3d.RANSAC,
                    confidence,
                    thresholdPx,
                    min(maximumIterations, LOCAL_RANSAC_ITERATIONS),
                    mask
                )
                estimate = estimated
                if (estimated.empty() || estimated.cols() != 3 ||
                    estimated.rows() < 3 || estimated.rows() % 3 != 0
                ) return emptyList()
                return (0 until estimated.rows() / 3).mapNotNull { candidateIndex ->
                    val candidate = estimated.rowRange(
                        candidateIndex * 3, candidateIndex * 3 + 3
                    ).clone()
                    try {
                        val essential = DoubleArray(9)
                        if (candidate.get(0, 0, essential) != essential.size) return@mapNotNull null
                        val fundamental = EpipolarGeometryMath.fundamentalFromEssential(essential, k)
                            ?: return@mapNotNull null
                        EssentialModel(essential, fundamental)
                    } finally {
                        candidate.release()
                    }
                }
            } catch (_: CvException) {
                return emptyList()
            } finally {
                subsetFirst.release()
                subsetSecond.release()
                mask.release()
                estimate?.release()
            }
        }

        val consensus = try {
            LocalConsensusOptimizer.refine(
                initialModel = initialModel,
                observationCount = firstPoints.size,
                initialInliers = initialRansacInliers,
                minimumInliers = 5,
                threshold = thresholdPx,
                maximumRefitRounds = maximumRefitRounds,
                fitCandidates = { _, indices -> fitEssentialCandidates(indices) },
                residual = { model, index ->
                    EpipolarGeometryMath.symmetricEpipolarError(
                        model.fundamental,
                        firstPoints[index].x,
                        firstPoints[index].y,
                        secondPoints[index].x,
                        secondPoints[index].y
                    )
                }
            )
        } catch (_: CvException) {
            null
        } ?: return null

        val essential = Mat(3, 3, CvType.CV_64F)
        val first = MatOfPoint2f()
        val second = MatOfPoint2f()
        val inlierMask = maskFromIndices(consensus.inlierIndices, firstPoints.size)
        val rotation = Mat()
        val translation = Mat()
        try {
            essential.put(0, 0, *consensus.model.essential)
            first.fromArray(*firstPoints)
            second.fromArray(*secondPoints)
            val inliers = Calib3d.recoverPose(
                essential, first, second, cameraMatrix, rotation, translation, inlierMask
            )
            if (inliers < 5) return null
            val finalInliers = inliersFromMask(inlierMask, firstPoints.size)
            if (finalInliers.size != inliers) return null
            val rotationValues = DoubleArray(9)
            val translationValues = DoubleArray(3)
            if (rotation.get(0, 0, rotationValues) != rotationValues.size ||
                translation.get(0, 0, translationValues) != translationValues.size ||
                rotationValues.any { !it.isFinite() } || translationValues.any { !it.isFinite() }
            ) return null
            return Result(
                essentialMatrix = consensus.model.essential.copyOf(),
                rotation = rotationValues,
                translation = translationValues,
                epipolarInlierCount = consensus.inlierIndices.size,
                inlierIndices = finalInliers,
                refitRounds = consensus.refitRounds
            )
        } catch (_: CvException) {
            return null
        } finally {
            essential.release()
            first.release()
            second.release()
            inlierMask.release()
            rotation.release()
            translation.release()
        }
    }

    fun inliersFromMask(mask: Mat, expectedSize: Int): List<Int> {
        if (mask.empty() || mask.channels() != 1 || mask.total().toInt() != expectedSize) {
            return emptyList()
        }
        return (0 until expectedSize).filter { index ->
            val coordinate = OpenCvMaskLayout.coordinates(mask.rows(), mask.cols(), index, expectedSize)
                ?: return@filter false
            mask.get(coordinate.first, coordinate.second)
                ?.firstOrNull()?.let { it != 0.0 } == true
        }
    }

    fun maskFromIndices(indices: List<Int>, expectedSize: Int): Mat {
        require(expectedSize > 0)
        val mask = Mat.zeros(expectedSize, 1, CvType.CV_8UC1)
        for (index in indices.distinct()) {
            require(index in 0 until expectedSize)
            mask.put(index, 0, byteArrayOf(1.toByte()))
        }
        return mask
    }

    private const val LOCAL_RANSAC_ITERATIONS = 300
}
