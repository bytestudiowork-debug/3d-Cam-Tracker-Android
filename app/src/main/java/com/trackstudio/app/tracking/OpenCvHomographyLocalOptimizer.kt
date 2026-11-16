package com.trackstudio.app.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvException
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point

internal object OpenCvHomographyLocalOptimizer {
    data class Result(
        val homography: DoubleArray,
        val inlierIndices: List<Int>,
        val refitRounds: Int
    )

    fun refine(
        firstPoints: Array<Point>,
        secondPoints: Array<Point>,
        initialHomography: DoubleArray,
        initialRansacInliers: List<Int>,
        thresholdPx: Double,
        confidence: Double = 0.999,
        maximumRefitRounds: Int = 2
    ): Result? {
        if (firstPoints.size != secondPoints.size || firstPoints.size < MINIMUM_FIT_POINTS ||
            firstPoints.any { !it.x.isFinite() || !it.y.isFinite() } ||
            secondPoints.any { !it.x.isFinite() || !it.y.isFinite() } ||
            !thresholdPx.isFinite() || thresholdPx <= 0.0 ||
            !confidence.isFinite() || confidence !in 0.0..1.0 || maximumRefitRounds < 0
        ) return null
        val initialModel = HomographyGeometryMath.fromMatrix(initialHomography) ?: return null

        fun fitCandidates(indices: List<Int>): List<HomographyTransform> {
            if (indices.size < MINIMUM_FIT_POINTS) return emptyList()
            val subsetFirst = MatOfPoint2f()
            val subsetSecond = MatOfPoint2f()
            val fitMask = Mat()
            var estimate: Mat? = null
            try {
                subsetFirst.fromArray(*indices.map(firstPoints::get).toTypedArray())
                subsetSecond.fromArray(*indices.map(secondPoints::get).toTypedArray())
                val fitted = Calib3d.findHomography(
                    subsetFirst,
                    subsetSecond,
                    0,
                    thresholdPx,
                    fitMask,
                    1,
                    confidence
                )
                estimate = fitted
                if (fitted.empty() || fitted.rows() != 3 || fitted.cols() != 3) {
                    return emptyList()
                }
                val values = DoubleArray(9)
                if (fitted.get(0, 0, values) != values.size * java.lang.Double.BYTES) return emptyList()
                return listOfNotNull(HomographyGeometryMath.fromMatrix(values))
            } catch (_: CvException) {
                return emptyList()
            } finally {
                subsetFirst.release()
                subsetSecond.release()
                fitMask.release()
                estimate?.release()
            }
        }

        val consensus = try {
            LocalConsensusOptimizer.refine(
                initialModel = initialModel,
                observationCount = firstPoints.size,
                initialInliers = initialRansacInliers,
                minimumInliers = MINIMUM_FIT_POINTS,
                threshold = thresholdPx,
                maximumRefitRounds = maximumRefitRounds.coerceAtMost(MAXIMUM_REFIT_ROUNDS),
                fitCandidates = { _, indices -> fitCandidates(indices) },
                residual = { model, index ->
                    HomographyGeometryMath.symmetricTransferError(
                        model,
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

        return Result(
            homography = consensus.model.matrix.copyOf(),
            inlierIndices = consensus.inlierIndices,
            refitRounds = consensus.refitRounds
        )
    }

    internal const val MINIMUM_FIT_POINTS = 4
    internal const val MAXIMUM_REFIT_ROUNDS = 2
}
