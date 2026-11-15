package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class HomographyLocalOptimizerTest {
    @Test
    fun knownProjectiveTransformHasZeroSymmetricTransferError() {
        val transform = HomographyGeometryMath.fromMatrix(
            doubleArrayOf(
                1.08, 0.07, 12.0,
                -0.04, 0.94, -7.0,
                0.0008, -0.0005, 1.0
            )
        )
        assertNotNull(transform)
        val firstX = 145.0
        val firstY = 83.0
        val second = HomographyGeometryMath.project(transform!!, firstX, firstY)
        assertNotNull(second)

        assertEquals(
            0.0,
            HomographyGeometryMath.symmetricTransferError(
                transform, firstX, firstY, second!![0], second[1]
            ),
            1e-9
        )
        val scaledTransform = HomographyGeometryMath.fromMatrix(
            DoubleArray(transform.matrix.size) { -7.0 * transform.matrix[it] }
        )!!
        assertEquals(
            HomographyGeometryMath.symmetricTransferError(
                transform, firstX, firstY, second[0] + 0.25, second[1] - 0.1
            ),
            HomographyGeometryMath.symmetricTransferError(
                scaledTransform, firstX, firstY, second[0] + 0.25, second[1] - 0.1
            ),
            1e-10
        )
    }

    @Test
    fun symmetricTransferErrorDetectsErrorsOnEitherImageAndRejectsInvalidModels() {
        val transform = HomographyGeometryMath.fromMatrix(translation(2.0, -1.0))!!
        val first = doubleArrayOf(20.0, 14.0)
        val second = HomographyGeometryMath.project(transform, first[0], first[1])!!

        val firstSideError = HomographyGeometryMath.symmetricTransferError(
            transform, first[0] + 1.0, first[1], second[0], second[1]
        )
        val secondSideError = HomographyGeometryMath.symmetricTransferError(
            transform, first[0], first[1], second[0] + 1.0, second[1]
        )
        assertTrue(firstSideError > 0.9)
        assertTrue(secondSideError > 0.9)
        assertNull(HomographyGeometryMath.fromMatrix(DoubleArray(9)))
        val horizon = HomographyGeometryMath.fromMatrix(
            doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        )!!
        assertEquals(
            Double.POSITIVE_INFINITY,
            HomographyGeometryMath.symmetricTransferError(horizon, -1.0, 2.0, 0.0, 0.0),
            0.0
        )
        assertEquals(
            Double.POSITIVE_INFINITY,
            HomographyGeometryMath.symmetricTransferError(
                transform, Double.NaN, first[1], second[0], second[1]
            ),
            0.0
        )
    }

    @Test
    fun boundedLocalRefitRecoversTransformAndReclassifiesGrossOutlier() {
        val firstPoints = (0 until 8).map { index ->
            doubleArrayOf(13.0 * index, 7.0 * (index % 3))
        }.toMutableList()
        val initialInlierCount = firstPoints.size
        val truth = translation(3.0, -2.0)
        val truthTransform = HomographyGeometryMath.fromMatrix(truth)!!
        val secondPoints = firstPoints.map { point ->
            HomographyGeometryMath.project(truthTransform, point[0], point[1])!!
        }.toMutableList()
        val outlierSource = firstPoints.last().copyOf()
        val expectedOutlier = HomographyGeometryMath.project(
            truthTransform, outlierSource[0], outlierSource[1]
        )!!
        firstPoints += outlierSource
        secondPoints += doubleArrayOf(expectedOutlier[0] + 25.0, expectedOutlier[1] - 18.0)
        val initialTransform = HomographyGeometryMath.fromMatrix(translation(3.6, -1.4))!!

        val result = LocalConsensusOptimizer.refine(
            initialModel = initialTransform,
            observationCount = firstPoints.size,
            initialInliers = (0 until initialInlierCount).toList(),
            minimumInliers = 4,
            threshold = 1.0,
            maximumRefitRounds = 2,
            fitCandidates = { _, indices ->
                val dx = indices.map { secondPoints[it][0] - firstPoints[it][0] }.average()
                val dy = indices.map { secondPoints[it][1] - firstPoints[it][1] }.average()
                listOfNotNull(HomographyGeometryMath.fromMatrix(translation(dx, dy)))
            },
            residual = { model, index ->
                HomographyGeometryMath.symmetricTransferError(
                    model,
                    firstPoints[index][0], firstPoints[index][1],
                    secondPoints[index][0], secondPoints[index][1]
                )
            }
        )

        assertNotNull(result)
        assertEquals((0 until 8).toList(), result!!.inlierIndices)
        assertEquals(1, result.refitRounds)
        val recovered = HomographyGeometryMath.project(result.model, 31.0, 19.0)!!
        assertTrue(abs(recovered[0] - 34.0) < 1e-9)
        assertTrue(abs(recovered[1] - 17.0) < 1e-9)
    }

    private fun translation(dx: Double, dy: Double) = doubleArrayOf(
        1.0, 0.0, dx,
        0.0, 1.0, dy,
        0.0, 0.0, 1.0
    )
}
