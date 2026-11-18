package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import java.util.Random
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class OpenCvInitialPairEvaluatorInstrumentationTest {

    @Test
    fun fSupportedUncalibratedPairIsConvertedToEssentialAndPoseRecovered() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())

        val matchCount = 160
        val firstKeypoints = Array(matchCount) { KeyPoint(0.0f, 0.0f, 1.0f) }
        val secondKeypoints = Array(matchCount) { KeyPoint(0.0f, 0.0f, 1.0f) }
        val firstDescriptors = Mat(matchCount, DESCRIPTOR_LENGTH, CvType.CV_8UC1)
        val secondDescriptors = Mat(matchCount, DESCRIPTOR_LENGTH, CvType.CV_8UC1)
        val cameraMatrix = Mat(3, 3, CvType.CV_64F)
        val matcher = SiftDescriptorMatcher()
        try {
            cameraMatrix.put(
                0, 0,
                APP_FOCAL, 0.0, PRINCIPAL_X,
                0.0, APP_FOCAL, PRINCIPAL_Y,
                0.0, 0.0, 1.0
            )
            val rotation = rotation(
                doubleArrayOf(0.3, 0.8, 0.2),
                angle = 0.28
            )
            val pointRandom = Random(20261118L)
            val descriptorRandom = Random(20261119L)
            val descriptors = ByteArray(matchCount * DESCRIPTOR_LENGTH)
            val descriptorRow = ByteArray(DESCRIPTOR_LENGTH)
            for (index in 0 until matchCount) {
                val x = pointRandom.nextDouble() * 5.0 - 2.5
                val y = pointRandom.nextDouble() * 3.6 - 1.8
                val z = pointRandom.nextDouble() * 4.0 + 4.0
                val secondX = rotation[0] * x + rotation[1] * y + rotation[2] * z + 1.8
                val secondY = rotation[3] * x + rotation[4] * y + rotation[5] * z + 0.12
                val secondZ = rotation[6] * x + rotation[7] * y + rotation[8] * z + 0.25
                val firstPixelX = TRUE_FOCAL * x / z + PRINCIPAL_X + pointRandom.nextGaussian() * 0.35
                val firstPixelY = TRUE_FOCAL * y / z + PRINCIPAL_Y + pointRandom.nextGaussian() * 0.35
                val secondPixelX = TRUE_FOCAL * secondX / secondZ + PRINCIPAL_X +
                    pointRandom.nextGaussian() * 0.35
                val secondPixelY = TRUE_FOCAL * secondY / secondZ + PRINCIPAL_Y +
                    pointRandom.nextGaussian() * 0.35
                firstKeypoints[index] = KeyPoint(firstPixelX.toFloat(), firstPixelY.toFloat(), 1.0f)
                secondKeypoints[index] = KeyPoint(secondPixelX.toFloat(), secondPixelY.toFloat(), 1.0f)
                descriptorRandom.nextBytes(descriptorRow)
                val descriptorValues = FloatArray(DESCRIPTOR_LENGTH) {
                    (descriptorRow[it].toInt() and 0xff).toFloat()
                }
                RootSiftDescriptorMath.normalizeDescriptorInPlace(descriptorValues)
                for (descriptorIndex in descriptorValues.indices) {
                    descriptors[index * DESCRIPTOR_LENGTH + descriptorIndex] =
                        descriptorValues[descriptorIndex].roundToInt().coerceIn(0, 255).toByte()
                }
            }
            assertTrue(
                firstDescriptors.put(0, 0, descriptors) == descriptors.size &&
                    secondDescriptors.put(0, 0, descriptors) == descriptors.size
            )

            val evaluation = OpenCvInitialPairEvaluator.evaluate(
                firstFrame = 0,
                secondFrame = 10,
                firstKeypoints = firstKeypoints,
                firstDescriptors = firstDescriptors,
                secondKeypoints = secondKeypoints,
                secondDescriptors = secondDescriptors,
                cameraMatrix = cameraMatrix,
                matcher = matcher,
                frameWidth = 1280,
                frameHeight = 720,
                policy = InitialPairPolicy.initialization(),
                ransacThresholdPx = 4.0
            )

            val candidate = evaluation.candidate
            val candidateDiagnostics = "matches=${candidate.matchedCount}, " +
                "inliers=${candidate.inlierCount}, " +
                "E/F=${candidate.essentialToFundamentalInlierRatio}, " +
                "F-derived=${candidate.fundamentalPoseRecovered}, " +
                "parallax=${candidate.medianParallaxDegrees}, " +
                "H=${candidate.homographyInlierRatio}"
            assertTrue("F-supported pose recovery was rejected: ${evaluation.reason}", evaluation.accepted)
            assertTrue(
                "Expected F-derived uncalibrated pose: $candidateDiagnostics",
                candidate.fundamentalPoseRecovered
            )
            assertTrue(
                "E/F support ratio should indicate unknown focal: $candidateDiagnostics",
                candidate.essentialToFundamentalInlierRatio < 0.95
            )
            assertTrue(
                "Too few cheiral inliers: $candidateDiagnostics",
                candidate.inlierCount >= 100
            )
            assertTrue(
                "Parallax below initialization threshold: $candidateDiagnostics",
                candidate.medianParallaxDegrees > 16.0
            )
            assertTrue(
                "Planar support ratio exceeds the expected bound: $candidateDiagnostics",
                candidate.homographyInlierRatio <= 0.8
            )
        } finally {
            matcher.clear()
            firstDescriptors.release()
            secondDescriptors.release()
            cameraMatrix.release()
        }
    }

    @Test
    fun initialHomographyLocalRefitDeterministicallyExpandsSupportOnSyntheticPlanarMatches() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())

        val inlierCount = 80
        val firstPoints = Array(100) { index ->
            val row = index / 10
            val column = index % 10
            Point(column * 31.0 + row * 2.1, row * 23.0 + column * 1.4)
        }
        val secondPoints = Array(firstPoints.size) { index ->
            val first = firstPoints[index]
            if (index < inlierCount) {
                Point(first.x + 5.0, first.y - 2.0)
            } else {
                Point(first.x + 120.0 + index, first.y - 90.0 - index)
            }
        }
        val approximateHomography = doubleArrayOf(
            1.0, 0.0, 5.8,
            0.0, 1.0, -2.2,
            0.0, 0.0, 1.0
        )
        val initialInliers = (0 until 60).toList()

        val firstRun = OpenCvInitialPairEvaluator.conservativeHomographySupportCount(
            firstPoints,
            secondPoints,
            approximateHomography,
            initialInliers,
            rawOpenCvInlierCount = initialInliers.size,
            thresholdPx = 4.0
        )
        val repeatedRun = OpenCvInitialPairEvaluator.conservativeHomographySupportCount(
            firstPoints,
            secondPoints,
            approximateHomography,
            initialInliers,
            rawOpenCvInlierCount = initialInliers.size,
            thresholdPx = 4.0
        )

        assertEquals(inlierCount, firstRun)
        assertEquals(firstRun, repeatedRun)
    }

    @Test
    fun oneWaySquaredHomographySupportDiffersFromCurrentSymmetricRefit() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())

        val first = arrayOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(0.0, 100.0),
            Point(100.0, 100.0),
            Point(20.0, 20.0)
        )
        val second = arrayOf(
            Point(0.0, 0.0),
            Point(10.0, 0.0),
            Point(0.0, 10.0),
            Point(10.0, 10.0),
            Point(2.5, 2.0)
        )
        val fitFirst = MatOfPoint2f()
        val fitSecond = MatOfPoint2f()
        val fitMask = Mat()
        var homography = Mat()
        try {
            fitFirst.fromArray(*first.take(4).toTypedArray())
            fitSecond.fromArray(*second.take(4).toTypedArray())
            homography = Calib3d.findHomography(
                fitFirst, fitSecond, 0, 4.0, fitMask, 1, 0.999
            )
            val model = requireNotNull(OpenCvMatrixReadback.readCv64F(homography, 3, 3))
            val transform = requireNotNull(HomographyGeometryMath.fromMatrix(model))

            val sourceStyleMask = first.indices.map { index ->
                squaredForwardTransferResidual(
                    model,
                    first[index].x, first[index].y,
                    second[index].x, second[index].y
                ) <= 4.0 * 4.0
            }
            val currentSymmetricMask = first.indices.map { index ->
                HomographyGeometryMath.symmetricTransferError(
                    transform,
                    first[index].x, first[index].y,
                    second[index].x, second[index].y
                ) <= 4.0
            }
            val localResult = requireNotNull(OpenCvHomographyLocalOptimizer.refine(
                firstPoints = first,
                secondPoints = second,
                initialHomography = model,
                initialRansacInliers = first.indices.toList(),
                thresholdPx = 4.0,
                maximumRefitRounds = 0
            ))

            assertEquals(listOf(true, true, true, true, true), sourceStyleMask)
            assertEquals(listOf(true, true, true, true, false), currentSymmetricMask)
            assertEquals(listOf(0, 1, 2, 3), localResult.inlierIndices)
            assertTrue(
                kotlin.math.abs(
                    squaredForwardTransferResidual(model, 20.0, 20.0, 2.5, 2.0) - 0.25
                ) < 1e-6
            )
        } finally {
            fitFirst.release()
            fitSecond.release()
            fitMask.release()
            homography.release()
        }
    }

    @Test
    fun poseRecoveryRetainsPositiveDepthPointsBeyondOpenCvDefaultFiftyUnitCutoff() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())

        val count = 32
        val cameraMatrix = Mat(3, 3, CvType.CV_64F)
        val essential = Mat(3, 3, CvType.CV_64F)
        val firstPoints = MatOfPoint2f()
        val secondPoints = MatOfPoint2f()
        val defaultMask = Mat(count, 1, CvType.CV_8UC1)
        val sourceCompatibleMask = Mat(count, 1, CvType.CV_8UC1)
        val defaultRotation = Mat()
        val defaultTranslation = Mat()
        val sourceCompatibleRotation = Mat()
        val sourceCompatibleTranslation = Mat()
        try {
            cameraMatrix.put(
                0, 0,
                TRUE_FOCAL, 0.0, PRINCIPAL_X,
                0.0, TRUE_FOCAL, PRINCIPAL_Y,
                0.0, 0.0, 1.0
            )
            // E = [t]_x R for R=I and unit translation t=(1, 0, 0).
            essential.put(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
            val first = Array(count) { index ->
                val x = -0.4 + (index % 8) * 0.1
                val y = -0.2 + (index / 8) * 0.1
                val z = 120.0 + index * 2.0
                Point(TRUE_FOCAL * x / z + PRINCIPAL_X, TRUE_FOCAL * y / z + PRINCIPAL_Y)
            }
            val second = Array(count) { index ->
                val x = -0.4 + (index % 8) * 0.1
                val y = -0.2 + (index / 8) * 0.1
                val z = 120.0 + index * 2.0
                Point(TRUE_FOCAL * (x + 1.0) / z + PRINCIPAL_X, TRUE_FOCAL * y / z + PRINCIPAL_Y)
            }
            firstPoints.fromArray(*first)
            secondPoints.fromArray(*second)
            val allInliers = ByteArray(count) { 1 }
            defaultMask.put(0, 0, allInliers)
            sourceCompatibleMask.put(0, 0, allInliers)

            val defaultCount = Calib3d.recoverPose(
                essential, firstPoints, secondPoints, cameraMatrix,
                defaultRotation, defaultTranslation, defaultMask
            )
            val sourceCompatibleCount = OpenCvTwoViewPoseRecovery.recover(
                essential, firstPoints, secondPoints, cameraMatrix,
                sourceCompatibleRotation, sourceCompatibleTranslation, sourceCompatibleMask
            )

            assertEquals(0, defaultCount)
            assertEquals(count, sourceCompatibleCount)
        } finally {
            cameraMatrix.release()
            essential.release()
            firstPoints.release()
            secondPoints.release()
            defaultMask.release()
            sourceCompatibleMask.release()
            defaultRotation.release()
            defaultTranslation.release()
            sourceCompatibleRotation.release()
            sourceCompatibleTranslation.release()
        }
    }

    private fun rotation(axisInput: DoubleArray, angle: Double): DoubleArray {
        val norm = sqrt(axisInput.sumOf { it * it })
        val x = axisInput[0] / norm
        val y = axisInput[1] / norm
        val z = axisInput[2] / norm
        val c = cos(angle)
        val s = sin(angle)
        val t = 1.0 - c
        return doubleArrayOf(
            t * x * x + c, t * x * y - s * z, t * x * z + s * y,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c
        )
    }

    private fun squaredForwardTransferResidual(
        homography: DoubleArray,
        firstX: Double,
        firstY: Double,
        secondX: Double,
        secondY: Double
    ): Double {
        val projectedX = homography[0] * firstX + homography[1] * firstY + homography[2]
        val projectedY = homography[3] * firstX + homography[4] * firstY + homography[5]
        val projectedScale = 1.0 / (
            homography[6] * firstX + homography[7] * firstY + homography[8]
        )
        val errorX = secondX - projectedX * projectedScale
        val errorY = secondY - projectedY * projectedScale
        return errorX * errorX + errorY * errorY
    }

    private companion object {
        const val TRUE_FOCAL = 800.0
        // Keep the mismatch large enough to exercise the source-ratio F-to-E fallback.
        const val APP_FOCAL = 500.0
        const val PRINCIPAL_X = 640.0
        const val PRINCIPAL_Y = 360.0
        const val DESCRIPTOR_LENGTH = 128
    }
}
