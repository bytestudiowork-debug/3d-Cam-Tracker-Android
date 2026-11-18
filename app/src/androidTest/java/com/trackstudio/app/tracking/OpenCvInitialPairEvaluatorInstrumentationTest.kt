package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
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

    @Test
    fun integratedProductionNativePathMatchesPinnedFocalAndMixedFixtures() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        val camera = doubleArrayOf(
            800.0, 0.0, PRINCIPAL_X,
            0.0, 800.0, PRINCIPAL_Y,
            0.0, 0.0, 1.0
        )
        val allInliers = "1".repeat(120)
        val nonplanarHMask =
            "000000000000000001111100000000000011111111110000000000001111111111" +
                "000000000001111111100000000000000111000000000000000000"
        val mixedHMask =
            "111111111111111111111111111111111111111111111111111111111111" +
                "111111111111111111111111111111111111111100000100000100000100"
        data class Expected(
            val scene: String,
            val focalPrior: Boolean,
            val configuration: SourceFaithfulTwoViewEstimator.Configuration,
            val selectedModel: SourceFaithfulTwoViewEstimator.InlierModel,
            val poseSource: SourceFaithfulTwoViewEstimator.PoseSource,
            val essentialSupport: Int,
            val fundamentalSupport: Int,
            val homographySupport: Int,
            val homographyMask: String,
            val passesHSeedGates: Boolean
        )
        val cases = listOf(
            Expected("nonplanar", true,
                SourceFaithfulTwoViewEstimator.Configuration.CALIBRATED,
                SourceFaithfulTwoViewEstimator.InlierModel.ESSENTIAL,
                SourceFaithfulTwoViewEstimator.PoseSource.ESSENTIAL,
                120, 120, 36, nonplanarHMask, true),
            Expected("nonplanar", false,
                SourceFaithfulTwoViewEstimator.Configuration.UNCALIBRATED,
                SourceFaithfulTwoViewEstimator.InlierModel.FUNDAMENTAL,
                SourceFaithfulTwoViewEstimator.PoseSource.FUNDAMENTAL,
                120, 120, 36, nonplanarHMask, true),
            Expected("mixed", true,
                SourceFaithfulTwoViewEstimator.Configuration.PLANAR_OR_PANORAMIC,
                SourceFaithfulTwoViewEstimator.InlierModel.ESSENTIAL,
                SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY,
                120, 120, 103, mixedHMask, false),
            Expected("mixed", false,
                SourceFaithfulTwoViewEstimator.Configuration.PLANAR_OR_PANORAMIC,
                SourceFaithfulTwoViewEstimator.InlierModel.FUNDAMENTAL,
                SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY,
                120, 120, 103, mixedHMask, false),
            Expected("planar", true,
                SourceFaithfulTwoViewEstimator.Configuration.PLANAR_OR_PANORAMIC,
                SourceFaithfulTwoViewEstimator.InlierModel.ESSENTIAL,
                SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY,
                120, 120, 120, allInliers, false),
            Expected("planar", false,
                SourceFaithfulTwoViewEstimator.Configuration.PLANAR_OR_PANORAMIC,
                SourceFaithfulTwoViewEstimator.InlierModel.FUNDAMENTAL,
                SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY,
                120, 120, 120, allInliers, false)
        )
        for (expected in cases) {
            val scene = sourceFixture(expected.scene)
            val estimate = requireNotNull(
                SourceFaithfulTwoViewEstimator.estimateInitialPair(
                    firstPixelsXY = flatten(scene.first),
                    secondPixelsXY = flatten(scene.second),
                    cameraMatrixRowMajor = camera,
                    bothCamerasHaveFocalPriors = expected.focalPrior,
                    maxErrorPixels = 4.0
                )
            )
            val label = "${expected.scene}, focalPrior=${expected.focalPrior}"
            assertEquals(label, expected.configuration, estimate.configuration)
            assertEquals(label, expected.selectedModel, estimate.selectedModel)
            assertEquals(label, expected.poseSource, estimate.poseSource)
            assertEquals(label, expected.essentialSupport, estimate.essential.inlierCount)
            assertEquals(label, expected.fundamentalSupport, estimate.fundamental.inlierCount)
            assertEquals(label, expected.homographySupport, estimate.homography.inlierCount)
            assertEquals(label, expected.homographyMask, maskString(estimate.homography.inlierMask))
            assertEquals(label, allInliers, maskString(estimate.selectedInlierMask))
            assertNotNull("$label pose missing", estimate.pose)
            assertEquals(label, expected.passesHSeedGates, estimate.passesInitialPairGates)
            if (expected.scene == "mixed") {
                assertEquals(
                    label,
                    7.294462,
                    Math.toDegrees(estimate.medianTriangulationAngleRadians),
                    0.02
                )
            }
        }

        for (focalPrior in listOf(true, false)) {
            val scene = sourceFixture("mixed_high")
            val estimate = requireNotNull(
                SourceFaithfulTwoViewEstimator.estimateInitialPair(
                    firstPixelsXY = flatten(scene.first),
                    secondPixelsXY = flatten(scene.second),
                    cameraMatrixRowMajor = camera,
                    bothCamerasHaveFocalPriors = focalPrior,
                    maxErrorPixels = 4.0
                )
            )
            assertEquals(SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY, estimate.poseSource)
            assertEquals(
                if (focalPrior) SourceFaithfulTwoViewEstimator.InlierModel.ESSENTIAL
                else SourceFaithfulTwoViewEstimator.InlierModel.FUNDAMENTAL,
                estimate.selectedModel
            )
            assertEquals(allInliers, maskString(estimate.selectedInlierMask))
            assertTrue(estimate.passesInitialPairGates)
            assertTrue(Math.toDegrees(estimate.medianTriangulationAngleRadians) > 16.0)
            assertTrue(kotlin.math.abs(requireNotNull(estimate.pose).translation[2]) < 0.95)
        }

        val acceptedFixture = sourceFixture("mixed_high")
        val descriptors = identicalDescriptors(acceptedFixture.first.size)
        val cameraMat = Mat(3, 3, CvType.CV_64F)
        val matcher = SiftDescriptorMatcher()
        try {
            cameraMat.put(0, 0, *camera)
            for (focalPrior in listOf(true, false)) {
                val evaluation = OpenCvInitialPairEvaluator.evaluate(
                    firstFrame = 0,
                    secondFrame = 10,
                    firstKeypoints = Array(acceptedFixture.first.size) { index ->
                        KeyPoint(acceptedFixture.first[index].x.toFloat(),
                            acceptedFixture.first[index].y.toFloat(), 1.0f)
                    },
                    firstDescriptors = descriptors.first,
                    secondKeypoints = Array(acceptedFixture.second.size) { index ->
                        KeyPoint(acceptedFixture.second[index].x.toFloat(),
                            acceptedFixture.second[index].y.toFloat(), 1.0f)
                    },
                    secondDescriptors = descriptors.second,
                    cameraMatrix = cameraMat,
                    matcher = matcher,
                    frameWidth = 1280,
                    frameHeight = 720,
                    policy = InitialPairPolicy.initialization(),
                    cameraFocalLengthHasPrior = focalPrior
                )
                val candidate = evaluation.candidate
                assertTrue("H seed candidate was rejected: ${evaluation.reason}", evaluation.accepted)
                assertTrue(candidate.homographyPoseRecovered)
                assertEquals(SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY, candidate.poseSource)
                assertEquals(
                    if (focalPrior) SourceFaithfulTwoViewEstimator.InlierModel.ESSENTIAL
                    else SourceFaithfulTwoViewEstimator.InlierModel.FUNDAMENTAL,
                    candidate.selectedInlierModel
                )
                assertTrue(candidate.inlierCount >= 100)
                assertEquals(candidate.inlierCount, candidate.verifiedMatches.size)
            }
        } finally {
            matcher.clear()
            descriptors.first.release()
            descriptors.second.release()
            cameraMat.release()
        }
    }

    private data class SourceScene(val first: Array<Point>, val second: Array<Point>)

    private fun sourceFixture(kind: String): SourceScene {
        val r = rotation(doubleArrayOf(0.25, 0.6, 0.31), 0.13)
        val translationX = if (kind == "mixed_high") 1.5 else 0.65
        val first = Array(120) { index ->
            val i = index.toDouble()
            val x = -2.0 + 0.037 * i + 0.31 * sin(0.31 * i)
            val y = -1.0 + 0.16 * (index % 11) + 0.17 * cos(0.47 * i)
            val z = when (kind) {
                "nonplanar" -> 2.8 + 0.19 * (index % 17)
                "mixed", "mixed_high" -> if (index < 100) 4.5 else 2.2 + 0.71 * (index % 6)
                else -> 4.5
            }
            Point(TRUE_FOCAL * x / z + PRINCIPAL_X, TRUE_FOCAL * y / z + PRINCIPAL_Y)
        }
        val second = Array(120) { index ->
            val i = index.toDouble()
            val x = -2.0 + 0.037 * i + 0.31 * sin(0.31 * i)
            val y = -1.0 + 0.16 * (index % 11) + 0.17 * cos(0.47 * i)
            val z = when (kind) {
                "nonplanar" -> 2.8 + 0.19 * (index % 17)
                "mixed", "mixed_high" -> if (index < 100) 4.5 else 2.2 + 0.71 * (index % 6)
                else -> 4.5
            }
            val x2 = r[0] * x + r[1] * y + r[2] * z + translationX
            val y2 = r[3] * x + r[4] * y + r[5] * z - 0.08
            val z2 = r[6] * x + r[7] * y + r[8] * z + 0.03
            Point(TRUE_FOCAL * x2 / z2 + PRINCIPAL_X, TRUE_FOCAL * y2 / z2 + PRINCIPAL_Y)
        }
        return SourceScene(first, second)
    }

    private fun flatten(points: Array<Point>): DoubleArray = DoubleArray(points.size * 2).also { values ->
        points.forEachIndexed { index, point ->
            values[2 * index] = point.x
            values[2 * index + 1] = point.y
        }
    }

    private fun maskString(mask: BooleanArray): String = buildString(mask.size) {
        mask.forEach { append(if (it) '1' else '0') }
    }

    private fun identicalDescriptors(count: Int): Pair<Mat, Mat> {
        val random = Random(20261119L)
        val row = ByteArray(DESCRIPTOR_LENGTH)
        val values = ByteArray(count * DESCRIPTOR_LENGTH)
        for (index in 0 until count) {
            random.nextBytes(row)
            val descriptor = FloatArray(DESCRIPTOR_LENGTH) {
                (row[it].toInt() and 0xff).toFloat()
            }
            RootSiftDescriptorMath.normalizeDescriptorInPlace(descriptor)
            for (column in descriptor.indices) {
                values[index * DESCRIPTOR_LENGTH + column] =
                    descriptor[column].roundToInt().coerceIn(0, 255).toByte()
            }
        }
        val first = Mat(count, DESCRIPTOR_LENGTH, CvType.CV_8UC1)
        val second = Mat(count, DESCRIPTOR_LENGTH, CvType.CV_8UC1)
        check(first.put(0, 0, values) == values.size)
        check(second.put(0, 0, values) == values.size)
        return first to second
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
