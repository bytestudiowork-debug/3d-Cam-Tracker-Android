package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
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

            assertTrue("F-supported pose recovery was rejected: ${evaluation.reason}", evaluation.accepted)
            assertTrue(evaluation.candidate.fundamentalPoseRecovered)
            assertTrue(evaluation.candidate.essentialToFundamentalInlierRatio < 0.95)
            assertTrue(evaluation.candidate.inlierCount >= 100)
            assertTrue(evaluation.candidate.medianParallaxDegrees > 16.0)
            assertTrue(evaluation.candidate.homographyInlierRatio <= 0.8)
        } finally {
            matcher.clear()
            firstDescriptors.release()
            secondDescriptors.release()
            cameraMatrix.release()
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

    private companion object {
        const val TRUE_FOCAL = 800.0
        const val APP_FOCAL = 640.0
        const val PRINCIPAL_X = 640.0
        const val PRINCIPAL_Y = 360.0
        const val DESCRIPTOR_LENGTH = 128
    }
}
