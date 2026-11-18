package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.trackstudio.app.tracking.glomap.GlomapImagePairInlierScorer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import java.util.Collections
import java.util.Random
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class GlobalPairEligibilityInstrumentationTest {
    private data class Correspondence(
        val firstX: Double,
        val firstY: Double,
        val secondX: Double,
        val secondY: Double
    )

    @Test
    fun weakFundamentalRansacDoesNotDiscardPoseLibAcceptableCandidate() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())

        val intrinsics = CameraIntrinsics(
            fx = FOCAL_X,
            skew = 0.0,
            cx = PRINCIPAL_X,
            fy = FOCAL_Y,
            cy = PRINCIPAL_Y
        )
        val rotation = rotation(doubleArrayOf(0.3, -0.5, 0.8), 0.12)
        val translation = doubleArrayOf(0.42, -0.12, 0.08)
        val pointRandom = Random(13L)
        val correspondences = ArrayList<Correspondence>(MATCH_COUNT)

        repeat(TRUE_INLIER_COUNT) { index ->
            val x = pointRandom.nextDouble() * 3.4 - 1.7
            val y = pointRandom.nextDouble() * 2.2 - 1.1
            val z = pointRandom.nextDouble() * 5.5 + 3.5
            val secondX = rotation[0] * x + rotation[1] * y + rotation[2] * z + translation[0]
            val secondY = rotation[3] * x + rotation[4] * y + rotation[5] * z + translation[1]
            val secondZ = rotation[6] * x + rotation[7] * y + rotation[8] * z + translation[2]
            correspondences += Correspondence(
                firstX = FOCAL_X * x / z + PRINCIPAL_X + pointRandom.nextGaussian() * NOISE_PX,
                firstY = FOCAL_Y * y / z + PRINCIPAL_Y + pointRandom.nextGaussian() * NOISE_PX,
                secondX = FOCAL_X * secondX / secondZ + PRINCIPAL_X +
                    pointRandom.nextGaussian() * NOISE_PX,
                secondY = FOCAL_Y * secondY / secondZ + PRINCIPAL_Y +
                    pointRandom.nextGaussian() * NOISE_PX
            )
        }
        repeat(MATCH_COUNT - TRUE_INLIER_COUNT) {
            correspondences += Correspondence(
                firstX = pointRandom.nextDouble() * IMAGE_WIDTH,
                firstY = pointRandom.nextDouble() * IMAGE_HEIGHT,
                secondX = pointRandom.nextDouble() * IMAGE_WIDTH,
                secondY = pointRandom.nextDouble() * IMAGE_HEIGHT
            )
        }
        Collections.shuffle(correspondences, Random(20261119L))

        val firstKeypoints = Array(MATCH_COUNT) { KeyPoint(0.0f, 0.0f, 1.0f) }
        val secondKeypoints = Array(MATCH_COUNT) { KeyPoint(0.0f, 0.0f, 1.0f) }
        correspondences.forEachIndexed { index, pair ->
            firstKeypoints[index] = KeyPoint(pair.firstX.toFloat(), pair.firstY.toFloat(), 1.0f)
            secondKeypoints[index] = KeyPoint(pair.secondX.toFloat(), pair.secondY.toFloat(), 1.0f)
        }

        val firstDescriptors = Mat(MATCH_COUNT, DESCRIPTOR_LENGTH, CvType.CV_8UC1)
        val secondDescriptors = Mat(MATCH_COUNT, DESCRIPTOR_LENGTH, CvType.CV_8UC1)
        val cameraMatrix = Mat(3, 3, CvType.CV_64F)
        val matcher = SiftDescriptorMatcher()
        try {
            val descriptorRandom = Random(20261120L)
            val descriptorBytes = ByteArray(MATCH_COUNT * DESCRIPTOR_LENGTH)
            val rawDescriptor = ByteArray(DESCRIPTOR_LENGTH)
            for (row in 0 until MATCH_COUNT) {
                descriptorRandom.nextBytes(rawDescriptor)
                val normalized = FloatArray(DESCRIPTOR_LENGTH) { column ->
                    (rawDescriptor[column].toInt() and 0xff).toFloat()
                }
                RootSiftDescriptorMath.normalizeDescriptorInPlace(normalized)
                for (column in normalized.indices) {
                    descriptorBytes[row * DESCRIPTOR_LENGTH + column] =
                        normalized[column].toInt().coerceIn(0, 255).toByte()
                }
            }
            assertEquals(descriptorBytes.size, firstDescriptors.put(0, 0, descriptorBytes))
            assertEquals(descriptorBytes.size, secondDescriptors.put(0, 0, descriptorBytes))
            cameraMatrix.put(0, 0, *intrinsics.toRowMajorMatrix())

            val first = GlobalReconstructionEngine.FrameFeatures(
                firstKeypoints, firstDescriptors, IMAGE_WIDTH, IMAGE_HEIGHT
            )
            val second = GlobalReconstructionEngine.FrameFeatures(
                secondKeypoints, secondDescriptors, IMAGE_WIDTH, IMAGE_HEIGHT
            )
            val engine = GlobalReconstructionEngine()
            val candidate = engine.estimateFundamentalPair(
                firstIndex = 0,
                secondIndex = 1,
                first = first,
                second = second,
                intrinsics = intrinsics,
                matcher = matcher
            ) ?: throw AssertionError("The raw-match budget should retain this candidate")

            assertEquals(MATCH_COUNT, candidate.matchCount)
            assertTrue(
                "The fixture must exercise the preexisting F-inlier rejection: " +
                    "${candidate.fundamentalInlierCount}/$MATCH_COUNT",
                candidate.fundamentalInlierCount < GlomapImagePairInlierScorer.MINIMUM_INLIERS
            )
            assertNull(candidate.fundamentalMatrix)

            val verified = engine.verifyPair(candidate, first, second, cameraMatrix)
                ?: throw AssertionError("PoseLib and essential rescoring should accept this pair")
            assertTrue(verified.inlierMatches.size >= GlomapImagePairInlierScorer.MINIMUM_INLIERS)
            assertTrue(
                verified.inlierMatches.size.toDouble() / verified.matchCount >=
                    GlomapImagePairInlierScorer.MINIMUM_INLIER_RATIO
            )
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
            t * x * z - s * y, t * z * y + s * x, t * z * z + c
        )
    }

    private companion object {
        const val MATCH_COUNT = 160
        const val TRUE_INLIER_COUNT = 40
        const val DESCRIPTOR_LENGTH = 128
        const val IMAGE_WIDTH = 1280
        const val IMAGE_HEIGHT = 720
        const val FOCAL_X = 780.0
        const val FOCAL_Y = 805.0
        const val PRINCIPAL_X = 640.0
        const val PRINCIPAL_Y = 360.0
        const val NOISE_PX = 0.25
    }
}
