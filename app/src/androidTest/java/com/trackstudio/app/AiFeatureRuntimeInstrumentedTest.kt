package com.trackstudio.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.trackstudio.ai.OnnxFeatureDetector
import com.trackstudio.ai.OnnxFeatureMatcher
import com.trackstudio.ai.TrackingFrame
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.random.Random

@RunWith(AndroidJUnit4::class)
class AiFeatureRuntimeInstrumentedTest {

    @Test
    fun packagedModelsMatchTranslatedFramesAndRestoreSourceCoordinates() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val width = 320
        val height = 240
        val channels = 3
        val shiftX = 10
        val shiftY = 10
        val firstPixels = ByteArray(width * height * channels)
        val random = Random(20261114)
        for (index in firstPixels.indices) {
            firstPixels[index] = random.nextInt(256).toByte()
        }

        // Translate by ten source pixels. With the 256-pixel inference limit,
        // this is an eight-pixel shift in model space and exercises coordinate
        // restoration after the detector resizes the source frame.
        val secondPixels = ByteArray(firstPixels.size)
        for (y in 0 until height - shiftY) {
            for (x in 0 until width - shiftX) {
                val sourceOffset = (y * width + x) * channels
                val targetOffset = ((y + shiftY) * width + x + shiftX) * channels
                System.arraycopy(firstPixels, sourceOffset, secondPixels, targetOffset, channels)
            }
        }

        val firstFrame = TrackingFrame(
            frameIndex = 0,
            timestampNs = 0L,
            width = width,
            height = height,
            pixels = firstPixels,
            channels = channels
        )
        val secondFrame = TrackingFrame(
            frameIndex = 1,
            timestampNs = 33_333_333L,
            width = width,
            height = height,
            pixels = secondPixels,
            channels = channels
        )

        OnnxFeatureDetector(
            context = context,
            maximumFeatures = 512,
            maximumInputDimension = 256,
            threadCount = 2
        ).use { detector ->
            val first = detector.detect(firstFrame)
            val second = detector.detect(secondFrame)
            assertFalse("The packaged extractor returned no first-frame features.", first.isEmpty())
            assertFalse("The packaged extractor returned no second-frame features.", second.isEmpty())
            assertTrue("The extractor exceeded its feature limit.", first.size <= 512 && second.size <= 512)

            for (observations in listOf(first, second)) {
                assertTrue(
                    "The extractor returned an invalid point or descriptor.",
                    observations.all {
                        it.x.isFinite() && it.y.isFinite() &&
                            it.x >= 0f && it.x < width.toFloat() &&
                            it.y >= 0f && it.y < height.toFloat() &&
                            it.descriptor?.size == 64 &&
                            it.descriptor?.all { value -> value.isFinite() } == true
                    }
                )
                assertTrue(
                    "The extractor returned non-normalized descriptors.",
                    observations.all { observation ->
                        val norm = kotlin.math.sqrt(
                            observation.descriptor!!.sumOf { it.toDouble() * it.toDouble() }
                        )
                        abs(norm - 1.0) < 1e-3
                    }
                )
            }

            OnnxFeatureMatcher(
                context = context,
                maximumFeatures = 512,
                minimumMatchConfidence = 0.1f,
                threadCount = 2
            ).use { matcher ->
                val matches = matcher.matchRaw(
                    firstKeypoints = first.toCoordinates(),
                    firstDescriptors = first.toDescriptors(),
                    firstCount = first.size,
                    firstWidth = width,
                    firstHeight = height,
                    secondKeypoints = second.toCoordinates(),
                    secondDescriptors = second.toDescriptors(),
                    secondCount = second.size,
                    secondWidth = width,
                    secondHeight = height
                )
                assertTrue(
                    "The packaged matcher found too few cross-frame correspondences.",
                    matches.size >= 3
                )
                assertTrue(
                    "The matcher returned an invalid feature index or confidence.",
                    matches.all {
                        it.firstFeatureIndex in first.indices &&
                            it.secondFeatureIndex in second.indices &&
                            it.confidence.isFinite() &&
                            it.confidence in 0f..1f
                    }
                )

                val horizontalOffsets = matches.map {
                    second[it.secondFeatureIndex].x - first[it.firstFeatureIndex].x
                }.sorted()
                val verticalOffsets = matches.map {
                    second[it.secondFeatureIndex].y - first[it.firstFeatureIndex].y
                }.sorted()
                val medianX = horizontalOffsets[horizontalOffsets.size / 2]
                val medianY = verticalOffsets[verticalOffsets.size / 2]

                assertTrue(
                    "Horizontal source-coordinate shift was not restored: $medianX px.",
                    abs(medianX - shiftX) < 4.5f
                )
                assertTrue(
                    "Vertical source-coordinate shift was not restored: $medianY px.",
                    abs(medianY - shiftY) < 4.5f
                )
            }
        }
    }

    private fun List<com.trackstudio.ai.FeatureObservation>.toCoordinates(): FloatArray =
        FloatArray(size * 2).also { values ->
            forEachIndexed { index, observation ->
                values[index * 2] = observation.x
                values[index * 2 + 1] = observation.y
            }
        }

    private fun List<com.trackstudio.ai.FeatureObservation>.toDescriptors(): FloatArray =
        FloatArray(size * 64).also { values ->
            forEachIndexed { index, observation ->
                val descriptor = requireNotNull(observation.descriptor)
                require(descriptor.size == 64)
                System.arraycopy(descriptor, 0, values, index * 64, 64)
            }
        }
}
