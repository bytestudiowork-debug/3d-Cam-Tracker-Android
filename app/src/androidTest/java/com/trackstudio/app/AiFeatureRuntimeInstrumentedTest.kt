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
import kotlin.random.Random

@RunWith(AndroidJUnit4::class)
class AiFeatureRuntimeInstrumentedTest {

    @Test
    fun packagedModelsRunOnAndroidCpuAndProduceConsistentMatches() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val width = 320
        val height = 240
        val pixels = ByteArray(width * height * 3)
        val random = Random(20261112)
        for (index in pixels.indices) {
            pixels[index] = random.nextInt(256).toByte()
        }
        val frame = TrackingFrame(
            frameIndex = 0,
            timestampNs = 0L,
            width = width,
            height = height,
            pixels = pixels,
            channels = 3
        )

        OnnxFeatureDetector(
            context = context,
            maximumFeatures = 512,
            maximumInputDimension = 320,
            threadCount = 2
        ).use { detector ->
            val observations = detector.detect(frame)
            assertFalse("The packaged extractor returned no features.", observations.isEmpty())
            assertTrue("The extractor exceeded its feature limit.", observations.size <= 512)
            assertTrue(
                "The extractor returned an invalid point.",
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
                    kotlin.math.abs(norm - 1.0) < 1e-3
                }
            )

            OnnxFeatureMatcher(
                context = context,
                maximumFeatures = 512,
                minimumMatchConfidence = 0.1f,
                threadCount = 2
            ).use { matcher ->
                val coordinates = FloatArray(observations.size * 2)
                val descriptors = FloatArray(observations.size * 64)
                observations.forEachIndexed { index, observation ->
                    coordinates[index * 2] = observation.x
                    coordinates[index * 2 + 1] = observation.y
                    System.arraycopy(
                        observation.descriptor!!,
                        0,
                        descriptors,
                        index * 64,
                        64
                    )
                }
                val matches = matcher.matchRaw(
                    firstKeypoints = coordinates,
                    firstDescriptors = descriptors,
                    firstCount = observations.size,
                    firstWidth = width,
                    firstHeight = height,
                    secondKeypoints = coordinates,
                    secondDescriptors = descriptors,
                    secondCount = observations.size,
                    secondWidth = width,
                    secondHeight = height
                )
                assertFalse("The packaged matcher returned no self-matches.", matches.isEmpty())
                assertTrue(
                    "The matcher returned an invalid feature index.",
                    matches.all {
                        it.firstFeatureIndex in observations.indices &&
                            it.secondFeatureIndex in observations.indices &&
                            it.confidence.isFinite() &&
                            it.confidence in 0f..1f
                    }
                )
            }
        }
    }
}
