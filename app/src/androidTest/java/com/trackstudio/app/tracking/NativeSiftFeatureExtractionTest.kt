package com.trackstudio.app.tracking

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint

@RunWith(AndroidJUnit4::class)
class NativeSiftFeatureExtractionTest {
    @Test
    fun fixedImageProducesAlignedDeterministicNativeFeatures() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        assertTrue("Native feature library is unavailable", NativeSiftExtractor.isAvailable)

        val pixels = fixturePixels()
        val image = Mat(IMAGE_SIZE, IMAGE_SIZE, CvType.CV_8UC1)
        val emptyMask = Mat()
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        val secondKeypoints = MatOfKeyPoint()
        val secondDescriptors = Mat()
        val extractor = AndroidSiftFeatureExtractor(MAX_FEATURES)
        try {
            assertEquals(pixels.size, image.put(0, 0, pixels))
            extractor.detectAndCompute(image, emptyMask, keypoints, descriptors)
            val firstPoints = keypoints.toArray()
            val firstBytes = ByteArray(descriptors.rows() * descriptors.cols())
            assertEquals(firstBytes.size, descriptors.get(0, 0, firstBytes))

            assertFalse("The fixed image produced no features", firstPoints.isEmpty())
            assertEquals(firstPoints.size, descriptors.rows())
            assertEquals(DESCRIPTOR_LENGTH, descriptors.cols())
            assertEquals(CvType.CV_8UC1, descriptors.type())
            assertTrue(firstPoints.all {
                it.pt.x.isFinite() && it.pt.y.isFinite() && it.size > 0f && it.angle.isFinite()
            })

            extractor.detectAndCompute(image, emptyMask, secondKeypoints, secondDescriptors)
            val secondPoints = secondKeypoints.toArray()
            val secondBytes = ByteArray(secondDescriptors.rows() * secondDescriptors.cols())
            assertEquals(secondBytes.size, secondDescriptors.get(0, 0, secondBytes))
            assertEquals(firstPoints.size, secondPoints.size)
            assertEquals(firstBytes.toList(), secondBytes.toList())
            firstPoints.indices.forEach { index ->
                assertEquals(firstPoints[index].pt.x, secondPoints[index].pt.x, 0f)
                assertEquals(firstPoints[index].pt.y, secondPoints[index].pt.y, 0f)
                assertEquals(firstPoints[index].size, secondPoints[index].size, 0f)
                assertEquals(firstPoints[index].angle, secondPoints[index].angle, 0f)
            }

            val nativeOutput = NativeSiftExtractor.extract(
                pixels, IMAGE_SIZE, IMAGE_SIZE, MAX_FEATURES
            ) ?: throw AssertionError("The native feature extractor returned no result.")
            assertEquals(firstPoints.size * 4, nativeOutput.keypointData.size)
            assertArrayEquals(firstBytes, nativeOutput.descriptorData)
            firstPoints.indices.forEach { index ->
                val offset = index * 4
                assertEquals(
                    nativeOutput.keypointData[offset],
                    firstPoints[index].pt.x.toFloat(),
                    0f
                )
                assertEquals(
                    nativeOutput.keypointData[offset + 1],
                    firstPoints[index].pt.y.toFloat(),
                    0f
                )
                assertEquals(2f * nativeOutput.keypointData[offset + 2], firstPoints[index].size, 0f)
                assertEquals(
                    Math.toDegrees(nativeOutput.keypointData[offset + 3].toDouble()).toFloat(),
                    firstPoints[index].angle,
                    0f
                )
            }

            Log.i(
                "SiftFeatureGolden",
                "FEATURE_GOLDEN count=${firstPoints.size} hash=${goldenHash(nativeOutput)}"
            )
        } finally {
            extractor.clear()
            image.release()
            emptyMask.release()
            keypoints.release()
            descriptors.release()
            secondKeypoints.release()
            secondDescriptors.release()
        }
    }

    @Test
    fun maskFiltersFeaturesByTheirPixelCenterWithoutBreakingDescriptorRows() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        assertTrue("Native feature library is unavailable", NativeSiftExtractor.isAvailable)

        val pixels = fixturePixels()
        val image = Mat(IMAGE_SIZE, IMAGE_SIZE, CvType.CV_8UC1)
        val emptyMask = Mat()
        val mask = Mat(IMAGE_SIZE, IMAGE_SIZE, CvType.CV_8UC1)
        val fullKeypoints = MatOfKeyPoint()
        val fullDescriptors = Mat()
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        val extractor = AndroidSiftFeatureExtractor(MAX_FEATURES)
        try {
            assertEquals(pixels.size, image.put(0, 0, pixels))
            extractor.detectAndCompute(image, emptyMask, fullKeypoints, fullDescriptors)
            val allFeatures = fullKeypoints.toArray()
            val allDescriptorBytes = ByteArray(fullDescriptors.rows() * fullDescriptors.cols())
            assertEquals(
                allDescriptorBytes.size,
                fullDescriptors.get(0, 0, allDescriptorBytes)
            )

            val maskPixels = ByteArray(pixels.size)
            for (y in 0 until IMAGE_SIZE) {
                for (x in IMAGE_SIZE / 2 until IMAGE_SIZE) {
                    maskPixels[y * IMAGE_SIZE + x] = 0xff.toByte()
                }
            }
            assertEquals(maskPixels.size, mask.put(0, 0, maskPixels))
            extractor.detectAndCompute(image, mask, keypoints, descriptors)

            val selected = keypoints.toArray()
            assertFalse("The masked half produced no features", selected.isEmpty())
            assertTrue(selected.all { it.pt.x >= IMAGE_SIZE / 2.0 })
            assertTrue(selected.size < allFeatures.size)
            assertEquals(selected.size, descriptors.rows())
            assertEquals(DESCRIPTOR_LENGTH, descriptors.cols())

            val selectedDescriptorBytes = ByteArray(descriptors.rows() * descriptors.cols())
            assertEquals(
                selectedDescriptorBytes.size,
                descriptors.get(0, 0, selectedDescriptorBytes)
            )
            var nextFullIndex = 0
            selected.forEachIndexed { selectedIndex, selectedPoint ->
                while (nextFullIndex < allFeatures.size &&
                    (allFeatures[nextFullIndex].pt.x != selectedPoint.pt.x ||
                        allFeatures[nextFullIndex].pt.y != selectedPoint.pt.y ||
                        allFeatures[nextFullIndex].size != selectedPoint.size ||
                        allFeatures[nextFullIndex].angle != selectedPoint.angle)
                ) {
                    nextFullIndex += 1
                }
                assertTrue(
                    "A masked feature was not present in the unmasked extraction",
                    nextFullIndex < allFeatures.size
                )
                val fullOffset = nextFullIndex * DESCRIPTOR_LENGTH
                val selectedOffset = selectedIndex * DESCRIPTOR_LENGTH
                assertArrayEquals(
                    allDescriptorBytes.copyOfRange(fullOffset, fullOffset + DESCRIPTOR_LENGTH),
                    selectedDescriptorBytes.copyOfRange(selectedOffset, selectedOffset + DESCRIPTOR_LENGTH)
                )
                nextFullIndex += 1
            }
        } finally {
            extractor.clear()
            image.release()
            emptyMask.release()
            mask.release()
            fullKeypoints.release()
            fullDescriptors.release()
            keypoints.release()
            descriptors.release()
        }
    }

    private fun fixturePixels(): ByteArray {
        val pixels = ByteArray(IMAGE_SIZE * IMAGE_SIZE)
        var state = 0x243f6a88
        for (index in pixels.indices) {
            state = state * 1664525 + 1013904223
            pixels[index] = (state ushr 24).toByte()
        }
        return pixels
    }

    private fun goldenHash(output: NativeSiftOutput): String {
        var hash = -3750763034362895579L
        fun mix(byte: Int) {
            hash = (hash xor (byte.toLong() and 0xffL)) * 1099511628211L
        }
        for (value in output.keypointData) {
            val bits = java.lang.Float.floatToIntBits(value)
            mix(bits)
            mix(bits ushr 8)
            mix(bits ushr 16)
            mix(bits ushr 24)
        }
        output.descriptorData.forEach { mix(it.toInt()) }
        return java.lang.Long.toUnsignedString(hash, 16).padStart(16, '0')
    }

    private companion object {
        const val IMAGE_SIZE = 256
        const val MAX_FEATURES = 400
        const val DESCRIPTOR_LENGTH = 128
    }
}
