package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar

@RunWith(AndroidJUnit4::class)
class IncrementalFrameOwnershipTest {
    @Test
    fun unsupportedChannelReleasesSourceWhenGrayscaleConversionFails() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        val source = Mat(12, 12, CvType.CV_8UC2)
        try {
            val failure = try {
                withSourceReleaseOnFailure(source) { IncrementalSfMSolver().toGray(it) }
                null
            } catch (error: IllegalArgumentException) {
                error
            }

            assertNotNull("Unsupported channels should fail grayscale conversion", failure)
            assertEquals("Unsupported image channel count: 2", failure?.message)
            assertTrue("Failed source Mat must be released", source.empty())
        } finally {
            source.release()
        }
    }

    @Test
    fun validColorFrameConvertsWithoutChangingSuccessfulCleanupOwnership() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        val source = Mat(12, 12, CvType.CV_8UC3).also {
            it.setTo(Scalar(24.0, 48.0, 72.0))
        }
        var gray: Mat? = null
        try {
            val converted = withSourceReleaseOnFailure(source) { IncrementalSfMSolver().toGray(it) }
            gray = converted
            assertEquals(1, converted.channels())
            assertEquals(12, converted.rows())
            assertEquals(12, converted.cols())
            assertFalse("Successful source remains owned by the caller's normal finally", source.empty())
        } finally {
            gray?.release()
            source.release()
        }
    }
}
