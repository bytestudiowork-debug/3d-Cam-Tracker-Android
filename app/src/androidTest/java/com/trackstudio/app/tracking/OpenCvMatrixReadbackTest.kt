package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat

@RunWith(AndroidJUnit4::class)
class OpenCvMatrixReadbackTest {
    @Test
    fun readsFiniteCv64fValuesForTheExpectedShape() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        val matrix = Mat(3, 1, CvType.CV_64F)
        try {
            val expected = doubleArrayOf(1.25, -2.5, 0.0)
            assertEquals(expected.size, matrix.put(0, 0, *expected))

            val actual = OpenCvMatrixReadback.readCv64F(matrix, 3, 1)
            assertNotNull(actual)
            assertArrayEquals(expected, actual!!, 0.0)
            assertNull(OpenCvMatrixReadback.readCv64F(matrix, 1, 3))
            assertNull(OpenCvMatrixReadback.readCv64F(matrix, 2, 2))
        } finally {
            matrix.release()
        }
    }

    @Test
    fun rejectsWrongDepthAndNonFiniteValues() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        val singlePrecision = Mat(3, 1, CvType.CV_32F)
        val nonFinite = Mat(3, 1, CvType.CV_64F)
        try {
            assertEquals(3, nonFinite.put(0, 0, Double.NaN, 0.0, 1.0))
            assertNull(OpenCvMatrixReadback.readCv64F(singlePrecision, 3, 1))
            assertNull(OpenCvMatrixReadback.readCv64F(nonFinite, 3, 1))
        } finally {
            singlePrecision.release()
            nonFinite.release()
        }
    }
}
