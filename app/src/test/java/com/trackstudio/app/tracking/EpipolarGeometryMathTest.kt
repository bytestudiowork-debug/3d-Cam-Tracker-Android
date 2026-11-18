package com.trackstudio.app.tracking

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class EpipolarGeometryMathTest {

    @Test
    fun fundamentalToEssentialConversionInvertsTheFundamentalTransformUpToScale() {
        val essential = doubleArrayOf(
            0.0, -0.12, 0.08,
            0.15, 0.0, -0.69,
            -0.04, 0.70, 0.0
        )
        val intrinsics = doubleArrayOf(
            780.0, 0.0, 640.0,
            0.0, 820.0, 360.0,
            0.0, 0.0, 1.0
        )
        val fundamental = EpipolarGeometryMath.fundamentalFromEssential(essential, intrinsics)
        assertNotNull(fundamental)
        fundamental ?: return

        val recovered = EpipolarGeometryMath.essentialFromFundamental(
            fundamental, intrinsics, intrinsics
        )

        assertNotNull(recovered)
        recovered ?: return
        val scale = essential.maxOf { abs(it) }
        assertArrayEquals(DoubleArray(9) { essential[it] / scale }, recovered, 1e-12)
    }

    @Test
    fun syntheticPixelResidualCanDisagreeWithPinnedColmapSampsonThreshold() {
        val focal = 800.0
        val intrinsics = doubleArrayOf(
            focal, 0.0, 640.0,
            0.0, focal, 360.0,
            0.0, 0.0, 1.0
        )
        val essential = doubleArrayOf(
            0.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0
        )
        val fundamental = EpipolarGeometryMath.fundamentalFromEssential(essential, intrinsics)
        assertNotNull(fundamental)
        fundamental ?: return

        val firstX = 0.25
        val firstY = 0.10
        val secondX = 0.45
        val secondY = 0.106
        val pixelError = EpipolarGeometryMath.symmetricEpipolarError(
            fundamental,
            firstX * focal + 640.0,
            firstY * focal + 360.0,
            secondX * focal + 640.0,
            secondY * focal + 360.0
        )

        // Test-only oracle for COLMAP c5f9cefc87e5dd596b638e4cee0ff543c7d14755:
        // normalized pinhole rays and squared Sampson residual. It is not used by production scoring.
        val lineInSecond = doubleArrayOf(
            essential[0] * firstX + essential[1] * firstY + essential[2],
            essential[3] * firstX + essential[4] * firstY + essential[5],
            essential[6] * firstX + essential[7] * firstY + essential[8]
        )
        val lineInFirst = doubleArrayOf(
            essential[0] * secondX + essential[3] * secondY + essential[6],
            essential[1] * secondX + essential[4] * secondY + essential[7],
            essential[2] * secondX + essential[5] * secondY + essential[8]
        )
        val algebraicError =
            secondX * lineInSecond[0] + secondY * lineInSecond[1] + lineInSecond[2]
        val gradientNormSquared =
            lineInSecond[0] * lineInSecond[0] + lineInSecond[1] * lineInSecond[1] +
                lineInFirst[0] * lineInFirst[0] + lineInFirst[1] * lineInFirst[1]
        val sampsonSquared = algebraicError * algebraicError / gradientNormSquared
        val colmapThresholdInNormalizedPinholeCoordinates = 4.0 / focal

        assertEquals(4.8, pixelError, 1e-9)
        assertEquals(1.8e-5, sampsonSquared, 1e-12)
        assertTrue("Android pixel residual should reject at 4 px", pixelError > 4.0)
        assertTrue(
            "Pinned COLMAP-style normalized Sampson support accepts this point",
            sampsonSquared <= colmapThresholdInNormalizedPinholeCoordinates *
                colmapThresholdInNormalizedPinholeCoordinates
        )
    }

    @Test
    fun fundamentalToEssentialRejectsDegenerateOrNonFiniteInputs() {
        val intrinsics = doubleArrayOf(
            800.0, 0.0, 640.0,
            0.0, 800.0, 360.0,
            0.0, 0.0, 1.0
        )

        assertNull(EpipolarGeometryMath.essentialFromFundamental(DoubleArray(8), intrinsics, intrinsics))
        assertNull(
            EpipolarGeometryMath.essentialFromFundamental(
                doubleArrayOf(Double.NaN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                intrinsics,
                intrinsics
            )
        )
        assertNull(
            EpipolarGeometryMath.essentialFromFundamental(
                DoubleArray(9), intrinsics, intrinsics
            )
        )
    }
}
