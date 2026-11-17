package com.trackstudio.app.tracking

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
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
