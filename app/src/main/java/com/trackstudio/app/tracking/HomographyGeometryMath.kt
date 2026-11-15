package com.trackstudio.app.tracking

import kotlin.math.abs
import kotlin.math.hypot

internal data class HomographyTransform(
    val matrix: DoubleArray,
    val inverse: DoubleArray
)

internal object HomographyGeometryMath {
    fun fromMatrix(values: DoubleArray): HomographyTransform? {
        val matrix = canonicalize(values) ?: return null
        val inverse = invert(matrix) ?: return null
        return HomographyTransform(matrix, canonicalize(inverse) ?: return null)
    }

    fun project(transform: HomographyTransform, x: Double, y: Double): DoubleArray? =
        project(transform.matrix, x, y)

    fun symmetricTransferError(
        transform: HomographyTransform,
        firstX: Double,
        firstY: Double,
        secondX: Double,
        secondY: Double
    ): Double {
        if (!firstX.isFinite() || !firstY.isFinite() ||
            !secondX.isFinite() || !secondY.isFinite()
        ) return Double.POSITIVE_INFINITY
        val forward = project(transform.matrix, firstX, firstY) ?: return Double.POSITIVE_INFINITY
        val backward = project(transform.inverse, secondX, secondY) ?: return Double.POSITIVE_INFINITY
        val forwardError = hypot(forward[0] - secondX, forward[1] - secondY)
        val backwardError = hypot(backward[0] - firstX, backward[1] - firstY)
        return maxOf(forwardError, backwardError)
            .takeIf { it.isFinite() } ?: Double.POSITIVE_INFINITY
    }

    private fun canonicalize(values: DoubleArray): DoubleArray? {
        if (values.size != 9 || values.any { !it.isFinite() }) return null
        val scale = values.maxOf { abs(it) }
        if (!scale.isFinite() || scale <= 1e-15) return null
        val normalized = DoubleArray(9) { values[it] / scale }
        val leading = normalized.firstOrNull { abs(it) > 1e-15 } ?: return null
        if (leading < 0.0) {
            for (index in normalized.indices) normalized[index] = -normalized[index]
        }
        return normalized
    }

    private fun invert(matrix: DoubleArray): DoubleArray? {
        val a = matrix[0]; val b = matrix[1]; val c = matrix[2]
        val d = matrix[3]; val e = matrix[4]; val f = matrix[5]
        val g = matrix[6]; val h = matrix[7]; val i = matrix[8]
        val cofactors = doubleArrayOf(
            e * i - f * h,
            c * h - b * i,
            b * f - c * e,
            f * g - d * i,
            a * i - c * g,
            c * d - a * f,
            d * h - e * g,
            b * g - a * h,
            a * e - b * d
        )
        val determinant = a * cofactors[0] + b * cofactors[3] + c * cofactors[6]
        if (!determinant.isFinite() || abs(determinant) <= 1e-14) return null
        val inverse = DoubleArray(9) { cofactors[it] / determinant }
        return inverse.takeIf { it.all(Double::isFinite) }
    }

    private fun project(matrix: DoubleArray, x: Double, y: Double): DoubleArray? {
        if (!x.isFinite() || !y.isFinite()) return null
        val denominator = matrix[6] * x + matrix[7] * y + matrix[8]
        if (!denominator.isFinite() || abs(denominator) <= 1e-12) return null
        val projectedX = (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator
        val projectedY = (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator
        if (!projectedX.isFinite() || !projectedY.isFinite()) return null
        return doubleArrayOf(projectedX, projectedY)
    }
}
