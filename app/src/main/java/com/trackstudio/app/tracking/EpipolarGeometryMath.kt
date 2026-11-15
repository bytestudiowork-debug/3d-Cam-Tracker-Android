package com.trackstudio.app.tracking

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

internal object EpipolarGeometryMath {
    fun fundamentalFromEssential(
        essential: DoubleArray,
        cameraMatrix: DoubleArray
    ): DoubleArray? {
        if (essential.size != 9 || cameraMatrix.size != 9 ||
            essential.any { !it.isFinite() } || cameraMatrix.any { !it.isFinite() }
        ) return null
        val inverseCamera = invert3x3(cameraMatrix) ?: return null
        val fundamental = multiply3x3(
            transpose(inverseCamera), multiply3x3(essential, inverseCamera)
        )
        val scale = fundamental.maxOf { abs(it) }
        if (!scale.isFinite() || scale <= 1e-15) return null
        return DoubleArray(9) { fundamental[it] / scale }
    }

    fun symmetricEpipolarError(
        fundamental: DoubleArray,
        firstX: Double,
        firstY: Double,
        secondX: Double,
        secondY: Double
    ): Double {
        if (fundamental.size != 9 || fundamental.any { !it.isFinite() } ||
            !firstX.isFinite() || !firstY.isFinite() ||
            !secondX.isFinite() || !secondY.isFinite()
        ) return Double.POSITIVE_INFINITY
        val first = doubleArrayOf(firstX, firstY, 1.0)
        val second = doubleArrayOf(secondX, secondY, 1.0)
        val lineSecond = multiply3x1(fundamental, first)
        val lineFirst = multiply3x1(transpose(fundamental), second)
        val numerator = abs(second[0] * lineSecond[0] +
            second[1] * lineSecond[1] + lineSecond[2])
        val secondNorm = sqrt(lineSecond[0] * lineSecond[0] + lineSecond[1] * lineSecond[1])
        val firstNorm = sqrt(lineFirst[0] * lineFirst[0] + lineFirst[1] * lineFirst[1])
        if (!numerator.isFinite() || !secondNorm.isFinite() || !firstNorm.isFinite() ||
            secondNorm <= 1e-15 || firstNorm <= 1e-15
        ) return Double.POSITIVE_INFINITY
        return max(numerator / secondNorm, numerator / firstNorm)
    }

    private fun invert3x3(matrix: DoubleArray): DoubleArray? {
        val a = matrix[0]; val b = matrix[1]; val c = matrix[2]
        val d = matrix[3]; val e = matrix[4]; val f = matrix[5]
        val g = matrix[6]; val h = matrix[7]; val i = matrix[8]
        val cofactors = doubleArrayOf(
            e * i - f * h, c * h - b * i, b * f - c * e,
            f * g - d * i, a * i - c * g, c * d - a * f,
            d * h - e * g, b * g - a * h, a * e - b * d
        )
        val determinant = a * cofactors[0] + b * cofactors[3] + c * cofactors[6]
        if (!determinant.isFinite() || abs(determinant) <= 1e-15) return null
        return DoubleArray(9) { cofactors[it] / determinant }
            .takeIf { values -> values.all(Double::isFinite) }
    }

    private fun transpose(matrix: DoubleArray) = doubleArrayOf(
        matrix[0], matrix[3], matrix[6],
        matrix[1], matrix[4], matrix[7],
        matrix[2], matrix[5], matrix[8]
    )

    private fun multiply3x3(first: DoubleArray, second: DoubleArray): DoubleArray {
        val output = DoubleArray(9)
        for (row in 0..2) for (column in 0..2) for (k in 0..2) {
            output[row * 3 + column] += first[row * 3 + k] * second[k * 3 + column]
        }
        return output
    }

    private fun multiply3x1(matrix: DoubleArray, vector: DoubleArray): DoubleArray = doubleArrayOf(
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]
    )
}
