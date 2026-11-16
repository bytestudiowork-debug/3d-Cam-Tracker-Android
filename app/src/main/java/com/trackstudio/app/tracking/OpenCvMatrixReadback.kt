package com.trackstudio.app.tracking

import org.opencv.core.CvType
import org.opencv.core.Mat

internal object OpenCvMatrixReadback {
    fun readCv64F(matrix: Mat, rows: Int, columns: Int): DoubleArray? {
        if (rows <= 0 || columns <= 0 || matrix.empty() ||
            matrix.rows() != rows || matrix.cols() != columns ||
            matrix.type() != CvType.CV_64F || matrix.channels() != 1
        ) return null

        val values = DoubleArray(rows * columns)
        val expectedByteCount = values.size * java.lang.Double.BYTES
        if (matrix.get(0, 0, values) != expectedByteCount || values.any { !it.isFinite() }) {
            return null
        }
        return values
    }
}
