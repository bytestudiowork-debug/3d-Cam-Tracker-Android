package com.trackstudio.app.tracking

/** Converts a linear inlier index into coordinates for either Nx1 or 1xN masks. */
internal object OpenCvMaskLayout {
    fun coordinates(rows: Int, columns: Int, index: Int, expectedSize: Int): Pair<Int, Int>? {
        if (rows <= 0 || columns <= 0 || expectedSize < 0 ||
            rows.toLong() * columns.toLong() != expectedSize.toLong() ||
            index !in 0 until expectedSize
        ) {
            return null
        }
        return index / columns to index % columns
    }
}
