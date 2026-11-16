package com.trackstudio.app.tracking

/** Enforces one pixel coordinate system across every decoded frame in a solve. */
internal object TrackingFrameDimensions {
    const val MISMATCH_MESSAGE =
        "All decoded frames in a reconstruction must use the same image dimensions."

    fun validate(
        referenceSize: Pair<Int, Int>?,
        width: Int,
        height: Int
    ): Pair<Int, Int> {
        require(width > 0 && height > 0) { "Decoded frame dimensions must be positive." }
        val actualSize = width to height
        require(referenceSize == null || actualSize == referenceSize) { MISMATCH_MESSAGE }
        return referenceSize ?: actualSize
    }
}
