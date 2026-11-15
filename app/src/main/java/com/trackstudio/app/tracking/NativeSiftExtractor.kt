package com.trackstudio.app.tracking

internal class NativeSiftOutput(
    val keypointData: FloatArray,
    val descriptorData: ByteArray
)

internal object NativeSiftExtractor {
    val isAvailable: Boolean = try {
        System.loadLibrary("trackstudio_camera")
        true
    } catch (_: UnsatisfiedLinkError) {
        false
    }

    fun extract(
        grayscaleImage: ByteArray,
        width: Int,
        height: Int,
        maximumFeatures: Int
    ): NativeSiftOutput? {
        if (!isAvailable || width <= 0 || height <= 0 || maximumFeatures <= 0) return null
        val expectedSize = width.toLong() * height.toLong()
        if (expectedSize != grayscaleImage.size.toLong()) return null
        return nativeExtract(grayscaleImage, width, height, maximumFeatures)
    }

    fun clear() {
        if (isAvailable) nativeClear()
    }

    private external fun nativeExtract(
        grayscaleImage: ByteArray,
        width: Int,
        height: Int,
        maximumFeatures: Int
    ): NativeSiftOutput?

    private external fun nativeClear()
}
