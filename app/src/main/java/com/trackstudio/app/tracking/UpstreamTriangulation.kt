package com.trackstudio.app.tracking

internal object UpstreamTriangulation {
    init {
        System.loadLibrary("trackstudio_reconstruction")
    }

    private external fun nativeTriangulate(
        firstProjection: DoubleArray,
        secondProjection: DoubleArray,
        firstX: Double,
        firstY: Double,
        secondX: Double,
        secondY: Double
    ): DoubleArray

    fun triangulate(
        firstProjection: DoubleArray,
        secondProjection: DoubleArray,
        firstX: Double,
        firstY: Double,
        secondX: Double,
        secondY: Double
    ): DoubleArray? {
        if (firstProjection.size != 12 || secondProjection.size != 12) return null
        if (!firstProjection.all(Double::isFinite) ||
            !secondProjection.all(Double::isFinite) ||
            !firstX.isFinite() || !firstY.isFinite() ||
            !secondX.isFinite() || !secondY.isFinite()
        ) return null

        return nativeTriangulate(
            firstProjection,
            secondProjection,
            firstX,
            firstY,
            secondX,
            secondY
        ).takeIf { result -> result.size == 3 && result.all(Double::isFinite) }
    }
}
