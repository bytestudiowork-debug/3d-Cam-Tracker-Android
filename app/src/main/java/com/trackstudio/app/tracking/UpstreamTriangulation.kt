package com.trackstudio.app.tracking

internal object UpstreamTriangulation {
    init {
        System.loadLibrary("trackstudio_reconstruction")
    }

    private external fun nativeTriangulateMultiView(
        projections: DoubleArray,
        cameraRays: DoubleArray,
        viewCount: Int
    ): DoubleArray

    private external fun nativeTriangulate(
        firstProjection: DoubleArray,
        secondProjection: DoubleArray,
        firstX: Double,
        firstY: Double,
        secondX: Double,
        secondY: Double
    ): DoubleArray

    fun triangulateMultiView(
        projections: DoubleArray,
        cameraRays: DoubleArray,
        viewCount: Int
    ): DoubleArray? {
        if (viewCount < 2 || projections.size != viewCount * 12 ||
            cameraRays.size != viewCount * 3
        ) return null
        if (!projections.all(Double::isFinite) ||
            !cameraRays.all(Double::isFinite)
        ) return null
        for (view in 0 until viewCount) {
            val offset = view * 3
            val normSquared = cameraRays[offset] * cameraRays[offset] +
                cameraRays[offset + 1] * cameraRays[offset + 1] +
                cameraRays[offset + 2] * cameraRays[offset + 2]
            if (!normSquared.isFinite() || normSquared <= 1e-12) return null
        }
        return nativeTriangulateMultiView(projections, cameraRays, viewCount)
            .takeIf { result -> result.size == 3 && result.all(Double::isFinite) }
    }

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
