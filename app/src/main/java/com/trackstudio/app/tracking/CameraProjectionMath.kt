package com.trackstudio.app.tracking

internal data class PinholeIntrinsics(
    val fx: Double,
    val fy: Double,
    val skew: Double,
    val cx: Double,
    val cy: Double
) {
    fun toRowMajorMatrix(): DoubleArray = doubleArrayOf(
        fx, skew, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0
    )
}

internal object CameraProjectionMath {
    fun pixelToNormalized(x: Double, y: Double, intrinsics: PinholeIntrinsics): DoubleArray {
        val normalizedY = (y - intrinsics.cy) / intrinsics.fy
        val normalizedX = (x - intrinsics.cx - intrinsics.skew * normalizedY) / intrinsics.fx
        return doubleArrayOf(normalizedX, normalizedY, 1.0)
    }

    fun project(point: DoubleArray, intrinsics: PinholeIntrinsics): DoubleArray {
        val inverseZ = 1.0 / point[2]
        return doubleArrayOf(
            (intrinsics.fx * point[0] + intrinsics.skew * point[1]) * inverseZ + intrinsics.cx,
            intrinsics.fy * point[1] * inverseZ + intrinsics.cy
        )
    }

    fun reprojectionError(
        cameraX: Double,
        cameraY: Double,
        cameraZ: Double,
        observedX: Double,
        observedY: Double,
        intrinsics: PinholeIntrinsics
    ): Double? {
        if (!cameraZ.isFinite() || cameraZ <= 1e-6) return null
        val dx = (intrinsics.fx * cameraX + intrinsics.skew * cameraY) / cameraZ +
            intrinsics.cx - observedX
        val dy = intrinsics.fy * cameraY / cameraZ + intrinsics.cy - observedY
        return kotlin.math.sqrt(dx * dx + dy * dy).takeIf { it.isFinite() }
    }

    fun jacobian(point: DoubleArray, intrinsics: PinholeIntrinsics): Array<DoubleArray> {
        val inverseZ = 1.0 / point[2]
        val inverseZSquared = inverseZ * inverseZ
        return arrayOf(
            doubleArrayOf(
                intrinsics.fx * inverseZ,
                intrinsics.skew * inverseZ,
                -(intrinsics.fx * point[0] + intrinsics.skew * point[1]) * inverseZSquared
            ),
            doubleArrayOf(
                0.0,
                intrinsics.fy * inverseZ,
                -intrinsics.fy * point[1] * inverseZSquared
            )
        )
    }
}
