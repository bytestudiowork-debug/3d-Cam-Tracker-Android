package com.trackstudio.app.tracking

internal object CameraPoseGeometry {
    fun cameraVectorToWorld(rotationWorldToCameraRowMajor: DoubleArray, x: Double, y: Double, z: Double): DoubleArray {
        require(rotationWorldToCameraRowMajor.size == 9)
        val r = rotationWorldToCameraRowMajor
        return doubleArrayOf(
            r[0] * x + r[3] * y + r[6] * z,
            r[1] * x + r[4] * y + r[7] * z,
            r[2] * x + r[5] * y + r[8] * z
        )
    }

    fun cameraCenterWorld(pose: CameraPose): DoubleArray {
        require(pose.translation.size == 3)
        val center = cameraVectorToWorld(
            pose.rotation,
            pose.translation[0],
            pose.translation[1],
            pose.translation[2]
        )
        return doubleArrayOf(-center[0], -center[1], -center[2])
    }
}
