package com.trackstudio.app.tracking

import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Point3

/** Fixed-intrinsics EPnP model estimator used by incremental PnP consensus refits. */
internal object EpnpPoseEstimator {
    data class Result(
        val rotation: DoubleArray,
        val translation: DoubleArray,
        val reprojectionError: Double
    )

    fun estimate(
        objectPoints: Array<Point3>,
        imagePoints: Array<Point>,
        cameraMatrix: Mat
    ): Result? {
        if (objectPoints.size < 4 || objectPoints.size != imagePoints.size ||
            cameraMatrix.empty() || cameraMatrix.rows() != 3 || cameraMatrix.cols() != 3
        ) return null

        val world = DoubleArray(objectPoints.size * 3)
        val image = DoubleArray(imagePoints.size * 2)
        objectPoints.forEachIndexed { index, point ->
            world[index * 3] = point.x
            world[index * 3 + 1] = point.y
            world[index * 3 + 2] = point.z
        }
        imagePoints.forEachIndexed { index, point ->
            image[index * 2] = point.x
            image[index * 2 + 1] = point.y
        }
        val intrinsics = doubleArrayOf(
            cameraMatrix.get(0, 0)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 2)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 2)?.firstOrNull() ?: Double.NaN
        )
        if (world.any { !it.isFinite() } || image.any { !it.isFinite() } ||
            intrinsics.any { !it.isFinite() } || intrinsics[0] <= 0.0 || intrinsics[3] <= 0.0
        ) return null

        val packed = nativeEstimate(world, image, intrinsics) ?: return null
        if (packed.size != 14 || packed[0] != 1.0 || packed.drop(1).any { !it.isFinite() }) {
            return null
        }
        val rotation = packed.copyOfRange(2, 11)
        val translation = packed.copyOfRange(11, 14)
        return Result(rotation, translation, packed[1])
    }

    private external fun nativeEstimate(
        objectPoints: DoubleArray,
        imagePoints: DoubleArray,
        intrinsics: DoubleArray
    ): DoubleArray?

    init {
        System.loadLibrary("trackstudio_camera")
    }
}
