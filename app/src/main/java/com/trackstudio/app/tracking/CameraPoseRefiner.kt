package com.trackstudio.app.tracking

import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Point3

/** Robustly refines one world-to-camera pose against its RANSAC inlier observations. */
internal object CameraPoseRefiner {
    internal const val MAX_ITERATIONS = 100
    internal const val HUBER_SCALE_PX = 1.0
    internal const val GRADIENT_TOLERANCE = 1.0

    internal data class Result(
        val rotationVector: DoubleArray,
        val translation: DoubleArray,
        val initialCost: Double,
        val finalCost: Double,
        val iterations: Int
    )

    fun refine(
        objectPoints: Array<Point3>,
        imagePoints: Array<Point>,
        cameraMatrix: Mat,
        initialRotationVector: Mat,
        initialTranslation: Mat
    ): Result? {
        require(objectPoints.size == imagePoints.size && objectPoints.size >= 4)
        require(cameraMatrix.rows() == 3 && cameraMatrix.cols() == 3 && !cameraMatrix.empty())

        val points3d = DoubleArray(objectPoints.size * 3)
        val points2d = DoubleArray(imagePoints.size * 2)
        objectPoints.forEachIndexed { index, point ->
            points3d[index * 3] = point.x
            points3d[index * 3 + 1] = point.y
            points3d[index * 3 + 2] = point.z
        }
        imagePoints.forEachIndexed { index, point ->
            points2d[index * 2] = point.x
            points2d[index * 2 + 1] = point.y
        }
        val intrinsics = doubleArrayOf(
            cameraMatrix.get(0, 0)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(0, 2)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 1)?.firstOrNull() ?: Double.NaN,
            cameraMatrix.get(1, 2)?.firstOrNull() ?: Double.NaN
        )
        val rotation = DoubleArray(3)
        val translation = DoubleArray(3)
        if (initialRotationVector.get(0, 0, rotation) != rotation.size ||
            initialTranslation.get(0, 0, translation) != translation.size
        ) return null

        val packed = nativeRefine(
            points3d,
            points2d,
            intrinsics,
            rotation + translation,
            HUBER_SCALE_PX,
            MAX_ITERATIONS,
            GRADIENT_TOLERANCE
        )
        if (packed.size != 10 || packed[0] != 1.0 ||
            packed.drop(1).any { !it.isFinite() } ||
            packed[2] > packed[1] + 1e-9 * maxOf(1.0, packed[1])
        ) return null

        return Result(
            rotationVector = packed.copyOfRange(4, 7),
            translation = packed.copyOfRange(7, 10),
            initialCost = packed[1],
            finalCost = packed[2],
            iterations = packed[3].toInt()
        )
    }

    private external fun nativeRefine(
        objectPoints: DoubleArray,
        imagePoints: DoubleArray,
        intrinsics: DoubleArray,
        initialPose: DoubleArray,
        huberScalePx: Double,
        maxIterations: Int,
        gradientTolerance: Double
    ): DoubleArray

    init {
        System.loadLibrary("trackstudio_camera")
    }
}
