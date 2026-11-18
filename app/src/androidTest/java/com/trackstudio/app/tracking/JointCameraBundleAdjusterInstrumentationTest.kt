package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat

@RunWith(AndroidJUnit4::class)
class JointCameraBundleAdjusterInstrumentationTest {
    @Test
    fun refinesTwoFrameWindowWithThirtyValidTwoViewLandmarks() {
        val intrinsics = doubleArrayOf(800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0)
        val centers = linkedMapOf(
            0 to doubleArrayOf(0.0, 0.0, 0.0),
            1 to doubleArrayOf(0.5, 0.0, 0.0)
        )
        val identity = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
        val rotations = centers.keys.associateWith { identity.copyOf() }
        val points = (0 until 30).associate { trackId ->
            trackId to doubleArrayOf(
                ((trackId % 6) - 2.5) * 0.28,
                ((trackId / 6) - 2.0) * 0.24,
                4.0 + 0.08 * trackId
            )
        }
        val tracks = points.map { (trackId, point) ->
            CameraFeatureTrack(
                trackId,
                centers.map { (frameId, center) ->
                    CameraFeatureObservation(
                        frameIndex = frameId,
                        x = intrinsics[0] * (point[0] - center[0]) / point[2] + intrinsics[2],
                        y = intrinsics[4] * (point[1] - center[1]) / point[2] + intrinsics[5]
                    )
                }
            )
        }
        val cameraMatrix = Mat(3, 3, CvType.CV_64F)
        try {
            cameraMatrix.put(0, 0, *intrinsics)
            val result = JointCameraBundleAdjuster().solve(
                frameCentersById = centers,
                rotationsByFrame = rotations,
                pointsByTrackId = points,
                tracks = tracks,
                cameraMatrix = cameraMatrix,
                imageWidth = 640,
                imageHeight = 480,
                maxIterations = 8,
                numThreads = 1,
                optimizeFocalLength = false,
                reprojectionLoss = JointCameraBundleAdjuster.ReprojectionLoss.SOFT_L1
            )

            assertNotNull("Two-frame local bundle adjustment should reach native Ceres", result)
            assertEquals(2, result!!.frameCentersById.size)
            assertEquals(30, result.pointsByTrackId.size)
            assertTrue(result.finalCost <= result.initialCost + 1e-9)
            assertEquals(1.0, result.positiveDepthRatio, 1e-8)
        } finally {
            cameraMatrix.release()
        }
    }
}
