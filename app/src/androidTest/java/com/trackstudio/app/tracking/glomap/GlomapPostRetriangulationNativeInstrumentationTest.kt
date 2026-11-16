package com.trackstudio.app.tracking.glomap

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.cos
import kotlin.math.sin

@RunWith(AndroidJUnit4::class)
class GlomapPostRetriangulationNativeInstrumentationTest {
    @Test
    fun syntheticTracksPassThroughBothNativeAdjustmentsAndFinalFilters() {
        val intrinsics = CameraIntrinsics(
            fx = 800.0,
            skew = 0.0,
            cx = 320.0,
            fy = 800.0,
            cy = 240.0
        )
        val centers = linkedMapOf(
            0 to doubleArrayOf(0.0, 0.0, 0.0),
            1 to doubleArrayOf(0.45, 0.0, 0.0),
            2 to doubleArrayOf(0.12, 0.38, 0.0)
        )
        val rotations = centers.keys.associateWith { identity() }
        val truthPoints = (0 until 24).associate { index ->
            index to doubleArrayOf(
                ((index % 6) - 2.5) * 0.28,
                ((index / 6) - 1.5) * 0.25,
                4.0 + 0.07 * index
            )
        }
        val tracks = truthPoints.map { (trackId, point) ->
            CameraFeatureTrack(
                trackId,
                centers.map { (frame, center) ->
                    val depth = point[2] - center[2]
                    CameraFeatureObservation(
                        frameIndex = frame,
                        x = intrinsics.fx * (point[0] - center[0]) / depth + intrinsics.cx,
                        y = intrinsics.fy * (point[1] - center[1]) / depth + intrinsics.cy
                    )
                }
            )
        }
        val initialPoints = truthPoints.mapValues { (trackId, point) ->
            doubleArrayOf(
                point[0] + 0.02 * sin(0.4 * trackId),
                point[1] - 0.015 * cos(0.3 * trackId),
                point[2] + 0.03 * sin(0.2 * trackId)
            )
        }
        val initialState = GlomapPostRetriangulationRefinement.State(
            frameCentersById = centers,
            rotationsByFrame = rotations,
            pointsByTrackId = initialPoints,
            tracks = tracks,
            intrinsics = intrinsics
        )
        var solveCalls = 0
        val result = GlomapPostRetriangulationRefinement.refine(
            initialState = initialState,
            solveBundleAdjustment = { state, _ ->
                solveCalls++
                val cameraMatrix = Mat(3, 3, CvType.CV_64F)
                try {
                    cameraMatrix.put(0, 0, *state.intrinsics.toRowMajorMatrix())
                    GlomapGlobalBundleAdjuster().solve(
                        frameCentersById = state.frameCentersById,
                        rotationsByFrame = state.rotationsByFrame,
                        pointsByTrackId = state.pointsByTrackId,
                        tracks = state.tracks,
                        cameraMatrix = cameraMatrix,
                        imageWidth = 640,
                        imageHeight = 480,
                        optimizeRotations = true,
                        optimizeTranslation = true,
                        maxIterations = 45,
                        numThreads = 1
                    )
                } finally {
                    cameraMatrix.release()
                }
            }
        )

        assertEquals(2, solveCalls)
        assertEquals(2, result.acceptedBundleAdjustmentCount)
        assertTrue(result.firstBundleAdjustmentAccepted)
        assertTrue(result.secondBundleAdjustmentAccepted)
        assertTrue(result.firstBundleAdjustment!!.finalCost <= result.firstBundleAdjustment.initialCost)
        assertTrue(result.secondBundleAdjustment!!.finalCost <= result.secondBundleAdjustment.initialCost)
        assertEquals(24, result.state.tracks.size)
        assertTrue(result.state.tracks.all { it.observations.size == 3 })
        assertTrue(result.state.frameCentersById.values.all { it.all(Double::isFinite) })
        assertTrue(result.state.pointsByTrackId.values.all { it.all(Double::isFinite) })
    }

    private fun identity() = doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
}
