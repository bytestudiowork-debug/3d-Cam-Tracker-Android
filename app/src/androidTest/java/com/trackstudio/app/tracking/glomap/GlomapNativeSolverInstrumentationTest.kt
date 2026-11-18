package com.trackstudio.app.tracking.glomap

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.OpenCvRuntime
import com.trackstudio.app.tracking.PrincipalPointRefinementPolicy
import org.junit.Assert.assertFalse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class GlomapNativeSolverInstrumentationTest {
    @Test
    fun pinnedGlobalBundleAdjustmentKeepsObservablePrincipalPointFixed() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())

        val imageWidth = 1000
        val imageHeight = 800
        val trueFx = 800.0
        val trueFy = 800.0
        val trueCx = 500.0
        val trueCy = 400.0
        val initialCx = 470.0
        val initialCy = 430.0
        val frameCenters = linkedMapOf(
            0 to doubleArrayOf(0.0, 0.0, 0.0),
            1 to doubleArrayOf(0.1, -0.05, 0.0),
            2 to doubleArrayOf(0.2, 0.1, 0.0),
            3 to doubleArrayOf(0.3, -0.12, 0.02),
            4 to doubleArrayOf(0.4, 0.08, 0.01),
            5 to doubleArrayOf(0.5, -0.15, 0.03),
            6 to doubleArrayOf(0.6, 0.12, 0.01),
            7 to doubleArrayOf(0.7, -0.08, 0.02)
        )
        val rotationAngles = listOf(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.03, 0.0, 0.0),
            doubleArrayOf(0.06, 0.02, 0.0),
            doubleArrayOf(0.09, 0.05, 0.0),
            doubleArrayOf(0.12, 0.08, 0.01),
            doubleArrayOf(0.15, 0.11, 0.02),
            doubleArrayOf(0.18, 0.14, 0.03),
            doubleArrayOf(0.21, 0.18, 0.04)
        )
        val rotations = rotationAngles.mapIndexed { frame, angles ->
            frame to rotationFromXyz(angles[0], angles[1], angles[2])
        }.toMap()
        val points = LinkedHashMap<Int, DoubleArray>()
        for (trackId in 0 until 120) {
            val depth = 6.0 + (trackId % 5) * 0.25
            val pixelX = 70.0 + (trackId % 12) * 78.0
            val pixelY = 60.0 + (trackId / 12) * 75.0
            points[trackId] = doubleArrayOf(
                (pixelX - trueCx) * depth / trueFx,
                (pixelY - trueCy) * depth / trueFy,
                depth
            )
        }
        val tracks = points.map { (trackId, point) ->
            CameraFeatureTrack(trackId, frameCenters.map { (frame, center) ->
                val dx = point[0] - center[0]
                val dy = point[1] - center[1]
                val dz = point[2] - center[2]
                val rotation = rotations.getValue(frame)
                val cameraX = rotation[0] * dx + rotation[1] * dy + rotation[2] * dz
                val cameraY = rotation[3] * dx + rotation[4] * dy + rotation[5] * dz
                val cameraZ = rotation[6] * dx + rotation[7] * dy + rotation[8] * dz
                CameraFeatureObservation(
                    frameIndex = frame,
                    x = trueFx * cameraX / cameraZ + trueCx,
                    y = trueFy * cameraY / cameraZ + trueCy
                )
            })
        }
        val evidence = PrincipalPointRefinementPolicy.assess(
            tracks = tracks,
            frameCentersById = frameCenters,
            rotationsByFrame = rotations,
            pointsByTrackId = points,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
        assertTrue("Synthetic fixture must satisfy the former refinement heuristic: $evidence",
            PrincipalPointRefinementPolicy.isObservable(evidence))

        val cameraMatrix = Mat(3, 3, CvType.CV_64F)
        try {
            cameraMatrix.put(
                0, 0,
                trueFx, 0.0, initialCx,
                0.0, trueFy, initialCy,
                0.0, 0.0, 1.0
            )
            val result = GlomapGlobalBundleAdjuster().solve(
                frameCentersById = frameCenters,
                rotationsByFrame = rotations,
                pointsByTrackId = points,
                tracks = tracks,
                cameraMatrix = cameraMatrix,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
                optimizeRotations = false,
                optimizeTranslation = false,
                maxIterations = 20,
                numThreads = 1
            ) ?: throw AssertionError("Native GLOMAP bundle adjustment rejected the synthetic scene")

            assertFalse("Pinned GLOMAP must not enable principal-point optimization",
                result.principalPointRefinementEnabled)
            assertEquals("cx must remain fixed at its input value", initialCx, result.intrinsics.cx, 0.0)
            assertEquals("cy must remain fixed at its input value", initialCy, result.intrinsics.cy, 0.0)
            assertTrue("The synthetic scene must retain positive-depth observations",
                result.positiveDepthRatio >= 0.9)
        } finally {
            cameraMatrix.release()
        }
    }

    @Test
    fun syntheticOnlyPointsProblemRoundTripsThroughJni() {
        val frameIds = listOf(41, 7, 93)
        val truthCenters = linkedMapOf(
            41 to doubleArrayOf(0.0, 0.0, 0.0),
            7 to doubleArrayOf(0.16, 0.015, 0.01),
            93 to doubleArrayOf(0.32, 0.06, 0.02)
        )
        val truthPoints = LinkedHashMap<Int, DoubleArray>()
        for (index in 0 until 18) {
            truthPoints[100 + index * 3] = doubleArrayOf(
                ((index % 6) - 2.5) * 0.09,
                ((index / 6) - 1.0) * 0.12,
                1.0 + 0.15 * index
            )
        }
        val identity = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
        val tracks = truthPoints.map { (trackId, point) ->
            GlomapGlobalPositioningProblemBuilder.Track(
                trackId = trackId,
                observations = frameIds.map { frameId ->
                    val center = truthCenters.getValue(frameId)
                    val depth = point[2] - center[2]
                    GlomapGlobalPositioningProblemBuilder.CameraObservation(
                        frameIndex = frameId,
                        rotation = identity,
                        cameraCenter = center,
                        undistortedFeatureDirection = doubleArrayOf(
                            (point[0] - center[0]) / depth,
                            (point[1] - center[1]) / depth,
                            1.0
                        ),
                        isCalibrated = true
                    )
                }
            )
        }
        val problem = GlomapGlobalPositioningProblemBuilder.build(tracks)
        assertEquals(truthPoints.size * frameIds.size, problem.constraints.size)

        val initialCenters = linkedMapOf(
            41 to doubleArrayOf(0.02, -0.01, 0.03),
            7 to doubleArrayOf(0.20, 0.04, -0.01),
            93 to doubleArrayOf(0.29, 0.10, 0.04)
        )
        val initialPoints = LinkedHashMap<Int, DoubleArray>()
        truthPoints.forEach { (trackId, point) ->
            val index = (trackId - 100) / 3
            initialPoints[trackId] = doubleArrayOf(
                point[0] + 0.025 * kotlin.math.sin(0.4 * index),
                point[1] - 0.02 * kotlin.math.cos(0.3 * index),
                point[2] + 0.035 * kotlin.math.sin(0.2 * index)
            )
        }

        var progressCalls = 0
        val result = GlomapNativeSolver().solve(
            problem = problem,
            frameCentersById = initialCenters,
            pointsByTrackId = initialPoints,
            maxIterations = 80,
            numThreads = 1,
            onIteration = { _, _ -> progressCalls++ }
        ) ?: throw AssertionError("Native GLOMAP positioning returned no result")

        assertEquals(frameIds, result.frameCentersById.keys.toList())
        assertEquals(truthPoints.keys.toList(), result.pointsByTrackId.keys.toList())
        assertTrue("JNI progress callback was not invoked", progressCalls > 0)
        assertTrue("Native positioning did not reduce its objective", result.finalCost < result.initialCost)
        assertEquals("First observation scale must remain fixed", 1.0, result.scales.first(), 1e-12)
        assertTrue(result.scales.all { it.isFinite() && it >= 1e-5 })
        assertTrue(result.frameCentersById.values.all { it.all(Double::isFinite) })
        assertTrue(result.pointsByTrackId.values.all { it.all(Double::isFinite) })

        for (frameId in frameIds.drop(1)) {
            val recovered = subtract(result.frameCentersById.getValue(frameId), result.frameCentersById.getValue(frameIds.first()))
            val expected = subtract(truthCenters.getValue(frameId), truthCenters.getValue(frameIds.first()))
            assertTrue("Recovered camera baseline for frame $frameId is inaccurate", distance(recovered, expected) < 0.03)
        }
        val referenceFrame = frameIds.first()
        val meanRelativePointError = truthPoints.map { (trackId, truthPoint) ->
            val recovered = subtract(result.pointsByTrackId.getValue(trackId), result.frameCentersById.getValue(referenceFrame))
            val expected = subtract(truthPoint, truthCenters.getValue(referenceFrame))
            distance(recovered, expected)
        }.average()
        assertTrue("Recovered relative landmark geometry is inaccurate", meanRelativePointError < 0.04)
    }

    private fun subtract(first: DoubleArray, second: DoubleArray) =
        DoubleArray(3) { first[it] - second[it] }

    private fun rotationFromXyz(x: Double, y: Double, z: Double): DoubleArray {
        val cx = kotlin.math.cos(x)
        val sx = kotlin.math.sin(x)
        val cy = kotlin.math.cos(y)
        val sy = kotlin.math.sin(y)
        val cz = kotlin.math.cos(z)
        val sz = kotlin.math.sin(z)
        return doubleArrayOf(
            cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
            sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
            -sy, cy * sx, cy * cx
        )
    }

    private fun distance(first: DoubleArray, second: DoubleArray): Double =
        sqrt((0 until 3).sumOf { (first[it] - second[it]) * (first[it] - second[it]) })
}
