package com.trackstudio.app.tracking.glomap

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class GlomapNativeSolverInstrumentationTest {
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

    private fun distance(first: DoubleArray, second: DoubleArray): Double =
        sqrt((0 until 3).sumOf { (first[it] - second[it]) * (first[it] - second[it]) })
}
