package com.trackstudio.app.tracking

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PrincipalPointRefinementPolicyTest {
    @Test
    fun broadCoverageAndParallaxPermitPrincipalPointRefinement() {
        val scene = scene(broadCoverage = true, movingCameras = true, rotatingCameras = true)
        val evidence = PrincipalPointRefinementPolicy.assess(
            tracks = scene.tracks,
            frameCentersById = scene.centers,
            rotationsByFrame = scene.rotations,
            pointsByTrackId = scene.points,
            imageWidth = 1280,
            imageHeight = 720
        )

        assertTrue(PrincipalPointRefinementPolicy.isObservable(evidence))
    }

    @Test
    fun concentratedImageCoverageRejectsPrincipalPointRefinement() {
        val scene = scene(broadCoverage = false, movingCameras = true, rotatingCameras = true)
        val evidence = PrincipalPointRefinementPolicy.assess(
            tracks = scene.tracks,
            frameCentersById = scene.centers,
            rotationsByFrame = scene.rotations,
            pointsByTrackId = scene.points,
            imageWidth = 1280,
            imageHeight = 720
        )

        assertFalse(PrincipalPointRefinementPolicy.isObservable(evidence))
    }

    @Test
    fun negligibleTranslationParallaxRejectsPrincipalPointRefinement() {
        val scene = scene(broadCoverage = true, movingCameras = false, rotatingCameras = true)
        val evidence = PrincipalPointRefinementPolicy.assess(
            tracks = scene.tracks,
            frameCentersById = scene.centers,
            rotationsByFrame = scene.rotations,
            pointsByTrackId = scene.points,
            imageWidth = 1280,
            imageHeight = 720
        )

        assertFalse(PrincipalPointRefinementPolicy.isObservable(evidence))
    }

    @Test
    fun nearLinearTranslationWithWeakRotationRejectsPrincipalPointRefinement() {
        val scene = scene(broadCoverage = true, movingCameras = true, rotatingCameras = false)
        val evidence = PrincipalPointRefinementPolicy.assess(
            tracks = scene.tracks,
            frameCentersById = scene.centers,
            rotationsByFrame = scene.rotations,
            pointsByTrackId = scene.points,
            imageWidth = 1280,
            imageHeight = 720
        )

        assertFalse(PrincipalPointRefinementPolicy.isObservable(evidence))
    }

    private data class Scene(
        val tracks: List<CameraFeatureTrack>,
        val centers: Map<Int, DoubleArray>,
        val rotations: Map<Int, DoubleArray>,
        val points: Map<Int, DoubleArray>
    )

    private fun scene(
        broadCoverage: Boolean,
        movingCameras: Boolean,
        rotatingCameras: Boolean
    ): Scene {
        val frameCount = 8
        val trackCount = 160
        val centers = (0 until frameCount).associateWith { frame ->
            doubleArrayOf(if (movingCameras) frame * 0.35 else 0.0, 0.0, 0.0)
        }
        val rotations = (0 until frameCount).associateWith { frame ->
            if (!rotatingCameras) {
                doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            } else {
                val phase = 2.0 * Math.PI * frame / frameCount
                rotation(
                    0.12 * kotlin.math.sin(phase),
                    0.17 * kotlin.math.sin(2.0 * phase),
                    -0.10 * kotlin.math.sin(phase)
                )
            }
        }
        val points = (0 until trackCount).associateWith { track ->
            doubleArrayOf((track % 16 - 7.5) * 0.05, (track / 16 - 4.5) * 0.04, 6.0)
        }
        val tracks = (0 until trackCount).map { track ->
            val cell = track % 16
            val column = cell % 4
            val row = cell / 4
            val x = if (broadCoverage) (column + 0.5) * 1280.0 / 4.0 else 640.0
            val y = if (broadCoverage) (row + 0.5) * 720.0 / 4.0 else 360.0
            CameraFeatureTrack(
                id = track,
                observations = (0 until frameCount).map { frame ->
                    CameraFeatureObservation(frame, x, y)
                }
            )
        }
        return Scene(tracks, centers, rotations, points)
    }

    private fun rotation(x: Double, y: Double, z: Double): DoubleArray {
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
}
