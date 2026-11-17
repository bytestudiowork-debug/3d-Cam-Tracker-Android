package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.CameraProjectionMath
import com.trackstudio.app.tracking.PinholeIntrinsics
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point3

class ColmapObservationFilterTest {
    private val intrinsics = PinholeIntrinsics(
        fx = 700.0,
        fy = 730.0,
        skew = 15.0,
        cx = 640.0,
        cy = 360.0
    )
    private val point = Point3(0.0, 0.0, 5.0)

    @Test
    fun removesLargeReprojectionOutlierAndRetainsStablePoint() {
        val scene = scene(0 to 0.0, 1 to 1.0, 2 to 2.0)
        val track = track(10, scene, outlierFrame = 1)

        val result = filter(scene, track, affectedFrames = setOf(0))

        assertEquals(listOf(0, 2), result.tracksById.getValue(10).observations.map { it.frameIndex })
        assertEquals(2, result.landmarksByTrackId.getValue(10).observationCount)
        assertTrue(result.landmarksByTrackId.getValue(10).meanReprojectionErrorPx < 1e-9)
        assertEquals(listOf(1), result.removedObservationsByTrack.getValue(10).map { it.frameIndex })
        assertEquals(1, result.filteredObservationCount)
        assertTrue(result.deletedTrackIds.isEmpty())
    }

    @Test
    fun reprojectionOutliersAreRemovedBeforeTriangulationAnglePruning() {
        // The two good views have only a 0.57-degree baseline. The third view
        // would make the point stable, but its reprojection error is an outlier.
        val scene = scene(0 to 0.0, 1 to 0.05, 2 to 1.0)
        val track = track(11, scene, outlierFrame = 2)

        val result = filter(scene, track, affectedFrames = setOf(0))

        assertTrue(result.deletedTrackIds.contains(11))
        assertFalse(result.tracksById.containsKey(11))
        assertFalse(result.landmarksByTrackId.containsKey(11))
        assertEquals(3, result.filteredObservationCount)
    }

    @Test
    fun deletingAnOutlierThatLeavesOneObservationDeletesThePoint() {
        val scene = scene(0 to 0.0, 1 to 1.0)
        val track = track(12, scene, outlierFrame = 1)

        val result = filter(scene, track, affectedFrames = setOf(0))

        assertTrue(result.deletedTrackIds.contains(12))
        assertFalse(result.landmarksByTrackId.containsKey(12))
        assertEquals(2, result.filteredObservationCount)
    }

    @Test
    fun preservesNonRegisteredObservationsWithoutCountingThemAsPointSupport() {
        val scene = scene(0 to 0.0, 1 to 2.0, 2 to 3.0)
        val poses = scene.poses + (
            2 to scene.poses.getValue(2).copy(status = CameraPoseStatus.UNREGISTERED)
        )
        val track = CameraFeatureTrack(
            13,
            scene.centers.map { (frame, centerX) -> observation(frame, centerX) }
        )

        val result = ColmapObservationFilter.filter(
            tracksById = mapOf(13 to track),
            landmarksByTrackId = mapOf(13 to landmark(13, track.length)),
            posesByFrame = poses,
            intrinsics = intrinsics,
            affectedFrameIds = setOf(0)
        )

        assertEquals(listOf(0, 1, 2), result.tracksById.getValue(13).observations.map { it.frameIndex })
        assertEquals(2, result.landmarksByTrackId.getValue(13).observationCount)
        assertTrue(result.deletedTrackIds.isEmpty())
    }

    @Test
    fun onlyPointsObservedInAffectedFramesOrExplicitlyChangedAreFiltered() {
        val scene = scene(0 to 0.0, 1 to 1.0, 4 to 2.0, 5 to 3.0)
        val selected = track(20, scene, outlierFrame = 1)
        val untouched = CameraFeatureTrack(
            21,
            listOf(
                observation(4, scene.centers.getValue(4)),
                observation(5, scene.centers.getValue(5))
            )
        )
        val landmarks = mapOf(20 to landmark(20, selected.length), 21 to landmark(21, 2))

        val result = ColmapObservationFilter.filter(
            tracksById = mapOf(20 to selected, 21 to untouched),
            landmarksByTrackId = landmarks,
            posesByFrame = scene.poses,
            intrinsics = intrinsics,
            affectedFrameIds = setOf(0)
        )

        assertEquals(untouched, result.tracksById.getValue(21))
        assertEquals(2, result.landmarksByTrackId.getValue(21).observationCount)
        assertEquals(setOf(20), result.removedObservationsByTrack.keys)
    }

    @Test
    fun changedTrackIdsSelectPointsEvenWhenNoImageSetIsProvided() {
        val scene = scene(0 to 0.0, 1 to 2.0)
        val track = track(30, scene, outlierFrame = 1)

        val result = ColmapObservationFilter.filter(
            tracksById = mapOf(30 to track),
            landmarksByTrackId = mapOf(30 to landmark(30, track.length)),
            posesByFrame = scene.poses,
            intrinsics = intrinsics,
            changedTrackIds = setOf(30)
        )

        assertTrue(result.deletedTrackIds.contains(30))
    }

    @Test
    fun rejectsBehindCameraObservationEvenWhenItsProjectionMatchesPixel() {
        fun pose(frameIndex: Int, xTranslation: Double, zTranslation: Double) = CameraPose(
            frameIndex = frameIndex,
            rotation = doubleArrayOf(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ),
            translation = doubleArrayOf(xTranslation, 0.0, zTranslation),
            status = CameraPoseStatus.PNP_REGISTERED
        )

        val poses = mapOf(
            0 to pose(0, xTranslation = 0.0, zTranslation = 0.0),
            1 to pose(1, xTranslation = 0.0, zTranslation = 20.0),
            2 to pose(2, xTranslation = -1.0, zTranslation = 20.0)
        )
        val cameraPointsByFrame = listOf(
            0 to doubleArrayOf(0.0, 0.0, -10.0),
            1 to doubleArrayOf(0.0, 0.0, 10.0),
            2 to doubleArrayOf(-1.0, 0.0, 10.0)
        )
        val track = CameraFeatureTrack(
            40,
            cameraPointsByFrame.map { (frameIndex, cameraPoint) ->
                val pixel = CameraProjectionMath.project(cameraPoint, intrinsics)
                CameraFeatureObservation(frameIndex, pixel[0], pixel[1])
            }
        )

        val result = ColmapObservationFilter.filter(
            tracksById = mapOf(track.id to track),
            landmarksByTrackId = mapOf(
                track.id to CameraLandmark(
                    trackId = track.id,
                    position = Point3(0.0, 0.0, -10.0),
                    observationCount = track.length,
                    meanReprojectionErrorPx = 0.0
                )
            ),
            posesByFrame = poses,
            intrinsics = intrinsics,
            affectedFrameIds = setOf(0)
        )

        assertEquals(listOf(1, 2), result.tracksById.getValue(40).observations.map { it.frameIndex })
        assertEquals(listOf(0), result.removedObservationsByTrack.getValue(40).map { it.frameIndex })
        assertEquals(2, result.landmarksByTrackId.getValue(40).observationCount)
        assertFalse(result.deletedTrackIds.contains(40))
    }

    private fun filter(
        scene: Scene,
        track: CameraFeatureTrack,
        affectedFrames: Set<Int>
    ) = ColmapObservationFilter.filter(
        tracksById = mapOf(track.id to track),
        landmarksByTrackId = mapOf(track.id to landmark(track.id, track.length)),
        posesByFrame = scene.poses,
        intrinsics = intrinsics,
        affectedFrameIds = affectedFrames
    )

    private fun track(id: Int, scene: Scene, outlierFrame: Int) = CameraFeatureTrack(
        id,
        scene.centers.map { (frame, centerX) ->
            observation(frame, centerX, noiseX = if (frame == outlierFrame) 10.0 else 0.0)
        }
    )

    private fun observation(
        frame: Int,
        centerX: Double,
        noiseX: Double = 0.0
    ): CameraFeatureObservation {
        val pixel = CameraProjectionMath.project(
            doubleArrayOf(point.x - centerX, point.y, point.z), intrinsics
        )
        return CameraFeatureObservation(frame, pixel[0] + noiseX, pixel[1])
    }

    private fun landmark(id: Int, observations: Int) = CameraLandmark(
        trackId = id,
        position = Point3(point.x, point.y, point.z),
        observationCount = observations,
        meanReprojectionErrorPx = 5.0
    )

    private fun scene(vararg frameCenters: Pair<Int, Double>): Scene {
        val centers = frameCenters.toMap()
        val poses = centers.mapValues { (frame, centerX) ->
            CameraPose(
                frameIndex = frame,
                rotation = doubleArrayOf(
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0
                ),
                translation = doubleArrayOf(-centerX, 0.0, 0.0),
                status = CameraPoseStatus.PNP_REGISTERED
            )
        }
        return Scene(centers, poses)
    }

    private data class Scene(
        val centers: Map<Int, Double>,
        val poses: Map<Int, CameraPose>
    )
}
