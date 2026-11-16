package com.trackstudio.app.tracking.glomap

import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Test

class GlomapGlobalPositioningProblemBuilderTest {
    @Test(expected = IllegalArgumentException::class)
    fun rejectsDuplicateTrackIdsBeforeBuildingConstraints() {
        val track = GlomapGlobalPositioningProblemBuilder.Track(
            trackId = 7,
            observations = listOf(observation(0), observation(1), observation(2))
        )

        GlomapGlobalPositioningProblemBuilder.build(listOf(track, track.copy()))
    }

    @Test
    fun repeatedFrameMeasurementsDoNotCountAsIndependentViews() {
        val problem = GlomapGlobalPositioningProblemBuilder.build(
            listOf(
                GlomapGlobalPositioningProblemBuilder.Track(
                    3,
                    listOf(observation(0), observation(0), observation(1))
                ),
                GlomapGlobalPositioningProblemBuilder.Track(
                    4,
                    listOf(observation(3), observation(3), observation(4), observation(5), observation(6))
                )
            )
        )

        assertEquals(3, problem.constraints.size)
        assertEquals(1, problem.acceptedTrackCount)
        assertEquals(1, problem.skippedShortTracks)
        assertEquals(4, problem.skippedInvalidObservations)
        assertEquals(setOf(4), problem.constraints.map { it.trackId }.toSet())
    }

    @Test
    fun createsOnePointToCameraConstraintPerValidObservation() {
        val track = GlomapGlobalPositioningProblemBuilder.Track(
            trackId = 7,
            observations = listOf(
                observation(0, calibrated = true),
                observation(1, calibrated = false),
                observation(2, calibrated = true)
            ),
            initializedPoint = doubleArrayOf(4.0, 0.0, 0.0)
        )
        val problem = GlomapGlobalPositioningProblemBuilder.build(listOf(track))
        assertEquals(3, problem.constraints.size)
        assertEquals(1, problem.acceptedTrackCount)
        assertEquals(0, problem.skippedShortTracks)
        assertArrayEquals(
            doubleArrayOf(2.0, 0.0, 0.0),
            problem.constraints.first().translationObservation,
            1e-12
        )
        assertEquals(1.0, problem.constraints.first().initialScale, 0.0)
        assertEquals(1e-5, problem.constraints.first().scaleLowerBound, 0.0)
        assertEquals(0.5, problem.constraints[1].lossScale, 0.0)
    }

    @Test
    fun initializedScaleMatchesUpstreamProjection() {
        assertEquals(
            0.5,
            GlomapGlobalPositioningProblemBuilder.initializeScale(
                doubleArrayOf(2.0, 0.0, 0.0),
                doubleArrayOf(4.0, 0.0, 0.0),
                doubleArrayOf(0.0, 0.0, 0.0)
            ),
            1e-12
        )
    }

    @Test
    fun skipsTracksBelowThreeViewsAndCountsInvalidObservations() {
        val short = GlomapGlobalPositioningProblemBuilder.Track(1, listOf(observation(0), observation(1)))
        val invalid = GlomapGlobalPositioningProblemBuilder.Track(
            2,
            listOf(observation(0), observation(1), observation(2, rotation = DoubleArray(9)))
        )
        val problem = GlomapGlobalPositioningProblemBuilder.build(listOf(short, invalid))
        assertEquals(0, problem.constraints.size)
        assertEquals(0, problem.acceptedTrackCount)
        assertEquals(2, problem.skippedShortTracks)
        assertEquals(1, problem.skippedInvalidObservations)
    }

    @Test
    fun requiresThreeValidObservationsAfterFiltering() {
        val track = GlomapGlobalPositioningProblemBuilder.Track(
            3,
            listOf(
                observation(0),
                observation(1),
                observation(2),
                observation(3, rotation = DoubleArray(9))
            )
        )

        val problem = GlomapGlobalPositioningProblemBuilder.build(listOf(track))

        assertEquals(3, problem.constraints.size)
        assertEquals(1, problem.acceptedTrackCount)
        assertEquals(0, problem.skippedShortTracks)
        assertEquals(1, problem.skippedInvalidObservations)
    }

    @Test
    fun initializesScalesToOneWhenScaleGenerationIsEnabled() {
        val track = GlomapGlobalPositioningProblemBuilder.Track(
            3,
            listOf(observation(0), observation(1), observation(2)),
            initializedPoint = doubleArrayOf(4.0, 0.0, 0.0)
        )
        val problem = GlomapGlobalPositioningProblemBuilder.build(
            listOf(track),
            generateScales = true
        )
        assertEquals(1.0, problem.constraints.first().initialScale, 0.0)
    }

    @Test
    fun transformsRotatedCameraDirectionIntoWorldFrame() {
        val rotatedObservation = GlomapGlobalPositioningProblemBuilder.CameraObservation(
            frameIndex = 0,
            rotation = doubleArrayOf(
                0.0, -1.0, 0.0,
                1.0, 0.0, 0.0,
                0.0, 0.0, 1.0
            ),
            cameraCenter = doubleArrayOf(0.0, 0.0, 0.0),
            undistortedFeatureDirection = doubleArrayOf(0.0, 2.0, 0.0),
            isCalibrated = true
        )
        val problem = GlomapGlobalPositioningProblemBuilder.build(
            listOf(
                GlomapGlobalPositioningProblemBuilder.Track(
                    trackId = 4,
                    observations = listOf(
                        rotatedObservation,
                        observation(1),
                        observation(2)
                    )
                )
            )
        )

        assertArrayEquals(
            doubleArrayOf(2.0, 0.0, 0.0),
            problem.constraints.first().translationObservation,
            1e-12
        )
    }

    private fun observation(
        frame: Int,
        calibrated: Boolean = true,
        rotation: DoubleArray = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        )
    ) = GlomapGlobalPositioningProblemBuilder.CameraObservation(
        frameIndex = frame,
        rotation = rotation,
        cameraCenter = doubleArrayOf(0.0, 0.0, 0.0),
        undistortedFeatureDirection = doubleArrayOf(2.0, 0.0, 0.0),
        isCalibrated = calibrated
    )
}
