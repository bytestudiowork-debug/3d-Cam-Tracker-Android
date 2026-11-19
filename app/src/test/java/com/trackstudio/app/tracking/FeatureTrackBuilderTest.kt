package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.KeyPoint

class FeatureTrackBuilderTest {

    @Test
    fun reversePassFillsGapWithoutDuplicatingInitializationObservations() {
        val builder = FeatureTrackBuilder()
        val frame0 = keypoint(10f, 10f)
        val frame1 = keypoint(12f, 10f)
        val frame2 = keypoint(14f, 10f)
        val frame3 = keypoint(16f, 10f)

        builder.seedPair(
            firstFrame = 0,
            secondFrame = 2,
            firstKeypoints = arrayOf(frame0),
            secondKeypoints = arrayOf(frame2),
            inlierMatches = listOf(0 to 0)
        )
        builder.addInlierMatches(
            frameIndex = 3,
            previousKeypoints = arrayOf(frame2),
            currentKeypoints = arrayOf(frame3),
            matches = listOf(0 to 0)
        )

        builder.resetToFrame(2)
        builder.addInlierMatchesBackward(
            frameIndex = 1,
            laterKeypoints = arrayOf(frame2),
            earlierKeypoints = arrayOf(frame1),
            matches = listOf(0 to 0)
        )
        builder.addInlierMatchesBackward(
            frameIndex = 0,
            laterKeypoints = arrayOf(frame1),
            earlierKeypoints = arrayOf(frame0),
            matches = listOf(0 to 0)
        )

        val track = builder.build().single()
        assertEquals(listOf(0, 1, 2, 3), track.observations.map { it.frameIndex })
        assertEquals(4, track.observations.map { it.frameIndex }.distinct().size)
        assertEquals(4, track.length)
    }

    @Test
    fun seedPairDoesNotReuseFirstFrameFeature() {
        val builder = FeatureTrackBuilder()

        builder.seedPair(
            firstFrame = 0,
            secondFrame = 1,
            firstKeypoints = arrayOf(keypoint(10f, 10f)),
            secondKeypoints = arrayOf(keypoint(20f, 20f), keypoint(30f, 30f)),
            inlierMatches = listOf(0 to 0, 0 to 1)
        )

        val track = builder.build().single()
        assertEquals(listOf(0, 1), track.observations.map { it.frameIndex })
        assertEquals(20.0, track.observations.last().x, 0.0)
    }

    @Test
    fun forwardExtensionDoesNotReusePreviousFrameFeature() {
        val builder = FeatureTrackBuilder()

        builder.addInlierMatches(
            frameIndex = 1,
            previousKeypoints = arrayOf(keypoint(10f, 10f)),
            currentKeypoints = arrayOf(keypoint(20f, 20f), keypoint(30f, 30f)),
            matches = listOf(0 to 0, 0 to 1)
        )

        val track = builder.build().single()
        assertEquals(listOf(0, 1), track.observations.map { it.frameIndex })
        assertEquals(20.0, track.observations.last().x, 0.0)
    }

    @Test
    fun backwardExtensionDoesNotReuseLaterFrameFeature() {
        val builder = FeatureTrackBuilder()

        builder.addInlierMatchesBackward(
            frameIndex = 1,
            laterKeypoints = arrayOf(keypoint(20f, 20f)),
            earlierKeypoints = arrayOf(keypoint(10f, 10f), keypoint(15f, 15f)),
            matches = listOf(0 to 0, 0 to 1)
        )

        val track = builder.build().single()
        assertEquals(listOf(1, 2), track.observations.map { it.frameIndex })
        assertEquals(10.0, track.observations.first().x, 0.0)
    }

    @Test
    fun reversePassCreatesTrackWhenFeatureWasNotInInitializationPair() {
        val builder = FeatureTrackBuilder()
        val frame2 = keypoint(20f, 20f)
        val frame1 = keypoint(18f, 20f)

        builder.seedPair(
            firstFrame = 0,
            secondFrame = 2,
            firstKeypoints = arrayOf(keypoint(10f, 10f)),
            secondKeypoints = arrayOf(keypoint(11f, 10f)),
            inlierMatches = listOf(0 to 0)
        )
        builder.resetToFrame(2)
        builder.addInlierMatchesBackward(
            frameIndex = 1,
            laterKeypoints = arrayOf(keypoint(11f, 10f), frame2),
            earlierKeypoints = arrayOf(keypoint(10f, 10f), frame1),
            matches = listOf(1 to 1)
        )

        val newTrack = builder.build().first { it.observations.first().x == 18.0 }
        assertEquals(listOf(1, 2), newTrack.observations.map { it.frameIndex })
    }


    @Test
    fun keyframeRelocalizationReconnectsExistingTrackAfterContinuityBreak() {
        val builder = FeatureTrackBuilder()
        val first = keypoint(10f, 10f)
        val anchor = keypoint(12f, 10f)
        val recovered = keypoint(18f, 11f)

        builder.seedPair(
            firstFrame = 0,
            secondFrame = 2,
            firstKeypoints = arrayOf(first),
            secondKeypoints = arrayOf(anchor),
            inlierMatches = listOf(0 to 0)
        )
        builder.breakContinuity()
        builder.addInlierMatchesFromFrame(
            sourceFrame = 2,
            frameIndex = 5,
            sourceKeypoints = arrayOf(anchor),
            currentKeypoints = arrayOf(recovered),
            matches = listOf(0 to 0)
        )

        val track = builder.build().single()
        assertEquals(setOf(0, 2, 5), track.observations.map { it.frameIndex }.toSet())
        assertEquals(18.0, track.observations.last().x, 0.0)
    }

    @Test
    fun keyframeRelocalizationReplacesProvisionalCurrentFeatureAssignment() {
        val builder = FeatureTrackBuilder()
        val first = keypoint(10f, 10f)
        val anchor = keypoint(12f, 10f)
        val adjacentPrevious = keypoint(13f, 10f)
        val recovered = keypoint(14f, 11f)

        builder.seedPair(
            firstFrame = 0,
            secondFrame = 1,
            firstKeypoints = arrayOf(first),
            secondKeypoints = arrayOf(anchor),
            inlierMatches = listOf(0 to 0)
        )
        builder.breakContinuity()
        builder.addInlierMatches(
            frameIndex = 3,
            previousKeypoints = arrayOf(adjacentPrevious),
            currentKeypoints = arrayOf(recovered),
            matches = listOf(0 to 0)
        )
        builder.addInlierMatchesFromFrame(
            sourceFrame = 1,
            frameIndex = 3,
            sourceKeypoints = arrayOf(anchor),
            currentKeypoints = arrayOf(recovered),
            matches = listOf(0 to 0)
        )

        val tracks = builder.build()
        assertEquals(1, tracks.size)
        assertEquals(listOf(0, 1, 3), tracks.single().observations.map { it.frameIndex })
        assertEquals(14.0, tracks.single().observations.last().x, 0.0)
        assertEquals(11.0, tracks.single().observations.last().y, 0.0)
    }

    @Test
    fun keyframeRelocalizationContinuesTrackIntoNextAdjacentFrame() {
        val builder = FeatureTrackBuilder()
        val first = keypoint(10f, 10f)
        val anchor = keypoint(12f, 10f)
        val recovered = keypoint(18f, 11f)
        val next = keypoint(20f, 12f)

        builder.seedPair(
            firstFrame = 0,
            secondFrame = 1,
            firstKeypoints = arrayOf(first),
            secondKeypoints = arrayOf(anchor),
            inlierMatches = listOf(0 to 0)
        )
        builder.breakContinuity()
        builder.addInlierMatchesFromFrame(
            sourceFrame = 1,
            frameIndex = 3,
            sourceKeypoints = arrayOf(anchor),
            currentKeypoints = arrayOf(recovered),
            matches = listOf(0 to 0)
        )
        builder.addInlierMatches(
            frameIndex = 4,
            previousKeypoints = arrayOf(recovered),
            currentKeypoints = arrayOf(next),
            matches = listOf(0 to 0)
        )

        val tracks = builder.build()
        assertEquals(1, tracks.size)
        assertEquals(listOf(0, 1, 3, 4), tracks.single().observations.map { it.frameIndex })
        assertEquals(20.0, tracks.single().observations.last().x, 0.0)
    }

    @Test
    fun replacedMapTracksRestoreFeatureAssignmentsForTheNextFrame() {
        val builder = FeatureTrackBuilder()
        builder.seedPair(
            firstFrame = 0,
            secondFrame = 1,
            firstKeypoints = arrayOf(keypoint(10f, 10f)),
            secondKeypoints = arrayOf(keypoint(20f, 10f)),
            inlierMatches = listOf(0 to 0)
        )
        val original = builder.build().single()

        builder.replaceMapTracks(
            replacedTrackIds = setOf(original.id),
            updatedTracks = listOf(original),
            removedTrackIds = emptySet(),
            activeFrameIndex = 1
        )
        builder.addInlierMatches(
            frameIndex = 2,
            previousKeypoints = arrayOf(keypoint(20f, 10f)),
            currentKeypoints = arrayOf(keypoint(30f, 10f)),
            matches = listOf(0 to 0)
        )

        val updated = builder.build().single()
        assertEquals(listOf(0, 1, 2), updated.observations.map { it.frameIndex })
        assertEquals(listOf(0, 0, 0), updated.observations.map { it.featureIndex })
    }

    @Test
    fun completedMapTrackTakesOwnershipFromRetainedFeatureTrack() {
        val builder = FeatureTrackBuilder()
        builder.seedPair(
            firstFrame = 0,
            secondFrame = 1,
            firstKeypoints = arrayOf(keypoint(10f, 10f), keypoint(11f, 10f)),
            secondKeypoints = arrayOf(keypoint(20f, 10f), keypoint(21f, 10f)),
            inlierMatches = listOf(0 to 0, 1 to 1)
        )
        builder.addInlierMatches(
            frameIndex = 2,
            previousKeypoints = arrayOf(keypoint(20f, 10f), keypoint(21f, 10f)),
            currentKeypoints = arrayOf(keypoint(31f, 10f)),
            matches = listOf(1 to 0)
        )

        val tracksBeforeUpdate = builder.build().associateBy { it.id }
        val mapTrack = tracksBeforeUpdate.getValue(0)
        val retainedTrack = tracksBeforeUpdate.getValue(1)
        assertEquals(listOf(0, 1), mapTrack.observations.map { it.frameIndex })
        assertEquals(listOf(0, 1, 2), retainedTrack.observations.map { it.frameIndex })

        builder.replaceMapTracks(
            replacedTrackIds = setOf(mapTrack.id),
            updatedTracks = listOf(
                mapTrack.copy(observations = mapTrack.observations + retainedTrack.observations.last())
            ),
            removedTrackIds = emptySet(),
            activeFrameIndex = 2
        )

        val updatedTracks = builder.build().associateBy { it.id }
        assertEquals(listOf(0, 1, 2), updatedTracks.getValue(mapTrack.id).observations.map { it.frameIndex })
        assertEquals(listOf(0, 1), updatedTracks.getValue(retainedTrack.id).observations.map { it.frameIndex })
        val featureOwners = builder.build().flatMap { track ->
            track.observations.map { it.frameIndex to it.featureIndex }
        }
        assertEquals(featureOwners.size, featureOwners.distinct().size)
    }

    @Test
    fun progressLabelAndFractionAreBounded() {
        val progress = CameraTrackingProgress(
            frameIndex = 240,
            frameCount = 200,
            phase = "Registering cameras",
            overallFraction = 1.2f,
            completedFrames = 240,
            totalFrames = 200
        )

        assertEquals("200/200 frames", progress.frameLabel)
        assertEquals(1f, progress.fraction)
        assertTrue(progress.fraction in 0f..1f)
    }

    private fun keypoint(x: Float, y: Float) = KeyPoint(x, y, 1f)
}
