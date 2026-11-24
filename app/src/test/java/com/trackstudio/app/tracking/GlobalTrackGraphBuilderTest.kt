package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GlobalTrackGraphBuilderTest {
    @Test(expected = IllegalArgumentException::class)
    fun rejectsVerifiedMatchesWithinTheSameFrame() {
        GlobalTrackGraphBuilder().addVerifiedMatches(0, 0, listOf(0 to 1))
    }

    @Test
    fun consolidatesSameFrameObservationsAtTenPixelBoundary() {
        val builder = joinedRepeatedFrameGraph(
            GlobalTrackGraphBuilder(maximumObservations = 3)
        )
        val tracks = builder.buildFromLocations(
            featureLocationsByFrame = mapOf(
                0 to listOf(observation(0, 0.0, 0.0), observation(0, 10.0, 0.0)),
                1 to listOf(observation(1, 1.0, 0.0)),
                2 to listOf(observation(2, 2.0, 0.0))
            ),
            activeFrames = setOf(0, 1, 2)
        )

        assertEquals(1, tracks.size)
        assertEquals(3, tracks.single().observations.size)
        assertEquals(listOf(0, 1, 2), tracks.single().observations.map { it.frameIndex })
        assertEquals(0.0, tracks.single().observations.first().x, 0.0)
    }

    @Test
    fun preservesTrackBySelectingOneObservationWhenSameFrameConflictExceedsTenPixels() {
        val builder = joinedRepeatedFrameGraph()
        val tracks = builder.buildFromLocations(
            featureLocationsByFrame = mapOf(
                0 to listOf(observation(0, 0.0, 0.0), observation(0, 10.0001, 0.0)),
                1 to listOf(observation(1, 1.0, 0.0)),
                2 to listOf(observation(2, 2.0, 0.0))
            ),
            activeFrames = setOf(0, 1, 2)
        )

        assertEquals(1, tracks.size)
        assertEquals(listOf(0, 1, 2), tracks.single().observations.map { it.frameIndex })
        assertEquals(0.0, tracks.single().observations.first().x, 0.0)
    }

    @Test
    fun sameFrameConflictKeepsLargestConsistentClusterAndDropsIsolatedOutlier() {
        val builder = GlobalTrackGraphBuilder()
        builder.addVerifiedMatches(0, 1, listOf(0 to 0))
        builder.addVerifiedMatches(1, 2, listOf(0 to 0))
        builder.addVerifiedMatches(0, 2, listOf(1 to 0, 2 to 0))

        val tracks = builder.buildFromLocations(
            featureLocationsByFrame = mapOf(
                0 to listOf(
                    observation(0, 0.0, 0.0),
                    observation(0, 1.0, 0.0),
                    observation(0, 50.0, 0.0)
                ),
                1 to listOf(observation(1, 1.0, 0.0)),
                2 to listOf(observation(2, 2.0, 0.0))
            ),
            activeFrames = setOf(0, 1, 2)
        )

        assertEquals(1, tracks.size)
        assertEquals(listOf(0, 1, 2), tracks.single().observations.map { it.frameIndex })
        assertTrue(tracks.single().observations.none { it.frameIndex == 0 && it.x == 50.0 })
    }

    @Test
    fun keepsOnlyTracksWithThreeThroughOneHundredObservationsAndSortsDeterministically() {
        val builder = GlobalTrackGraphBuilder()
        addChain(builder, 0, 2)
        addChain(builder, 2, 3)
        addChain(builder, 5, 101)
        addChain(builder, 200, 1100)
        addChain(builder, 1300, 100)
        addChain(builder, 106, 4)
        addChain(builder, 110, 3)
        addChain(builder, 113, 3)

        val frameLocations = (0 until 1400).associateWith { frame ->
            listOf(observation(frame, frame.toDouble(), 0.0))
        }
        val tracks = builder.buildFromLocations(
            featureLocationsByFrame = frameLocations,
            activeFrames = frameLocations.keys
        )

        assertEquals(listOf(100, 4, 3, 3, 3), tracks.map { it.observations.size })
        assertEquals(listOf(0, 1, 2, 3, 4), tracks.map { it.id })
        assertEquals(listOf(1300, 106, 113, 110, 2), tracks.map { it.observations.first().frameIndex })
    }

    @Test
    fun excludesMovedFrameTwoNodeWhenItsCorrespondenceIsNotVerified() {
        val builder = GlobalTrackGraphBuilder(minimumObservations = 3)
        builder.addVerifiedMatches(0, 1, listOf(0 to 0, 1 to 1, 2 to 2))
        // The third feature is displaced in frame 2 and is absent from this verified edge.
        builder.addVerifiedMatches(1, 2, listOf(0 to 0, 1 to 1))

        val tracks = builder.buildFromLocations(
            featureLocationsByFrame = mapOf(
                0 to listOf(observation(0, 10.0, 20.0), observation(0, 30.0, 20.0), observation(0, 50.0, 20.0)),
                1 to listOf(observation(1, 11.0, 20.0), observation(1, 31.0, 20.0), observation(1, 51.0, 20.0)),
                2 to listOf(observation(2, 12.0, 20.0), observation(2, 32.0, 20.0), observation(2, 57.0, 26.0))
            ),
            activeFrames = setOf(0, 1, 2)
        )

        assertEquals(2, tracks.size)
        assertTrue(tracks.all { track -> track.observations.map { it.frameIndex } == listOf(0, 1, 2) })
        assertTrue(tracks.none { track ->
            track.observations.any { it.frameIndex == 2 && it.x == 57.0 && it.y == 26.0 }
        })
    }

    private fun joinedRepeatedFrameGraph(
        builder: GlobalTrackGraphBuilder = GlobalTrackGraphBuilder()
    ) = builder.apply {
        addVerifiedMatches(0, 1, listOf(0 to 0))
        addVerifiedMatches(1, 2, listOf(0 to 0))
        addVerifiedMatches(0, 2, listOf(1 to 0))
    }

    private fun addChain(builder: GlobalTrackGraphBuilder, firstFrame: Int, length: Int) {
        for (frame in firstFrame until firstFrame + length - 1) {
            builder.addVerifiedMatches(frame, frame + 1, listOf(0 to 0))
        }
    }

    private fun observation(frame: Int, x: Double, y: Double) =
        CameraFeatureObservation(frame, x, y)
}
