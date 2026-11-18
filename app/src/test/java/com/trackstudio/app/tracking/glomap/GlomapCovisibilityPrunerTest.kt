package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraFeatureTrack
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GlomapCovisibilityPrunerTest {
    @Test
    fun separatesStrongClustersWhenCrossPairsMissWeakBridgeCutoff() {
        val tracks = buildList {
            addAll(cliqueTracks(listOf(0, 1, 2), count = 30, firstId = 0))
            addAll(cliqueTracks(listOf(3, 4, 5), count = 40, firstId = 100))
            addAll(cliqueTracks(listOf(1, 2, 3), count = 21, firstId = 200))
        }

        val result = GlomapCovisibilityPruner.prune(
            tracks = tracks,
            allFrameIds = (0..6).toSet(),
            activeFrameIds = (0..5).toSet()
        )

        assertEquals(2, result.clusterCount)
        assertEquals(30.0, result.strongThreshold!!, 0.0)
        assertEquals(0, result.clusterIdByFrame.getValue(0))
        assertEquals(0, result.clusterIdByFrame.getValue(2))
        assertEquals(1, result.clusterIdByFrame.getValue(3))
        assertEquals(1, result.clusterIdByFrame.getValue(5))
        assertEquals(-1, result.clusterIdByFrame.getValue(6))
        assertEquals(21L, result.eligiblePairWeights.getValue(1 to 3))
        assertEquals(
            mapOf((1 to 3) to 21L, (2 to 3) to 21L),
            result.invalidatedPairWeights
        )
    }

    @Test
    fun mergesStrongClustersWhenTwoWeakPairsBridgeThem() {
        val tracks = buildList {
            addAll(cliqueTracks(listOf(0, 1, 2), count = 30, firstId = 0))
            addAll(cliqueTracks(listOf(3, 4, 5), count = 40, firstId = 100))
            addAll(cliqueTracks(listOf(1, 2, 3), count = 23, firstId = 200))
        }

        val result = GlomapCovisibilityPruner.prune(
            tracks = tracks,
            allFrameIds = (0..5).toSet()
        )

        assertEquals(1, result.clusterCount)
        assertEquals(30.0, result.strongThreshold!!, 0.0)
        assertEquals(setOf(0), result.clusterIdByFrame.values.toSet())
        assertEquals(23L, result.eligiblePairWeights.getValue(1 to 3))
        assertEquals(23L, result.eligiblePairWeights.getValue(2 to 3))
        assertEquals(emptyMap<Pair<Int, Int>, Long>(), result.invalidatedPairWeights)
    }

    @Test
    fun strongThresholdIsStrictAndPairCountsNeedAtLeastFiveTracks() {
        val exactlyAtThreshold = cliqueTracks(listOf(0, 1, 2), count = 30, firstId = 0)
        val thresholdResult = GlomapCovisibilityPruner.prune(
            tracks = exactlyAtThreshold,
            allFrameIds = setOf(0, 1, 2)
        )
        assertEquals(30.0, thresholdResult.strongThreshold!!, 0.0)
        assertEquals(0, thresholdResult.clusterCount)
        assertEquals(setOf(-1), thresholdResult.clusterIdByFrame.values.toSet())

        val belowPairCutoff = GlomapCovisibilityPruner.prune(
            tracks = cliqueTracks(listOf(0, 1, 2), count = 4, firstId = 100),
            allFrameIds = setOf(0, 1, 2)
        )
        assertNull(belowPairCutoff.strongThreshold)
        assertEquals(0, belowPairCutoff.clusterCount)
        assertEquals(emptyMap<Pair<Int, Int>, Long>(), belowPairCutoff.eligiblePairWeights)
    }

    @Test
    fun appliesFrameObservationFilterAndIgnoresTracksWithOnlyTwoViews() {
        val tracks = buildList {
            addAll(cliqueTracks(listOf(0, 1, 2), count = 30, firstId = 0))
            repeat(30) { offset ->
                add(
                    CameraFeatureTrack(
                        id = 500 + offset,
                        observations = listOf(
                            CameraFeatureObservation(10, 0.0, 0.0),
                            CameraFeatureObservation(11, 1.0, 0.0)
                        )
                    )
                )
            }
        }

        val result = GlomapCovisibilityPruner.prune(
            tracks = tracks,
            allFrameIds = setOf(0, 1, 2, 10, 11),
            minNumObservations = 31
        )

        assertNull(result.strongThreshold)
        assertEquals(0, result.clusterCount)
        assertEquals(setOf(-1), result.clusterIdByFrame.values.toSet())
    }

    private fun cliqueTracks(
        frames: List<Int>,
        count: Int,
        firstId: Int
    ): List<CameraFeatureTrack> = List(count) { offset ->
        CameraFeatureTrack(
            id = firstId + offset,
            observations = frames.mapIndexed { index, frame ->
                CameraFeatureObservation(
                    frameIndex = frame,
                    x = index.toDouble(),
                    y = 0.0
                )
            }
        )
    }
}
