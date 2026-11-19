package com.trackstudio.app.tracking

import org.opencv.core.KeyPoint

/**
 * Builds persistent 2D feature tracks from geometrically verified matches
 * between consecutive frames. It stores only observations, not image data.
 *
 * A track is continued only when its feature survives the current pair's
 * geometric verification. A failed pair breaks continuity so unrelated
 * detections are never joined into one track.
 */
internal class FeatureTrackBuilder {
    private var nextTrackId = 0
    private var previousAssignments = mutableMapOf<Int, Int>()
    private val assignmentsByFrame = mutableMapOf<Int, MutableMap<Int, Int>>()
    private val tracks = linkedMapOf<Int, MutableList<CameraFeatureObservation>>()

    /** Seeds persistent IDs from the selected, geometrically verified initialization pair. */
    fun seedPair(
        firstFrame: Int,
        secondFrame: Int,
        firstKeypoints: Array<KeyPoint>,
        secondKeypoints: Array<KeyPoint>,
        inlierMatches: List<Pair<Int, Int>>
    ) {
        require(firstFrame >= 0 && secondFrame > firstFrame)
        val firstAssignments = mutableMapOf<Int, Int>()
        val assignments = mutableMapOf<Int, Int>()
        for ((firstIndex, secondIndex) in inlierMatches) {
            if (firstIndex !in firstKeypoints.indices ||
                firstIndex in firstAssignments ||
                secondIndex !in secondKeypoints.indices ||
                secondIndex in assignments
            ) continue
            val id = nextTrackId++
            tracks[id] = mutableListOf(
                CameraFeatureObservation(
                    firstFrame,
                    firstKeypoints[firstIndex].pt.x,
                    firstKeypoints[firstIndex].pt.y,
                    firstIndex
                ),
                CameraFeatureObservation(
                    secondFrame,
                    secondKeypoints[secondIndex].pt.x,
                    secondKeypoints[secondIndex].pt.y,
                    secondIndex
                )
            )
            firstAssignments[firstIndex] = id
            assignments[secondIndex] = id
        }
        assignmentsByFrame[firstFrame] = firstAssignments
        assignmentsByFrame[secondFrame] = assignments
        previousAssignments = assignments
    }

    /** Restores feature-to-track IDs at a known frame before a reverse pass. */
    fun resetToFrame(frameIndex: Int) {
        previousAssignments = assignmentsByFrame[frameIndex]?.toMutableMap() ?: mutableMapOf()
    }

    /**
     * Extends tracks backwards. Match pairs are (later-frame index, earlier-frame index).
     * Observations are inserted in frame order, including inside tracks seeded by
     * the initialization pair.
     */
    fun addInlierMatchesBackward(
        frameIndex: Int,
        laterKeypoints: Array<KeyPoint>,
        earlierKeypoints: Array<KeyPoint>,
        matches: List<Pair<Int, Int>>
    ) {
        val earlierAssignments = mutableMapOf<Int, Int>()
        val usedLaterIndices = HashSet<Int>()
        for ((laterIndex, earlierIndex) in matches) {
            if (laterIndex !in laterKeypoints.indices ||
                laterIndex in usedLaterIndices ||
                earlierIndex !in earlierKeypoints.indices ||
                earlierIndex in earlierAssignments
            ) continue

            val trackId = previousAssignments[laterIndex] ?: run {
                val newId = nextTrackId++
                tracks[newId] = mutableListOf(
                    CameraFeatureObservation(
                        frameIndex,
                        earlierKeypoints[earlierIndex].pt.x,
                        earlierKeypoints[earlierIndex].pt.y,
                        earlierIndex
                    ),
                    CameraFeatureObservation(
                        frameIndex + 1,
                        laterKeypoints[laterIndex].pt.x,
                        laterKeypoints[laterIndex].pt.y,
                        laterIndex
                    )
                )
                newId
            }

            val observations = tracks[trackId] ?: continue
            val hasLaterObservation = observations.any { it.frameIndex == frameIndex + 1 }
            if (!hasLaterObservation) continue
            if (observations.none { it.frameIndex == frameIndex }) {
                val observation = CameraFeatureObservation(
                    frameIndex,
                    earlierKeypoints[earlierIndex].pt.x,
                    earlierKeypoints[earlierIndex].pt.y,
                    earlierIndex
                )
                val insertAt = observations.indexOfFirst { it.frameIndex > frameIndex }
                if (insertAt < 0) observations.add(observation)
                else observations.add(insertAt, observation)
            }
            earlierAssignments[earlierIndex] = trackId
            usedLaterIndices += laterIndex
        }
        assignmentsByFrame[frameIndex] = earlierAssignments
        previousAssignments = earlierAssignments
    }

    fun addInlierMatches(
        frameIndex: Int,
        previousKeypoints: Array<KeyPoint>,
        currentKeypoints: Array<KeyPoint>,
        matches: List<Pair<Int, Int>>
    ) {
        val previousFrameAssignments = previousAssignments.toMutableMap()
        val currentAssignments = mutableMapOf<Int, Int>()
        val usedPreviousIndices = HashSet<Int>()

        for ((previousIndex, currentIndex) in matches) {
            if (previousIndex !in previousKeypoints.indices ||
                previousIndex in usedPreviousIndices ||
                currentIndex !in currentKeypoints.indices ||
                currentIndex in currentAssignments
            ) continue

            val trackId = previousAssignments[previousIndex] ?: run {
                val newId = nextTrackId++
                tracks[newId] = mutableListOf(
                    CameraFeatureObservation(
                        frameIndex = frameIndex - 1,
                        x = previousKeypoints[previousIndex].pt.x,
                        y = previousKeypoints[previousIndex].pt.y,
                        featureIndex = previousIndex
                    )
                )
                previousFrameAssignments[previousIndex] = newId
                newId
            }

            val observations = tracks[trackId] ?: continue
            val lastFrame = observations.lastOrNull()?.frameIndex
            if (lastFrame != frameIndex - 1) continue

            observations += CameraFeatureObservation(
                frameIndex = frameIndex,
                x = currentKeypoints[currentIndex].pt.x,
                y = currentKeypoints[currentIndex].pt.y,
                featureIndex = currentIndex
            )
            currentAssignments[currentIndex] = trackId
            usedPreviousIndices += previousIndex
        }

        assignmentsByFrame[frameIndex - 1] = previousFrameAssignments
        assignmentsByFrame[frameIndex] = currentAssignments
        previousAssignments = currentAssignments
    }


    /**
     * Reconnects current observations directly to a known keyframe.
     * This is a recovery path when adjacent-frame matching loses continuity.
     * Only IDs already established at the source frame are reused.
     */
    fun addInlierMatchesFromFrame(
        sourceFrame: Int,
        frameIndex: Int,
        sourceKeypoints: Array<KeyPoint>,
        currentKeypoints: Array<KeyPoint>,
        matches: List<Pair<Int, Int>>
    ) {
        require(sourceFrame >= 0 && frameIndex >= 0 && sourceFrame != frameIndex)
        val sourceAssignments = assignmentsByFrame[sourceFrame] ?: return
        val currentAssignments = assignmentsByFrame[frameIndex]?.toMutableMap() ?: mutableMapOf()
        val usedTrackIds = currentAssignments.values.toMutableSet()

        for ((sourceIndex, currentIndex) in matches) {
            if (sourceIndex !in sourceKeypoints.indices ||
                currentIndex !in currentKeypoints.indices
            ) continue
            val trackId = sourceAssignments[sourceIndex] ?: continue
            val displacedTrackId = currentAssignments[currentIndex]
            if (displacedTrackId == trackId) continue
            val observations = tracks[trackId] ?: continue
            if (observations.any { it.frameIndex == frameIndex }) continue
            if (!usedTrackIds.add(trackId)) continue

            if (displacedTrackId != null) {
                currentAssignments.remove(currentIndex)
                if (currentAssignments.values.none { it == displacedTrackId }) {
                    tracks[displacedTrackId]?.removeAll { it.frameIndex == frameIndex }
                    usedTrackIds.remove(displacedTrackId)
                }
            }
            val observation = CameraFeatureObservation(
                frameIndex,
                currentKeypoints[currentIndex].pt.x,
                currentKeypoints[currentIndex].pt.y,
                currentIndex
            )
            val insertAt = observations.indexOfFirst { it.frameIndex > frameIndex }
            if (insertAt < 0) observations.add(observation)
            else observations.add(insertAt, observation)
            currentAssignments[currentIndex] = trackId
        }

        assignmentsByFrame[frameIndex] = currentAssignments
        previousAssignments = currentAssignments
    }

    fun breakContinuity() {
        // Do not clear the map stored in assignmentsByFrame. A recovery pass
        // may still need those keyframe-to-track IDs after continuity is lost.
        previousAssignments = mutableMapOf()
    }

    /** Replaces triangulated tracks after local COLMAP merge/completion updates. */
    fun replaceMapTracks(
        replacedTrackIds: Set<Int>,
        updatedTracks: List<CameraFeatureTrack>,
        removedTrackIds: Set<Int>,
        activeFrameIndex: Int
    ) {
        require(removedTrackIds.all(replacedTrackIds::contains))
        require(updatedTracks.map { it.id }.distinct().size == updatedTracks.size)
        require(updatedTracks.all { it.id in replacedTrackIds })

        val ownerByFeature = HashMap<Long, Int>()
        val updatedObservationsByTrack = linkedMapOf<Int, MutableList<CameraFeatureObservation>>()
        for (track in updatedTracks) {
            if (track.id in removedTrackIds) continue
            require(track.observations.map { it.frameIndex }.distinct().size == track.observations.size) {
                "A map track cannot contain multiple observations from one frame."
            }
            val seenFeatures = HashSet<Long>()
            for (observation in track.observations) {
                if (observation.frameIndex < 0 || observation.featureIndex < 0) continue
                val key = featureNode(observation.frameIndex, observation.featureIndex)
                require(seenFeatures.add(key)) { "A map track cannot repeat a feature observation." }
                val previousOwner = ownerByFeature.putIfAbsent(key, track.id)
                require(previousOwner == null || previousOwner == track.id) {
                    "A feature observation cannot belong to multiple map tracks."
                }
            }
            updatedObservationsByTrack[track.id] = track.observations.toMutableList()
        }

        val retainedTracks = linkedMapOf<Int, MutableList<CameraFeatureObservation>>()
        for ((trackId, observations) in tracks) {
            if (trackId in replacedTrackIds) continue
            val seenFeatures = HashSet<Long>()
            val retainedObservations = observations.filter { observation ->
                if (observation.frameIndex < 0 || observation.featureIndex < 0) {
                    true
                } else {
                    val key = featureNode(observation.frameIndex, observation.featureIndex)
                    seenFeatures.add(key) && ownerByFeature.putIfAbsent(key, trackId) == null
                }
            }
            if (retainedObservations.isNotEmpty()) {
                retainedTracks[trackId] = retainedObservations.toMutableList()
            }
        }
        tracks.clear()
        tracks.putAll(retainedTracks)
        tracks.putAll(updatedObservationsByTrack)

        assignmentsByFrame.clear()
        for ((trackId, observations) in tracks) {
            for (observation in observations) {
                if (observation.frameIndex < 0 || observation.featureIndex < 0) continue
                assignmentsByFrame.getOrPut(observation.frameIndex) { mutableMapOf() }
                    .putIfAbsent(observation.featureIndex, trackId)
            }
        }
        previousAssignments = assignmentsByFrame[activeFrameIndex]?.toMutableMap() ?: mutableMapOf()
    }

    private fun featureNode(frameIndex: Int, featureIndex: Int): Long =
        (frameIndex.toLong() shl 32) or (featureIndex.toLong() and 0xffffffffL)

    /**
     * Creates an immutable snapshot. When [trackIds] is supplied, only those
     * tracks are copied. PnP registration uses this map-sized view rather than
     * allocating snapshots for every short-lived feature in the clip.
     */
    fun build(
        minimumObservations: Int = 2,
        trackIds: Set<Int>? = null
    ): List<CameraFeatureTrack> {
        val source: Sequence<Pair<Int, MutableList<CameraFeatureObservation>>> =
            if (trackIds == null) tracks.asSequence().map { entry -> entry.key to entry.value }
            else trackIds.asSequence().mapNotNull { id ->
                tracks[id]?.let { observations -> id to observations }
            }
        return source.mapNotNull { (id, observations) ->
            if (observations.size < minimumObservations) null
            else CameraFeatureTrack(id, observations.toList())
        }.toList()
    }
}

data class CameraFeatureObservation(
    val frameIndex: Int,
    val x: Double,
    val y: Double,
    val featureIndex: Int = -1
)

data class CameraFeatureTrack(
    val id: Int,
    val observations: List<CameraFeatureObservation>
) {
    val length: Int
        get() = observations.size
}
