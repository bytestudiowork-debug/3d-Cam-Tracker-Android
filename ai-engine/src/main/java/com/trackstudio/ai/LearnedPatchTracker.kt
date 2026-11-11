package com.trackstudio.ai

/**
 * Carries feature identities between consecutive frames using learned descriptors.
 * Unmatched current-frame features receive new identities.
 */
class LearnedPatchTracker(
    private val detector: FeatureDetector,
    private val matcher: FeatureMatcher = DescriptorFeatureMatcher()
) : PatchTracker {

    override fun track(
        previous: TrackingFrame,
        current: TrackingFrame,
        previousObservations: List<FeatureObservation>
    ): List<FeatureObservation> {
        require(current.frameIndex > previous.frameIndex) {
            "The current frame must follow the previous frame."
        }
        if (previousObservations.isEmpty()) {
            return detector.detect(current).mapIndexed { index, observation ->
                observation.copy(
                    trackId = observation.trackId ?: createTrackId(current.frameIndex, index)
                )
            }
        }

        val currentObservations = detector.detect(current)
        if (currentObservations.isEmpty()) return emptyList()

        val matches = matcher.match(previousObservations, currentObservations)
        val assigned = HashMap<Int, Long>(matches.size)
        for (match in matches) {
            val old = previousObservations.getOrNull(match.firstFeatureIndex) ?: continue
            assigned[match.secondFeatureIndex] = old.trackId
                ?: createTrackId(previous.frameIndex, match.firstFeatureIndex)
        }

        return currentObservations.mapIndexed { index, observation ->
            observation.copy(
                trackId = assigned[index] ?: createTrackId(current.frameIndex, index)
            )
        }
    }

    private fun createTrackId(frameIndex: Int, featureIndex: Int): Long =
        (frameIndex.toLong() shl 32) or (featureIndex.toLong() and 0xffffffffL)
}
