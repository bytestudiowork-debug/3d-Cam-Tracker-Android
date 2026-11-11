package com.trackstudio.ai

/**
 * Platform-neutral data contracts for the independent tracking module.
 * Image decoding and reconstruction remain outside this module.
 */
data class TrackingFrame(
    val frameIndex: Int,
    val timestampNs: Long,
    val width: Int,
    val height: Int,
    val pixels: ByteArray,
    val channels: Int
) {
    init {
        require(frameIndex >= 0) { "Frame index must not be negative." }
        require(timestampNs >= 0L) { "Timestamp must not be negative." }
        require(width > 0 && height > 0) { "Frame dimensions must be positive." }
        require(channels in 1..4) { "Channel count must be between one and four." }
        require(pixels.size == width * height * channels) {
            "Pixel buffer size does not match frame dimensions."
        }
    }
}

data class FeatureObservation(
    val frameIndex: Int,
    val x: Float,
    val y: Float,
    val confidence: Float,
    val trackId: Long? = null,
    val descriptor: FloatArray? = null
)

data class FeatureMatch(
    val firstFeatureIndex: Int,
    val secondFeatureIndex: Int,
    val confidence: Float
)

data class FrameSignature(
    val frameIndex: Int,
    val timestampNs: Long,
    val values: FloatArray
)

interface FeatureDetector {
    fun detect(frame: TrackingFrame): List<FeatureObservation>
}

interface FeatureMatcher {
    fun match(
        first: List<FeatureObservation>,
        second: List<FeatureObservation>
    ): List<FeatureMatch>
}

interface PatchTracker {
    fun track(
        previous: TrackingFrame,
        current: TrackingFrame,
        previousObservations: List<FeatureObservation>
    ): List<FeatureObservation>
}

interface LoopClosureDetector {
    fun findCandidates(
        query: FrameSignature,
        history: List<FrameSignature>
    ): List<Int>
}

/**
 * Removes invalid and duplicate observations before data leaves this module.
 */
object ObservationFilter {
    fun filter(
        observations: List<FeatureObservation>,
        frameWidth: Int,
        frameHeight: Int,
        maximumCount: Int = 8_000
    ): List<FeatureObservation> {
        require(frameWidth > 0 && frameHeight > 0)
        require(maximumCount > 0)

        val unique = LinkedHashMap<Pair<Int, Int>, FeatureObservation>()
        observations.asSequence()
            .filter { observation ->
                observation.frameIndex >= 0 &&
                    observation.x.isFinite() &&
                    observation.y.isFinite() &&
                    observation.confidence.isFinite() &&
                    observation.x >= 0f &&
                    observation.y >= 0f &&
                    observation.x < frameWidth &&
                    observation.y < frameHeight &&
                    observation.confidence in 0f..1f
            }
            .sortedByDescending { it.confidence }
            .forEach { observation ->
                val key = (observation.x * 4f).toInt() to (observation.y * 4f).toInt()
                if (unique.size < maximumCount) unique.putIfAbsent(key, observation)
            }

        return unique.values.toList()
    }
}
