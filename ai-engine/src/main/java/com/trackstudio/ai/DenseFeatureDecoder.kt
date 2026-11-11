package com.trackstudio.ai

import kotlin.math.sqrt

/**
 * Converts dense score and descriptor tensors into sparse feature observations.
 * Scores use [cellHeight, cellWidth, 65]; the final score channel is the no-point class.
 */
internal object DenseFeatureDecoder {
    fun decode(
        frameIndex: Int,
        imageWidth: Int,
        imageHeight: Int,
        cellHeight: Int,
        cellWidth: Int,
        scores: FloatArray,
        descriptors: FloatArray,
        descriptorSize: Int = 256,
        confidenceThreshold: Float = 0.015f,
        nmsRadius: Int = 4,
        maximumFeatures: Int = 500
    ): List<FeatureObservation> {
        require(imageWidth > 0 && imageHeight > 0)
        require(cellHeight > 0 && cellWidth > 0)
        require(scores.size == cellHeight * cellWidth * 65)
        require(descriptorSize > 0)
        require(descriptors.size == cellHeight * cellWidth * descriptorSize)
        require(confidenceThreshold in 0f..1f)
        require(nmsRadius >= 0 && maximumFeatures > 0)

        val fullWidth = cellWidth * 8
        val fullHeight = cellHeight * 8
        val heatmap = FloatArray(fullWidth * fullHeight)

        for (cellY in 0 until cellHeight) {
            for (cellX in 0 until cellWidth) {
                val base = (cellY * cellWidth + cellX) * 65
                var maximum = Float.NEGATIVE_INFINITY
                for (channel in 0 until 65) maximum = maxOf(maximum, scores[base + channel])
                var denominator = 0.0
                for (channel in 0 until 65) {
                    denominator += kotlin.math.exp((scores[base + channel] - maximum).toDouble())
                }
                if (denominator <= 0.0 || !denominator.isFinite()) continue

                for (channel in 0 until 64) {
                    val probability = (
                        kotlin.math.exp((scores[base + channel] - maximum).toDouble()) /
                            denominator
                        ).toFloat()
                    val x = cellX * 8 + channel % 8
                    val y = cellY * 8 + channel / 8
                    heatmap[y * fullWidth + x] = probability
                }
            }
        }

        data class Candidate(val x: Int, val y: Int, val score: Float)
        val candidates = ArrayList<Candidate>()
        for (y in 0 until minOf(fullHeight, imageHeight)) {
            for (x in 0 until minOf(fullWidth, imageWidth)) {
                val score = heatmap[y * fullWidth + x]
                if (score >= confidenceThreshold) candidates += Candidate(x, y, score)
            }
        }
        candidates.sortByDescending { it.score }

        val suppressed = BooleanArray(fullWidth * fullHeight)
        val selected = ArrayList<Candidate>(minOf(maximumFeatures, candidates.size))
        for (candidate in candidates) {
            val index = candidate.y * fullWidth + candidate.x
            if (suppressed[index]) continue
            selected += candidate
            if (selected.size >= maximumFeatures) break
            for (y in maxOf(0, candidate.y - nmsRadius)..minOf(fullHeight - 1, candidate.y + nmsRadius)) {
                for (x in maxOf(0, candidate.x - nmsRadius)..minOf(fullWidth - 1, candidate.x + nmsRadius)) {
                    suppressed[y * fullWidth + x] = true
                }
            }
        }

        return selected.map { candidate ->
            val descriptor = sampleDescriptor(
                descriptors, cellHeight, cellWidth, descriptorSize,
                candidate.x / 8f, candidate.y / 8f
            )
            FeatureObservation(
                frameIndex = frameIndex,
                x = candidate.x.toFloat(),
                y = candidate.y.toFloat(),
                confidence = candidate.score,
                descriptor = descriptor
            )
        }
    }

    /** Decodes SuperPoint's post-softmax full-resolution score map. */
    fun decodePixelScores(
        frameIndex: Int,
        imageWidth: Int,
        imageHeight: Int,
        cellHeight: Int,
        cellWidth: Int,
        scores: FloatArray,
        descriptors: FloatArray,
        descriptorSize: Int = 256,
        confidenceThreshold: Float = 0.001f,
        nmsRadius: Int = 4,
        maximumFeatures: Int = 500,
        descriptorChannelsFirst: Boolean = true
    ): List<FeatureObservation> {
        require(imageWidth > 0 && imageHeight > 0)
        require(cellHeight > 0 && cellWidth > 0)
        require(scores.size == imageWidth * imageHeight)
        require(descriptorSize > 0)
        require(descriptors.size == cellHeight * cellWidth * descriptorSize)
        require(confidenceThreshold in 0f..1f)
        require(nmsRadius >= 0 && maximumFeatures > 0)

        data class Candidate(val x: Int, val y: Int, val score: Float)
        val candidates = ArrayList<Candidate>()
        for (y in 0 until imageHeight) {
            for (x in 0 until imageWidth) {
                val score = scores[y * imageWidth + x]
                if (score.isFinite() && score >= confidenceThreshold) {
                    candidates += Candidate(x, y, score)
                }
            }
        }
        candidates.sortByDescending { it.score }

        val suppressed = BooleanArray(imageWidth * imageHeight)
        val selected = ArrayList<Candidate>(minOf(maximumFeatures, candidates.size))
        for (candidate in candidates) {
            val index = candidate.y * imageWidth + candidate.x
            if (suppressed[index]) continue
            selected += candidate
            if (selected.size >= maximumFeatures) break
            for (y in maxOf(0, candidate.y - nmsRadius)..minOf(imageHeight - 1, candidate.y + nmsRadius)) {
                for (x in maxOf(0, candidate.x - nmsRadius)..minOf(imageWidth - 1, candidate.x + nmsRadius)) {
                    suppressed[y * imageWidth + x] = true
                }
            }
        }

        return selected.map { candidate ->
            val descriptor = sampleDescriptor(
                descriptors, cellHeight, cellWidth, descriptorSize,
                (candidate.x + 0.5f) / 8f - 0.5f,\n                (candidate.y + 0.5f) / 8f - 0.5f, descriptorChannelsFirst
            )
            FeatureObservation(
                frameIndex = frameIndex,
                x = candidate.x.toFloat(),
                y = candidate.y.toFloat(),
                confidence = candidate.score,
                descriptor = descriptor
            )
        }
    }

    private fun sampleDescriptor(
        descriptors: FloatArray,
        cellHeight: Int,
        cellWidth: Int,
        descriptorSize: Int,
        x: Float,
        y: Float,
        channelsFirst: Boolean = false
    ): FloatArray {
        val sx = x.coerceIn(0f, (cellWidth - 1).toFloat())
        val sy = y.coerceIn(0f, (cellHeight - 1).toFloat())
        val x0 = sx.toInt()
        val y0 = sy.toInt()
        val x1 = minOf(x0 + 1, cellWidth - 1)
        val y1 = minOf(y0 + 1, cellHeight - 1)
        val dx = sx - x0
        val dy = sy - y0
        val i00 = if (channelsFirst) y0 * cellWidth + x0 else (y0 * cellWidth + x0) * descriptorSize
        val i10 = if (channelsFirst) y0 * cellWidth + x1 else (y0 * cellWidth + x1) * descriptorSize
        val i01 = if (channelsFirst) y1 * cellWidth + x0 else (y1 * cellWidth + x0) * descriptorSize
        val i11 = if (channelsFirst) y1 * cellWidth + x1 else (y1 * cellWidth + x1) * descriptorSize
        val result = FloatArray(descriptorSize)
        var normSquared = 0.0

        for (i in 0 until descriptorSize) {
            val offset = if (channelsFirst) i * cellHeight * cellWidth else i
            val top = descriptors[i00 + offset] * (1f - dx) + descriptors[i10 + offset] * dx
            val bottom = descriptors[i01 + offset] * (1f - dx) + descriptors[i11 + offset] * dx
            val value = top * (1f - dy) + bottom * dy
            result[i] = value
            normSquared += value.toDouble() * value.toDouble()
        }
        val norm = sqrt(normSquared).toFloat()
        if (norm > 1e-12f && norm.isFinite()) {
            for (i in result.indices) result[i] /= norm
        }
        return result
    }
}
