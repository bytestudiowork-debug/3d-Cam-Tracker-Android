package com.trackstudio.ai

import kotlin.math.sqrt

/**
 * Matches normalized learned descriptors with mutual-nearest and ratio checks.
 * Inputs are capped before pairwise comparison to keep work bounded on mobile CPUs.
 */
class DescriptorFeatureMatcher(
    private val minimumSimilarity: Float = 0.55f,
    private val maximumDistanceRatio: Float = 0.8f,
    private val maximumFeaturesPerFrame: Int = 500
) : FeatureMatcher {

    init {
        require(minimumSimilarity in -1f..1f)
        require(maximumDistanceRatio in 0f..1f)
        require(maximumFeaturesPerFrame > 0)
    }

    override fun match(
        first: List<FeatureObservation>,
        second: List<FeatureObservation>
    ): List<FeatureMatch> {
        val left = first.withIndex()
            .filter { it.value.descriptor?.isNotEmpty() == true }
            .sortedByDescending { it.value.confidence }
            .take(maximumFeaturesPerFrame)
        val right = second.withIndex()
            .filter { it.value.descriptor?.isNotEmpty() == true }
            .sortedByDescending { it.value.confidence }
            .take(maximumFeaturesPerFrame)
        if (left.isEmpty() || right.isEmpty()) return emptyList()

        val dimensions = left.first().value.descriptor!!.size
        if (dimensions == 0 ||
            left.any { it.value.descriptor!!.size != dimensions } ||
            right.any { it.value.descriptor!!.size != dimensions }) return emptyList()

        val leftBest = IntArray(left.size) { -1 }
        val leftBestSimilarity = FloatArray(left.size) { Float.NEGATIVE_INFINITY }
        val leftSecondSimilarity = FloatArray(left.size) { Float.NEGATIVE_INFINITY }
        val rightBest = IntArray(right.size) { -1 }
        val rightBestSimilarity = FloatArray(right.size) { Float.NEGATIVE_INFINITY }

        for (i in left.indices) {
            val a = left[i].value.descriptor!!
            for (j in right.indices) {
                val b = right[j].value.descriptor!!
                val similarity = cosineSimilarity(a, b)
                if (!similarity.isFinite()) continue

                if (similarity > leftBestSimilarity[i]) {
                    leftSecondSimilarity[i] = leftBestSimilarity[i]
                    leftBestSimilarity[i] = similarity
                    leftBest[i] = j
                } else if (similarity > leftSecondSimilarity[i]) {
                    leftSecondSimilarity[i] = similarity
                }

                if (similarity > rightBestSimilarity[j]) {
                    rightBestSimilarity[j] = similarity
                    rightBest[j] = i
                }
            }
        }

        val matches = ArrayList<FeatureMatch>()
        for (i in left.indices) {
            val j = leftBest[i]
            if (j < 0 || rightBest[j] != i) continue
            val bestSimilarity = leftBestSimilarity[i]
            if (bestSimilarity < minimumSimilarity) continue

            val bestDistance = sqrt((2f - 2f * bestSimilarity).coerceAtLeast(0f))
            val secondSimilarity = leftSecondSimilarity[i]
            val secondDistance = if (secondSimilarity.isFinite()) {
                sqrt((2f - 2f * secondSimilarity).coerceAtLeast(0f))
            } else {
                2f
            }
            if (secondDistance > 1e-6f &&
                bestDistance / secondDistance > maximumDistanceRatio) continue

            matches += FeatureMatch(
                firstFeatureIndex = left[i].index,
                secondFeatureIndex = right[j].index,
                confidence = bestSimilarity.coerceIn(0f, 1f)
            )
        }
        return matches.sortedByDescending { it.confidence }
    }

    private fun cosineSimilarity(first: FloatArray, second: FloatArray): Float {
        var dot = 0.0
        var firstNorm = 0.0
        var secondNorm = 0.0
        for (index in first.indices) {
            val a = first[index]
            val b = second[index]
            if (!a.isFinite() || !b.isFinite()) return Float.NaN
            dot += a.toDouble() * b.toDouble()
            firstNorm += a.toDouble() * a.toDouble()
            secondNorm += b.toDouble() * b.toDouble()
        }
        if (firstNorm <= 1e-12 || secondNorm <= 1e-12) return Float.NaN
        return (dot / sqrt(firstNorm * secondNorm)).toFloat().coerceIn(-1f, 1f)
    }
}
