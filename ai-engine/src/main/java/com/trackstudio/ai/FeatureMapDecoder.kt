package com.trackstudio.ai

import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Decodes the dense output tensors into sparse observations.
 * Tensor layouts are NCHW and the detector stride is eight.
 */
object FeatureMapDecoder {
    fun decode(
        frameIndex: Int,
        inputWidth: Int,
        inputHeight: Int,
        descriptorMap: FloatArray,
        keypointLogits: FloatArray,
        reliabilityMap: FloatArray,
        maximumFeatures: Int,
        descriptorChannels: Int = 64,
        nmsRadius: Int = 2,
        confidenceThreshold: Float = 0.05f
    ): List<FeatureObservation> {
        require(frameIndex >= 0)
        require(inputWidth >= 8 && inputHeight >= 8)
        require(inputWidth % 8 == 0 && inputHeight % 8 == 0)
        require(maximumFeatures > 0)
        require(descriptorChannels > 0)
        require(nmsRadius >= 0)
        require(confidenceThreshold in 0f..1f)

        val cellWidth = inputWidth / 8
        val cellHeight = inputHeight / 8
        val cellCount = cellWidth * cellHeight
        require(descriptorMap.size == descriptorChannels * cellCount)
        require(keypointLogits.size == KEYPOINT_CHANNELS * cellCount)
        require(reliabilityMap.size == cellCount)

        val keypointMap = FloatArray(inputWidth * inputHeight)
        val probabilities = FloatArray(KEYPOINT_CHANNELS)
        for (cellY in 0 until cellHeight) {
            for (cellX in 0 until cellWidth) {
                val cell = cellY * cellWidth + cellX
                var maximum = Float.NEGATIVE_INFINITY
                for (channel in 0 until KEYPOINT_CHANNELS) {
                    val value = keypointLogits[channel * cellCount + cell]
                    if (value > maximum) maximum = value
                }
                var denominator = 0.0
                for (channel in 0 until KEYPOINT_CHANNELS) {
                    val value = exp((keypointLogits[channel * cellCount + cell] - maximum).toDouble())
                    probabilities[channel] = value.toFloat()
                    denominator += value
                }
                if (denominator <= 0.0 || !denominator.isFinite()) continue
                for (offset in 0 until 64) {
                    val x = cellX * 8 + offset % 8
                    val y = cellY * 8 + offset / 8
                    keypointMap[y * inputWidth + x] =
                        (probabilities[offset].toDouble() / denominator).toFloat()
                }
            }
        }

        val horizontalMaximum = FloatArray(keypointMap.size)
        for (y in 0 until inputHeight) {
            val row = y * inputWidth
            for (x in 0 until inputWidth) {
                var maximum = Float.NEGATIVE_INFINITY
                for (offset in -nmsRadius..nmsRadius) {
                    val sampleX = x + offset
                    if (sampleX in 0 until inputWidth) {
                        maximum = maxOf(maximum, keypointMap[row + sampleX])
                    }
                }
                horizontalMaximum[row + x] = maximum
            }
        }
        val localMaximum = FloatArray(keypointMap.size)
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                var maximum = Float.NEGATIVE_INFINITY
                for (offset in -nmsRadius..nmsRadius) {
                    val sampleY = y + offset
                    if (sampleY in 0 until inputHeight) {
                        maximum = maxOf(maximum, horizontalMaximum[sampleY * inputWidth + x])
                    }
                }
                localMaximum[y * inputWidth + x] = maximum
            }
        }

        val normalizedDescriptors = descriptorMap.copyOf()
        for (cell in 0 until cellCount) {
            var squaredNorm = 0.0
            for (channel in 0 until descriptorChannels) {
                val value = normalizedDescriptors[channel * cellCount + cell]
                squaredNorm += value.toDouble() * value.toDouble()
            }
            val norm = sqrt(squaredNorm).toFloat()
            if (norm > 1e-12f && norm.isFinite()) {
                val inverse = 1f / norm
                for (channel in 0 until descriptorChannels) {
                    normalizedDescriptors[channel * cellCount + cell] *= inverse
                }
            } else {
                for (channel in 0 until descriptorChannels) {
                    normalizedDescriptors[channel * cellCount + cell] = 0f
                }
            }
        }

        val candidates = ArrayList<FeatureObservation>()
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                val pixel = y * inputWidth + x
                val keypointScore = keypointMap[pixel]
                if (keypointScore <= confidenceThreshold || keypointScore != localMaximum[pixel]) continue

                val nearestX = floor(x.toDouble() / (inputWidth - 1) * inputWidth).toInt()
                    .coerceIn(0, inputWidth - 1)
                val nearestY = floor(y.toDouble() / (inputHeight - 1) * inputHeight).toInt()
                    .coerceIn(0, inputHeight - 1)
                var score = keypointMap[nearestY * inputWidth + nearestX] *
                    sampleBilinear(
                        reliabilityMap, cellWidth, cellHeight,
                        x.toFloat() / (inputWidth - 1) * cellWidth - 0.5f,
                        y.toFloat() / (inputHeight - 1) * cellHeight - 0.5f
                    )
                if (x == 0 && y == 0) score = -1f
                if (!score.isFinite() || score <= 0f) continue

                val descriptor = FloatArray(descriptorChannels)
                val gridX = x.toFloat() / (inputWidth - 1) * cellWidth - 0.5f
                val gridY = y.toFloat() / (inputHeight - 1) * cellHeight - 0.5f
                for (channel in 0 until descriptorChannels) {
                    descriptor[channel] = sampleChannelBicubic(
                        normalizedDescriptors, channel, cellCount, cellWidth, cellHeight,
                        gridX, gridY
                    )
                }
                var normSquared = 0.0
                for (value in descriptor) normSquared += value.toDouble() * value.toDouble()
                val norm = sqrt(normSquared).toFloat()
                if (norm <= 1e-12f || !norm.isFinite()) continue
                for (index in descriptor.indices) descriptor[index] /= norm

                candidates += FeatureObservation(
                    frameIndex = frameIndex,
                    x = x.toFloat(),
                    y = y.toFloat(),
                    confidence = score.coerceIn(0f, 1f),
                    descriptor = descriptor
                )
            }
        }
        return candidates.sortedByDescending { it.confidence }.take(maximumFeatures)
    }

    private fun sampleBilinear(
        map: FloatArray,
        width: Int,
        height: Int,
        x: Float,
        y: Float
    ): Float {
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val fx = x - x0
        val fy = y - y0
        fun at(px: Int, py: Int): Float =
            if (px in 0 until width && py in 0 until height) map[py * width + px] else 0f
        val top = at(x0, y0) * (1f - fx) + at(x0 + 1, y0) * fx
        val bottom = at(x0, y0 + 1) * (1f - fx) + at(x0 + 1, y0 + 1) * fx
        return top * (1f - fy) + bottom * fy
    }

    private fun sampleChannelBicubic(
        map: FloatArray,
        channel: Int,
        cellCount: Int,
        width: Int,
        height: Int,
        x: Float,
        y: Float
    ): Float {
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        var value = 0.0
        for (sampleY in y0 - 1..y0 + 2) {
            val weightY = cubicWeight(y - sampleY)
            if (weightY == 0f) continue
            for (sampleX in x0 - 1..x0 + 2) {
                val weightX = cubicWeight(x - sampleX)
                if (weightX == 0f || sampleX !in 0 until width ||
                    sampleY !in 0 until height
                ) continue
                value += (map[channel * cellCount + sampleY * width + sampleX] *
                    weightX * weightY).toDouble()
            }
        }
        return value.toFloat()
    }

    private fun cubicWeight(distance: Float): Float {
        val x = kotlin.math.abs(distance)
        val a = -0.75f
        return when {
            x <= 1f -> (a + 2f) * x * x * x - (a + 3f) * x * x + 1f
            x < 2f -> a * x * x * x - 5f * a * x * x + 8f * a * x - 4f * a
            else -> 0f
        }
    }

    private const val KEYPOINT_CHANNELS = 65
    private const val NMS_RADIUS = 2
}
