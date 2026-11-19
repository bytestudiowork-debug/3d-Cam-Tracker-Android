package com.trackstudio.app.tracking

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.features2d.BFMatcher
import org.opencv.core.KeyPoint
import kotlin.math.acos
import kotlin.math.floor
import kotlin.math.sqrt

internal class AndroidSiftFeatureExtractor(private val maxFeatures: Int) {
    init {
        require(maxFeatures > 0)
    }

    fun detectAndCompute(
        image: Mat,
        mask: Mat,
        keypoints: MatOfKeyPoint,
        descriptors: Mat
    ) {
        require(!image.empty() && image.type() == CvType.CV_8UC1)
        val pixelCount = image.rows().toLong() * image.cols().toLong()
        require(pixelCount in 1..MAX_IMAGE_PIXELS) {
            "The grayscale image exceeds the native feature-extraction limit."
        }
        val grayscale = ByteArray(pixelCount.toInt())
        require(image.get(0, 0, grayscale) == grayscale.size) {
            "Could not read the grayscale image pixels."
        }

        val maskPixels = if (mask.empty()) {
            null
        } else {
            require(mask.type() == CvType.CV_8UC1 &&
                mask.rows() == image.rows() && mask.cols() == image.cols()
            ) { "The feature mask must be an 8-bit single-channel image of equal dimensions." }
            ByteArray(grayscale.size).also { pixels ->
                require(mask.get(0, 0, pixels) == pixels.size) {
                    "Could not read the feature mask pixels."
                }
            }
        }

        val output = NativeSiftExtractor.extract(
            grayscale, image.cols(), image.rows(), maxFeatures
        ) ?: throw IllegalStateException("The native feature extractor could not be loaded or run.")
        require(output.keypointData.size % KEYPOINT_STRIDE == 0)
        val nativeFeatureCount = output.keypointData.size / KEYPOINT_STRIDE
        require(output.descriptorData.size.toLong() ==
            nativeFeatureCount.toLong() * DESCRIPTOR_LENGTH
        ) { "The native feature extractor returned inconsistent arrays." }

        val selectedKeypoints = ArrayList<KeyPoint>(nativeFeatureCount)
        val selectedDescriptors = ArrayList<ByteArray>(nativeFeatureCount)
        for (index in 0 until nativeFeatureCount) {
            val offset = index * KEYPOINT_STRIDE
            val x = output.keypointData[offset]
            val y = output.keypointData[offset + 1]
            if (maskPixels != null) {
                val pixelX = floor(x.toDouble()).toInt().coerceIn(0, image.cols() - 1)
                val pixelY = floor(y.toDouble()).toInt().coerceIn(0, image.rows() - 1)
                if ((maskPixels[pixelY * image.cols() + pixelX].toInt() and 0xff) == 0) {
                    continue
                }
            }
            val scale = output.keypointData[offset + 2]
            val angleDegrees = Math.toDegrees(
                output.keypointData[offset + 3].toDouble()
            ).toFloat()
            selectedKeypoints += KeyPoint(x, y, 2.0f * scale, angleDegrees, 0.0f, 0, -1)
            selectedDescriptors += output.descriptorData.copyOfRange(
                index * DESCRIPTOR_LENGTH,
                (index + 1) * DESCRIPTOR_LENGTH
            )
        }

        keypoints.release()
        keypoints.fromArray(*selectedKeypoints.toTypedArray())
        descriptors.release()
        if (selectedDescriptors.isNotEmpty()) {
            descriptors.create(selectedDescriptors.size, DESCRIPTOR_LENGTH, CvType.CV_8UC1)
            val flattened = ByteArray(selectedDescriptors.size * DESCRIPTOR_LENGTH)
            selectedDescriptors.forEachIndexed { index, row ->
                System.arraycopy(row, 0, flattened, index * DESCRIPTOR_LENGTH, DESCRIPTOR_LENGTH)
            }
            check(descriptors.put(0, 0, flattened) == flattened.size) {
                "Could not write the native descriptors into the OpenCV matrix."
            }
        }
    }

    fun clear() = NativeSiftExtractor.clear()

    private companion object {
        const val KEYPOINT_STRIDE = 4
        const val DESCRIPTOR_LENGTH = 128
        const val MAX_IMAGE_PIXELS = 32L * 1024L * 1024L
    }
}

internal object RootSiftDescriptorMath {
    const val DESCRIPTOR_NORM = 512.0
    const val DESCRIPTOR_LENGTH = 128

    fun normalizeDescriptor(values: FloatArray): FloatArray {
        val normalized = values.copyOf()
        normalizeDescriptorInPlace(normalized)
        return normalized
    }

    fun normalizeDescriptorInPlace(values: FloatArray) {
        require(values.isNotEmpty())
        require(values.all { it.isFinite() && it >= 0f })
        val l1Norm = values.sumOf { it.toDouble() }
        if (l1Norm <= 1e-20) {
            values.fill(0f)
            return
        }
        for (index in values.indices) {
            values[index] = (sqrt(values[index].toDouble() / l1Norm) * DESCRIPTOR_NORM).toFloat()
        }
    }

    fun angularDistance(first: FloatArray, second: FloatArray): Double {
        require(first.size == second.size && first.isNotEmpty())
        require(first.all { it.isFinite() && it >= 0f })
        require(second.all { it.isFinite() && it >= 0f })
        val dotProduct = first.indices.sumOf { first[it].toDouble() * second[it] }
        return angularDistanceFromDotProduct(dotProduct)
    }

    fun angularDistance(
        first: ByteArray,
        firstOffset: Int,
        second: ByteArray,
        secondOffset: Int
    ): Double {
        require(firstOffset >= 0 && firstOffset + DESCRIPTOR_LENGTH <= first.size)
        require(secondOffset >= 0 && secondOffset + DESCRIPTOR_LENGTH <= second.size)
        var dotProduct = 0.0
        for (index in 0 until DESCRIPTOR_LENGTH) {
            val firstValue = first[firstOffset + index].toInt() and 0xff
            val secondValue = second[secondOffset + index].toInt() and 0xff
            dotProduct += firstValue.toDouble() * secondValue
        }
        return angularDistanceFromDotProduct(dotProduct)
    }

    private fun angularDistanceFromDotProduct(dotProduct: Double): Double {
        // This brute-force helper uses the fixed 512-scale descriptor convention.
        val cosine = (dotProduct / (DESCRIPTOR_NORM * DESCRIPTOR_NORM)).coerceIn(-1.0, 1.0)
        return acos(cosine)
    }

    fun normalizeAndQuantizeInPlace(descriptors: Mat) {
        if (descriptors.empty()) return
        require(descriptors.type() == CvType.CV_32F)
        val row = FloatArray(descriptors.cols())
        for (index in 0 until descriptors.rows()) {
            descriptors.get(index, 0, row)
            normalizeDescriptorInPlace(row)
            descriptors.put(index, 0, row)
        }
        val quantized = Mat()
        try {
            descriptors.convertTo(quantized, CvType.CV_8U)
            quantized.copyTo(descriptors)
        } finally {
            quantized.release()
        }
    }
}

internal data class RankedDescriptorMatch(
    val queryIndex: Int,
    val trainIndex: Int,
    val normalizedL2Distance: Double
)

internal object SiftMatchPolicy {
    const val DEFAULT_RATIO = 0.8
    const val DEFAULT_MAX_DISTANCE = 0.7

    fun selectMutualMatches(
        forwardRows: List<List<RankedDescriptorMatch>>,
        reverseRows: List<List<RankedDescriptorMatch>>,
        maxRatio: Double = DEFAULT_RATIO,
        maxDistance: Double = DEFAULT_MAX_DISTANCE,
        maximumMatches: Int = Int.MAX_VALUE
    ): List<Pair<Int, Int>> {
        require(maxRatio.isFinite() && maxRatio > 0.0 && maxRatio < 1.0)
        require(maxDistance.isFinite() && maxDistance > 0.0)
        if (maximumMatches <= 0) return emptyList()

        val forward = bestPassingMatches(forwardRows, maxRatio, maxDistance)
        val reverse = bestPassingMatches(reverseRows, maxRatio, maxDistance)
        val selected = ArrayList<Pair<Int, Int>>()
        for ((queryIndex, match) in forward.toSortedMap()) {
            val reverseMatch = reverse[match.trainIndex] ?: continue
            if (reverseMatch.trainIndex != queryIndex) continue
            selected += queryIndex to match.trainIndex
            if (selected.size >= maximumMatches) break
        }
        return selected
    }

    private fun bestPassingMatches(
        rows: List<List<RankedDescriptorMatch>>,
        maxRatio: Double,
        maxDistance: Double
    ): Map<Int, RankedDescriptorMatch> {
        val accepted = HashMap<Int, RankedDescriptorMatch>()
        for (row in rows) {
            val candidates = row.asSequence()
                .filter {
                    it.queryIndex >= 0 && it.trainIndex >= 0 &&
                        it.normalizedL2Distance.isFinite() && it.normalizedL2Distance >= 0.0
                }
                .sortedWith(
                    compareBy<RankedDescriptorMatch> { it.normalizedL2Distance }
                        .thenBy { it.trainIndex }
                )
                .take(2)
                .toList()
            if (candidates.isEmpty()) continue
            val best = candidates[0]
            val secondBestDistance = candidates.getOrNull(1)?.normalizedL2Distance ?: Double.MAX_VALUE
            if (best.normalizedL2Distance > maxDistance) continue
            if (best.normalizedL2Distance >= maxRatio * secondBestDistance) continue
            val previous = accepted[best.queryIndex]
            if (previous == null || best.normalizedL2Distance < previous.normalizedL2Distance ||
                (best.normalizedL2Distance == previous.normalizedL2Distance &&
                    best.trainIndex < previous.trainIndex)
            ) {
                accepted[best.queryIndex] = best
            }
        }
        return accepted
    }
}

internal class SiftDescriptorMatcher(
    private val ratioThreshold: Float = SiftMatchPolicy.DEFAULT_RATIO.toFloat(),
    private val maximumDistance: Double = SiftMatchPolicy.DEFAULT_MAX_DISTANCE
) {
    private val matcher = BFMatcher.create(Core.NORM_L2, false)

    init {
        require(ratioThreshold.isFinite() && ratioThreshold > 0f && ratioThreshold < 1f)
        require(maximumDistance.isFinite() && maximumDistance > 0.0)
    }

    fun match(
        firstDescriptors: Mat,
        secondDescriptors: Mat,
        maximumMatches: Int = Int.MAX_VALUE
    ): List<Pair<Int, Int>> {
        if (firstDescriptors.empty() || secondDescriptors.empty() || maximumMatches <= 0) {
            return emptyList()
        }
        requireDescriptorFormat(firstDescriptors)
        requireDescriptorFormat(secondDescriptors)
        val forward = knnRows(firstDescriptors, secondDescriptors)
        val reverse = knnRows(secondDescriptors, firstDescriptors)
        return SiftMatchPolicy.selectMutualMatches(
            forwardRows = forward,
            reverseRows = reverse,
            maxRatio = ratioThreshold.toDouble(),
            maxDistance = maximumDistance,
            maximumMatches = maximumMatches
        )
    }

    fun clear() = matcher.clear()

    private fun requireDescriptorFormat(descriptors: Mat) {
        require(descriptors.type() == CvType.CV_8UC1 &&
            descriptors.cols() == RootSiftDescriptorMath.DESCRIPTOR_LENGTH
        ) { "SIFT matching requires quantized 128-dimensional single-channel descriptors." }
    }

    private fun knnRows(
        queryDescriptors: Mat,
        trainDescriptors: Mat
    ): List<List<RankedDescriptorMatch>> {
        val rows = mutableListOf<MatOfDMatch>()
        return try {
            // Keep ranking, thresholding, and ratio testing in the same L2 metric.
            matcher.knnMatch(queryDescriptors, trainDescriptors, rows, 2)
            rows.map { row ->
                row.toArray().mapNotNull { match ->
                    if (match.queryIdx !in 0 until queryDescriptors.rows() ||
                        match.trainIdx !in 0 until trainDescriptors.rows()
                    ) return@mapNotNull null
                    RankedDescriptorMatch(
                        queryIndex = match.queryIdx,
                        trainIndex = match.trainIdx,
                        normalizedL2Distance =
                            match.distance.toDouble() / RootSiftDescriptorMath.DESCRIPTOR_NORM
                    )
                }
            }
        } finally {
            rows.forEach { it.release() }
        }
    }

}
