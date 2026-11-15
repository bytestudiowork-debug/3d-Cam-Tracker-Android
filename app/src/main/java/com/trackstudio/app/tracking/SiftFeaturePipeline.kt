package com.trackstudio.app.tracking

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt

internal class AndroidSiftFeatureExtractor(maxFeatures: Int) {
    private val sift: SIFT

    init {
        require(maxFeatures > 0)
        sift = SIFT.create(
            maxFeatures,
            3,
            0.02,
            10.0,
            1.6,
            CvType.CV_32F
        )
    }

    fun detectAndCompute(
        image: Mat,
        mask: Mat,
        keypoints: MatOfKeyPoint,
        descriptors: Mat
    ) {
        require(!image.empty() && image.channels() == 1)
        val doubledImage = Mat()
        val doubledMask = Mat()
        val detectedKeypoints = MatOfKeyPoint()
        try {
            Imgproc.resize(
                image,
                doubledImage,
                Size(image.cols() * 2.0, image.rows() * 2.0),
                0.0,
                0.0,
                Imgproc.INTER_LINEAR
            )
            if (!mask.empty()) {
                Imgproc.resize(
                    mask,
                    doubledMask,
                    doubledImage.size(),
                    0.0,
                    0.0,
                    Imgproc.INTER_NEAREST
                )
            }
            sift.detectAndCompute(doubledImage, doubledMask, detectedKeypoints, descriptors)
            val originalScaleKeypoints = detectedKeypoints.toArray()
            for (keypoint in originalScaleKeypoints) {
                keypoint.pt.x *= 0.5
                keypoint.pt.y *= 0.5
                keypoint.size *= 0.5f
            }
            keypoints.fromArray(*originalScaleKeypoints)
            RootSiftDescriptorMath.normalizeAndQuantizeInPlace(descriptors)
        } finally {
            doubledImage.release()
            doubledMask.release()
            detectedKeypoints.release()
        }
    }

    fun clear() = sift.clear()
}

internal object RootSiftDescriptorMath {
    const val DESCRIPTOR_NORM = 512.0

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
    val distance: Double
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
        val maximumDistanceInDescriptorUnits = maxDistance * RootSiftDescriptorMath.DESCRIPTOR_NORM
        for (row in rows) {
            val candidates = row.asSequence()
                .filter { it.queryIndex >= 0 && it.trainIndex >= 0 && it.distance.isFinite() && it.distance >= 0.0 }
                .sortedBy { it.distance }
                .take(2)
                .toList()
            if (candidates.size < 2) continue
            val best = candidates[0]
            val secondBest = candidates[1]
            if (best.distance > maximumDistanceInDescriptorUnits) continue
            if (best.distance >= maxRatio * secondBest.distance) continue
            val previous = accepted[best.queryIndex]
            if (previous == null || best.distance < previous.distance) {
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

    private fun knnRows(
        queryDescriptors: Mat,
        trainDescriptors: Mat
    ): List<List<RankedDescriptorMatch>> {
        val rows = mutableListOf<MatOfDMatch>()
        return try {
            matcher.knnMatch(queryDescriptors, trainDescriptors, rows, 2)
            rows.map { row ->
                row.toArray().map { match ->
                    RankedDescriptorMatch(
                        queryIndex = match.queryIdx,
                        trainIndex = match.trainIdx,
                        distance = match.distance.toDouble()
                    )
                }
            }
        } finally {
            rows.forEach { it.release() }
        }
    }
}
