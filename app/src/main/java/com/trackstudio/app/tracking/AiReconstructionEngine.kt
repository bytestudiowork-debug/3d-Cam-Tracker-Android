package com.trackstudio.app.tracking

import android.content.Context
import com.trackstudio.ai.OnnxFeatureDetector
import com.trackstudio.ai.OnnxFeatureMatcher
import com.trackstudio.ai.TrackingFrame
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.KeyPoint
import org.opencv.imgproc.Imgproc

/**
 * Supplies learned observations and pair matches to the independent geometry stages.
 * The model sessions are scoped to one solve and are always released afterwards.
 */
internal class AiReconstructionEngine(
    context: Context,
    private val extractorAssetPath: String = "ai/extractor.onnx",
    private val matcherAssetPath: String = "ai/matcher.onnx"
) : ReconstructionEngine {
    private val appContext = context.applicationContext

    override suspend fun solve(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit
    ): CameraTrackingResult {
        val detector = OnnxFeatureDetector(
            context = appContext,
            modelAssetPath = extractorAssetPath,
            maximumFeatures = MAXIMUM_FEATURES,
            threadCount = 2
        )
        val matcher = try {
            OnnxFeatureMatcher(
                context = appContext,
                modelAssetPath = matcherAssetPath,
                maximumFeatures = MAXIMUM_FEATURES,
                minimumMatchConfidence = MINIMUM_MATCH_CONFIDENCE,
                threadCount = 2
            )
        } catch (failure: Throwable) {
            detector.close()
            throw failure
        }

        return try {
            val adapter = AiFeatureFrameAdapter(detector)
            GlobalReconstructionEngine(
                frameFeatureExtractor = adapter::extract,
                descriptorNormType = Core.NORM_L2,
                frameFeatureMatcher = { first, second ->
                    adapter.match(first, second, matcher)
                }
            ).solve(frameCount, cameraMatrix, frameProvider, onProgress)
        } finally {
            matcher.close()
            detector.close()
        }
    }

    private companion object {
        const val MAXIMUM_FEATURES = 1400
        const val MINIMUM_MATCH_CONFIDENCE = 0.1f
    }
}

/** Converts model observations into the frame representation consumed by SfM. */
private class AiFeatureFrameAdapter(
    private val detector: OnnxFeatureDetector
) {
    fun extract(
        source: Mat,
        frameIndex: Int,
        maximumFeatures: Int
    ): GlobalReconstructionEngine.FrameFeatures {
        require(!source.empty()) { "Cannot extract features from an empty frame." }
        require(source.depth() == CvType.CV_8U) { "AI feature extraction requires 8-bit image frames." }

        val rgb = Mat()
        try {
            when (source.channels()) {
                1 -> Imgproc.cvtColor(source, rgb, Imgproc.COLOR_GRAY2RGB)
                3 -> Imgproc.cvtColor(source, rgb, Imgproc.COLOR_BGR2RGB)
                4 -> Imgproc.cvtColor(source, rgb, Imgproc.COLOR_BGRA2RGB)
                else -> throw IllegalArgumentException(
                    "Unsupported frame channel count: ${source.channels()}"
                )
            }

            val width = rgb.cols()
            val height = rgb.rows()
            val pixels = ByteArray(width * height * 3)
            check(rgb.get(0, 0, pixels) == pixels.size) {
                "Unable to read the RGB frame pixels."
            }
            val observations = detector.detect(
                TrackingFrame(
                    frameIndex = frameIndex,
                    timestampNs = 0L,
                    width = width,
                    height = height,
                    pixels = pixels,
                    channels = 3
                )
            ).take(maximumFeatures.coerceAtLeast(0))

            if (observations.isEmpty()) {
                return GlobalReconstructionEngine.FrameFeatures(
                    emptyArray(), Mat(), source.cols(), source.rows()
                )
            }

            val descriptorSize = observations.first().descriptor?.size
                ?: throw IllegalStateException("The feature detector returned an observation without a descriptor.")
            require(observations.all { it.descriptor?.size == descriptorSize }) {
                "The feature detector returned inconsistent descriptor dimensions."
            }
            val descriptorValues = FloatArray(observations.size * descriptorSize)
            observations.forEachIndexed { index, observation ->
                System.arraycopy(
                    observation.descriptor!!,
                    0,
                    descriptorValues,
                    index * descriptorSize,
                    descriptorSize
                )
            }
            val descriptors = Mat(observations.size, descriptorSize, CvType.CV_32F)
            try {
                check(descriptors.put(0, 0, descriptorValues) == descriptorValues.size) {
                    "Unable to pack learned feature descriptors."
                }
                val keypoints = Array(observations.size) { index ->
                    val observation = observations[index]
                    KeyPoint(observation.x, observation.y, 1f)
                }
                return GlobalReconstructionEngine.FrameFeatures(
                    keypoints = keypoints,
                    descriptors = descriptors,
                    width = source.cols(),
                    height = source.rows()
                )
            } catch (failure: Throwable) {
                descriptors.release()
                throw failure
            }
        } finally {
            rgb.release()
        }
    }

    fun match(
        first: GlobalReconstructionEngine.FrameFeatures,
        second: GlobalReconstructionEngine.FrameFeatures,
        matcher: OnnxFeatureMatcher
    ): List<Pair<Int, Int>> {
        if (first.keypoints.isEmpty() || second.keypoints.isEmpty() ||
            first.descriptors.empty() || second.descriptors.empty()
        ) return emptyList()

        val descriptorSize = first.descriptors.cols()
        require(descriptorSize == 64 && second.descriptors.cols() == descriptorSize)
        val firstDescriptors = FloatArray(first.keypoints.size * descriptorSize)
        val secondDescriptors = FloatArray(second.keypoints.size * descriptorSize)
        check(first.descriptors.get(0, 0, firstDescriptors) == firstDescriptors.size)
        check(second.descriptors.get(0, 0, secondDescriptors) == secondDescriptors.size)

        val firstCoordinates = FloatArray(first.keypoints.size * 2)
        val secondCoordinates = FloatArray(second.keypoints.size * 2)
        first.keypoints.forEachIndexed { index, keypoint ->
            firstCoordinates[index * 2] = keypoint.pt.x
            firstCoordinates[index * 2 + 1] = keypoint.pt.y
        }
        second.keypoints.forEachIndexed { index, keypoint ->
            secondCoordinates[index * 2] = keypoint.pt.x
            secondCoordinates[index * 2 + 1] = keypoint.pt.y
        }

        return matcher.matchRaw(
            firstKeypoints = firstCoordinates,
            firstDescriptors = firstDescriptors,
            firstCount = first.keypoints.size,
            firstWidth = first.width,
            firstHeight = first.height,
            secondKeypoints = secondCoordinates,
            secondDescriptors = secondDescriptors,
            secondCount = second.keypoints.size,
            secondWidth = second.width,
            secondHeight = second.height
        ).map { it.firstFeatureIndex to it.secondFeatureIndex }
    }
}
