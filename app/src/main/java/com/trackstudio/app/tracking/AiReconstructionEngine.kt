package com.trackstudio.app.tracking

import android.content.Context
import com.trackstudio.ai.OnDeviceFeatureDetector
import com.trackstudio.ai.TrackingFrame
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.KeyPoint
import org.opencv.imgproc.Imgproc

/**
 * Uses learned local features as input to the existing global geometry stages.
 * The geometric solver remains shared; only feature extraction and descriptor
 * distance change. The model asset is loaded lazily when solve() is called.
 */
internal class AiReconstructionEngine(
    context: Context,
    private val modelAssetPath: String = "ai/feature_model.tflite"
) : ReconstructionEngine {
    private val appContext = context.applicationContext

    override suspend fun solve(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit
    ): CameraTrackingResult {
        val detector = OnDeviceFeatureDetector(
            context = appContext,
            modelAssetPath = modelAssetPath,
            maximumFeatures = MAXIMUM_FEATURES
        )
        return try {
            val adapter = AiFeatureFrameAdapter(detector)
            GlobalReconstructionEngine(
                frameFeatureExtractor = adapter::extract,
                descriptorNormType = Core.NORM_L2
            ).solve(frameCount, cameraMatrix, frameProvider, onProgress)
        } finally {
            detector.close()
        }
    }

    private companion object {
        const val MAXIMUM_FEATURES = 500
    }
}

/** Converts model observations into the frame representation consumed by SfM. */
private class AiFeatureFrameAdapter(
    private val detector: OnDeviceFeatureDetector
) {
    fun extract(source: Mat, frameIndex: Int, maximumFeatures: Int): GlobalReconstructionEngine.FrameFeatures {
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
            )

            val valid = observations.asSequence()
                .filter { observation ->
                    val descriptor = observation.descriptor
                    descriptor != null && descriptor.size == detector.descriptorSize &&
                        observation.x.isFinite() && observation.y.isFinite() &&
                        descriptor.all { it.isFinite() }
                }
                .take(maximumFeatures.coerceAtLeast(0))
                .toList()

            if (valid.isEmpty()) {
                return GlobalReconstructionEngine.FrameFeatures(
                    emptyArray(), Mat(), source.cols(), source.rows()
                )
            }

            val descriptorSize = detector.descriptorSize
            val descriptorValues = FloatArray(valid.size * descriptorSize)
            valid.forEachIndexed { index, observation ->
                System.arraycopy(
                    observation.descriptor!!,
                    0,
                    descriptorValues,
                    index * descriptorSize,
                    descriptorSize
                )
            }
            val descriptors = Mat(valid.size, descriptorSize, CvType.CV_32F)
            try {
                check(descriptors.put(0, 0, descriptorValues) == descriptorValues.size) {
                    "Unable to pack learned feature descriptors."
                }
                val keypoints = Array(valid.size) { index ->
                    val observation = valid[index]
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

}
