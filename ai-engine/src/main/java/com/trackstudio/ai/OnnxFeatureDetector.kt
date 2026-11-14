package com.trackstudio.ai

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

class OnnxFeatureDetector(
    context: Context,
    modelAssetPath: String = "ai/extractor.onnx",
    private val maximumFeatures: Int = 1400,
    private val maximumInputDimension: Int = 640,
    private val descriptorChannels: Int = 64,
    private val nmsRadius: Int = 2,
    threadCount: Int = 2
) : FeatureDetector, AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        require(maximumFeatures > 0)
        require(maximumInputDimension >= 32)
        require(descriptorChannels > 0)
        require(nmsRadius >= 0)
        require(threadCount > 0)
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(threadCount)
        options.setInterOpNumThreads(1)
        session = try {
            environment.createSession(ModelAssetLoader.copyAsset(context.applicationContext, modelAssetPath).absolutePath, options)
        } finally {
            options.close()
        }
    }

    @Synchronized
    override fun detect(frame: TrackingFrame): List<FeatureObservation> {
        val scale = min(1f, maximumInputDimension.toFloat() / max(frame.width, frame.height))
        val inputWidth = max(32, floor(frame.width * scale / 32f).toInt() * 32)
        val inputHeight = max(32, floor(frame.height * scale / 32f).toInt() * 32)
        val input = prepareInput(frame, inputWidth, inputHeight)
        val tensor = OnnxTensor.createTensor(
            environment,
            floatBuffer(input),
            longArrayOf(1L, 3L, inputHeight.toLong(), inputWidth.toLong())
        )
        try {
            session.run(mapOf("image" to tensor)).use { result ->
                val descriptors = readFloats(result.get("descriptors").get())
                val keypointLogits = readFloats(result.get("keypoint_logits").get())
                val reliability = readFloats(result.get("reliability").get())
                val observations = FeatureMapDecoder.decode(
                    frameIndex = frame.frameIndex,
                    inputWidth = inputWidth,
                    inputHeight = inputHeight,
                    descriptorMap = descriptors,
                    keypointLogits = keypointLogits,
                    reliabilityMap = reliability,
                    maximumFeatures = maximumFeatures,
                    descriptorChannels = descriptorChannels,
                    nmsRadius = nmsRadius
                )
                val scaleX = frame.width.toFloat() / inputWidth
                val scaleY = frame.height.toFloat() / inputHeight
                return observations.map { it.copy(x = it.x * scaleX, y = it.y * scaleY) }
            }
        } finally {
            tensor.close()
        }
    }

    private fun prepareInput(frame: TrackingFrame, width: Int, height: Int): FloatArray {
        val planeSize = width * height
        val values = FloatArray(planeSize * 3)
        for (y in 0 until height) {
            val sourceY = ((y + 0.5f) * frame.height / height - 0.5f)
                .coerceIn(0f, (frame.height - 1).toFloat())
            val y0 = sourceY.toInt()
            val y1 = min(y0 + 1, frame.height - 1)
            val fy = sourceY - y0
            for (x in 0 until width) {
                val sourceX = ((x + 0.5f) * frame.width / width - 0.5f)
                    .coerceIn(0f, (frame.width - 1).toFloat())
                val x0 = sourceX.toInt()
                val x1 = min(x0 + 1, frame.width - 1)
                val fx = sourceX - x0
                val offset = y * width + x
                for (channel in 0..2) {
                    val top = color(frame, x0, y0, channel) * (1f - fx) +
                        color(frame, x1, y0, channel) * fx
                    val bottom = color(frame, x0, y1, channel) * (1f - fx) +
                        color(frame, x1, y1, channel) * fx
                    values[channel * planeSize + offset] =
                        (top * (1f - fy) + bottom * fy) / 255f
                }
            }
        }
        return values
    }

    private fun color(frame: TrackingFrame, x: Int, y: Int, channel: Int): Float {
        val offset = (y * frame.width + x) * frame.channels
        val sourceChannel = if (frame.channels == 1) 0 else channel.coerceAtMost(frame.channels - 1)
        return (frame.pixels[offset + sourceChannel].toInt() and 0xff).toFloat()
    }

    private fun floatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(values)
                rewind()
            }

    private fun readFloats(value: Any): FloatArray {
        val tensor = value as OnnxTensor
        val buffer = tensor.floatBuffer
        val values = FloatArray(buffer.remaining())
        buffer.get(values)
        return values
    }

    override fun close() {
        session.close()
    }
}
