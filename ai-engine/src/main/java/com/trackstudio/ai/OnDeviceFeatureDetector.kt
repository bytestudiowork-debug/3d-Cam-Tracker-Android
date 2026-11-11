package com.trackstudio.ai

import android.content.Context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Runs the packaged dense local-feature TFLite model.
 *
 * Supports a single-frame dense model and a paired-input export. When the
 * latter is used, the same frame is supplied to both image inputs.
 */
class OnDeviceFeatureDetector(
    context: Context,
    modelAssetPath: String = "ai/feature_model.tflite",
    private val confidenceThreshold: Float = 0.001f,
    private val maximumFeatures: Int = 500,
    threadCount: Int = 2
) : FeatureDetector, AutoCloseable {

    private val interpreter: Interpreter
    private val inputWidth: Int
    private val inputHeight: Int
    private val inputChannels: Int
    private val inputChannelsFirst: Boolean
    private val cellWidth: Int
    private val cellHeight: Int
    private val legacyCellScores: Boolean
    private val descriptorChannelsFirst: Boolean
    private val inputBuffers: Array<ByteBuffer>
    private val outputBuffers: Array<ByteBuffer>
    private val outputValues: Array<FloatArray>
    private val modelInputs: Array<Any>
    private val modelOutputs: MutableMap<Int, Any>

    init {
        require(threadCount > 0)
        require(confidenceThreshold in 0f..1f && maximumFeatures > 0)
        val modelBuffer = context.assets.openFd(modelAssetPath).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).channel.use { channel ->
                channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.startOffset,
                    descriptor.declaredLength
                ).order(ByteOrder.nativeOrder())
            }
        }
        interpreter = Interpreter(modelBuffer, Interpreter.Options().setNumThreads(threadCount))

        require(interpreter.inputTensorCount in 1..2) {
            "The feature model must have one or two image inputs."
        }
        val inputShape = interpreter.getInputTensor(0).shape()
        require(interpreter.getInputTensor(0).dataType() == DataType.FLOAT32) {
            "The feature model input must use float32 values."
        }
        require(inputShape.size == 4 && inputShape[0] == 1) {
            "The feature model input must have a fixed batch-one image shape."
        }
        inputChannelsFirst = inputShape[1] in 1..4
        inputChannels = if (inputChannelsFirst) inputShape[1] else inputShape[3]
        require(inputChannels in 1..4) { "Unsupported feature model input channel count." }
        inputHeight = if (inputChannelsFirst) inputShape[2] else inputShape[1]
        inputWidth = if (inputChannelsFirst) inputShape[3] else inputShape[2]
        require(inputHeight > 0 && inputWidth > 0 &&
            inputHeight % 8 == 0 && inputWidth % 8 == 0) {
            "The feature model input dimensions must be positive multiples of eight."
        }
        cellHeight = inputHeight / 8
        cellWidth = inputWidth / 8

        require(interpreter.outputTensorCount in 2..4) {
            "The feature model must expose two dense outputs or a two-image output pair."
        }
        val scoreTensor = interpreter.getOutputTensor(0)
        val descriptorTensor = interpreter.getOutputTensor(1)
        require(scoreTensor.dataType() == DataType.FLOAT32 &&
            descriptorTensor.dataType() == DataType.FLOAT32) {
            "The feature model score and descriptor outputs must use float32 values."
        }
        val scoreShape = scoreTensor.shape()
        val descriptorShape = descriptorTensor.shape()
        legacyCellScores = scoreShape.contentEquals(intArrayOf(1, cellHeight, cellWidth, 65))
        if (legacyCellScores) {
            require(descriptorShape.contentEquals(intArrayOf(1, cellHeight, cellWidth, 256))) {
                "The legacy descriptor output must have [1, height/8, width/8, 256] layout."
            }
            descriptorChannelsFirst = false
        } else {
            val validScoreShape =
                scoreShape.contentEquals(intArrayOf(1, inputHeight, inputWidth)) ||
                    scoreShape.contentEquals(intArrayOf(1, inputHeight, inputWidth, 1)) ||
                    scoreShape.contentEquals(intArrayOf(1, 1, inputHeight, inputWidth))
            require(validScoreShape) {
                "The dense score output must contain one full-resolution score per pixel."
            }
            descriptorChannelsFirst = descriptorShape.contentEquals(
                intArrayOf(1, 256, cellHeight, cellWidth)
            )
            val descriptorChannelsLast = descriptorShape.contentEquals(
                intArrayOf(1, cellHeight, cellWidth, 256)
            )
            require(descriptorChannelsFirst || descriptorChannelsLast) {
                "The dense descriptor output must have 256 channels at 1/8 resolution."
            }
        }

        inputBuffers = Array(interpreter.inputTensorCount) { index ->
            val tensor = interpreter.getInputTensor(index)
            require(tensor.dataType() == DataType.FLOAT32 && tensor.shape().contentEquals(inputShape)) {
                "All feature model image inputs must have the same float32 shape."
            }
            ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        outputBuffers = Array(interpreter.outputTensorCount) { index ->
            val tensor = interpreter.getOutputTensor(index)
            require(tensor.dataType() == DataType.FLOAT32) {
                "All feature model outputs must use float32 values."
            }
            ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        outputValues = Array(2) { index ->
            FloatArray(interpreter.getOutputTensor(index).numBytes() / Float.SIZE_BYTES)
        }
        modelInputs = Array<Any>(inputBuffers.size) { inputBuffers[it] }
        modelOutputs = HashMap<Int, Any>(outputBuffers.size).apply {
            outputBuffers.forEachIndexed { index, buffer -> put(index, buffer) }
        }
    }

    @Synchronized
    override fun detect(frame: TrackingFrame): List<FeatureObservation> {
        inputBuffers[0].clear()
        fillInput(frame, inputBuffers[0])
        inputBuffers[0].rewind()
        for (index in 1 until inputBuffers.size) {
            inputBuffers[index].clear()
            inputBuffers[index].put(inputBuffers[0].duplicate())
            inputBuffers[index].rewind()
        }
        outputBuffers.forEach { it.clear() }
        interpreter.runForMultipleInputsOutputs(modelInputs, modelOutputs)
        outputBuffers[0].rewind()
        outputBuffers[1].rewind()
        outputBuffers[0].asFloatBuffer().get(outputValues[0])
        outputBuffers[1].asFloatBuffer().get(outputValues[1])

        val decoded = if (legacyCellScores) {
            DenseFeatureDecoder.decode(
                frameIndex = frame.frameIndex,
                imageWidth = inputWidth,
                imageHeight = inputHeight,
                cellHeight = cellHeight,
                cellWidth = cellWidth,
                scores = outputValues[0],
                descriptors = outputValues[1],
                confidenceThreshold = confidenceThreshold,
                maximumFeatures = maximumFeatures
            )
        } else {
            DenseFeatureDecoder.decodePixelScores(
                frameIndex = frame.frameIndex,
                imageWidth = inputWidth,
                imageHeight = inputHeight,
                cellHeight = cellHeight,
                cellWidth = cellWidth,
                scores = outputValues[0],
                descriptors = outputValues[1],
                confidenceThreshold = confidenceThreshold,
                maximumFeatures = maximumFeatures,
                descriptorChannelsFirst = descriptorChannelsFirst
            )
        }
        val sourceScaled = if (frame.width == inputWidth && frame.height == inputHeight) {
            decoded
        } else {
            decoded.map { observation ->
                observation.copy(
                    x = observation.x * frame.width / inputWidth,
                    y = observation.y * frame.height / inputHeight
                )
            }
        }
        return ObservationFilter.filter(
            sourceScaled,
            frameWidth = frame.width,
            frameHeight = frame.height,
            maximumCount = maximumFeatures
        )
    }

    private fun fillInput(frame: TrackingFrame, target: ByteBuffer) {
        if (inputChannelsFirst) {
            for (channel in 0 until inputChannels) {
                for (y in 0 until inputHeight) {
                    for (x in 0 until inputWidth) {
                        target.putFloat(sample(frame, x, y, channel))
                    }
                }
            }
        } else {
            for (y in 0 until inputHeight) {
                for (x in 0 until inputWidth) {
                    for (channel in 0 until inputChannels) {
                        target.putFloat(sample(frame, x, y, channel))
                    }
                }
            }
        }
    }

    private fun sample(frame: TrackingFrame, x: Int, y: Int, channel: Int): Float {
        val sourceX = ((x + 0.5f) * frame.width / inputWidth - 0.5f)
            .coerceIn(0f, (frame.width - 1).toFloat())
        val sourceY = ((y + 0.5f) * frame.height / inputHeight - 0.5f)
            .coerceIn(0f, (frame.height - 1).toFloat())
        val x0 = sourceX.toInt()
        val y0 = sourceY.toInt()
        val x1 = minOf(x0 + 1, frame.width - 1)
        val y1 = minOf(y0 + 1, frame.height - 1)
        val fx = sourceX - x0
        val fy = sourceY - y0
        val top = color(frame, x0, y0, channel) * (1f - fx) +
            color(frame, x1, y0, channel) * fx
        val bottom = color(frame, x0, y1, channel) * (1f - fx) +
            color(frame, x1, y1, channel) * fx
        return (top * (1f - fy) + bottom * fy) / 255f
    }

    private fun color(frame: TrackingFrame, x: Int, y: Int, channel: Int): Float {
        val offset = (y * frame.width + x) * frame.channels
        if (inputChannels == 1) {
            val red = frame.pixels[offset].toInt() and 0xff
            if (frame.channels == 1) return red.toFloat()
            val green = frame.pixels[offset + 1].toInt() and 0xff
            val blue = frame.pixels[offset + 2].toInt() and 0xff
            return 0.299f * red + 0.587f * green + 0.114f * blue
        }
        val sourceChannel = when {
            frame.channels == 1 -> 0
            channel < frame.channels -> channel
            else -> 0
        }
        return (frame.pixels[offset + sourceChannel].toInt() and 0xff).toFloat()
    }

    @Synchronized
    override fun close() {
        interpreter.close()
    }
}
