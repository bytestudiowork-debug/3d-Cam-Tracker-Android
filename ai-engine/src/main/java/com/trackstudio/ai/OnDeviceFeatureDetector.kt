package com.trackstudio.ai

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * Runs a compact dense local-feature model packaged in the module assets.
 *
 * Input: float32 grayscale [1, height, width, 1].
 * Outputs: float32 scores [1, height/8, width/8, 65] and
 * descriptors [1, height/8, width/8, 256].
 */
class OnDeviceFeatureDetector(
    context: Context,
    modelAssetPath: String = "ai/feature_model.tflite",
    private val confidenceThreshold: Float = 0.015f,
    private val maximumFeatures: Int = 500,
    threadCount: Int = 2
) : FeatureDetector, AutoCloseable {

    private val interpreter: Interpreter
    private val inputWidth: Int
    private val inputHeight: Int
    private val cellWidth: Int
    private val cellHeight: Int
    private val descriptorSize: Int
    private val inputBuffer: ByteBuffer
    private val scoreOutputBuffer: ByteBuffer
    private val descriptorOutputBuffer: ByteBuffer
    private val scoreValues: FloatArray
    private val descriptorValues: FloatArray
    private val modelInputs: Array<Any>
    private val modelOutputs: MutableMap<Int, Any>

    init {
        require(threadCount > 0)
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

        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        require(inputTensor.dataType() == org.tensorflow.lite.DataType.FLOAT32) {
            "The feature model input must use float32 values."
        }
        require(inputShape.size == 4 && inputShape[0] == 1 && inputShape[3] == 1) {
            "The feature model input must use [1, height, width, 1] layout."
        }
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        require(inputHeight > 0 && inputWidth > 0 &&
            inputHeight % 8 == 0 && inputWidth % 8 == 0) {
            "The feature model input dimensions must be positive multiples of eight."
        }
        cellHeight = inputHeight / 8
        cellWidth = inputWidth / 8

        val scoreTensor = interpreter.getOutputTensor(0)
        val descriptorTensor = interpreter.getOutputTensor(1)
        val scoreShape = scoreTensor.shape()
        val descriptorShape = descriptorTensor.shape()
        require(scoreTensor.dataType() == org.tensorflow.lite.DataType.FLOAT32 &&
            descriptorTensor.dataType() == org.tensorflow.lite.DataType.FLOAT32) {
            "The feature model outputs must use float32 values."
        }
        require(scoreShape.contentEquals(intArrayOf(1, cellHeight, cellWidth, 65))) {
            "The feature model score output has an unsupported shape."
        }
        require(descriptorShape.size == 4 &&
            descriptorShape[0] == 1 &&
            descriptorShape[1] == cellHeight &&
            descriptorShape[2] == cellWidth &&
            descriptorShape[3] == 256) {
            "The feature model descriptor output has an unsupported shape."
        }
        descriptorSize = descriptorShape[3]
        require(confidenceThreshold in 0f..1f && maximumFeatures > 0)

        // Reuse direct buffers and output arrays across frames. Detection is
        // synchronized, so one set is sufficient and avoids per-frame churn.
        inputBuffer = ByteBuffer.allocateDirect(inputWidth * inputHeight * 4)
            .order(ByteOrder.nativeOrder())
        scoreOutputBuffer = ByteBuffer.allocateDirect(cellHeight * cellWidth * 65 * 4)
            .order(ByteOrder.nativeOrder())
        descriptorOutputBuffer = ByteBuffer.allocateDirect(cellHeight * cellWidth * descriptorSize * 4)
            .order(ByteOrder.nativeOrder())
        scoreValues = FloatArray(cellHeight * cellWidth * 65)
        descriptorValues = FloatArray(cellHeight * cellWidth * descriptorSize)
        modelInputs = arrayOf(inputBuffer)
        modelOutputs = hashMapOf(
            0 to scoreOutputBuffer,
            1 to descriptorOutputBuffer
        )
    }

    @Synchronized
    override fun detect(frame: TrackingFrame): List<FeatureObservation> {
        inputBuffer.clear()
        fillInput(frame, inputBuffer)
        inputBuffer.rewind()
        scoreOutputBuffer.clear()
        descriptorOutputBuffer.clear()

        interpreter.runForMultipleInputsOutputs(modelInputs, modelOutputs)

        scoreOutputBuffer.rewind()
        descriptorOutputBuffer.rewind()
        scoreOutputBuffer.asFloatBuffer().get(scoreValues)
        descriptorOutputBuffer.asFloatBuffer().get(descriptorValues)

        val decoded = DenseFeatureDecoder.decode(
            frameIndex = frame.frameIndex,
            imageWidth = frame.width,
            imageHeight = frame.height,
            cellHeight = cellHeight,
            cellWidth = cellWidth,
            scores = scoreValues,
            descriptors = descriptorValues,
            descriptorSize = descriptorSize,
            confidenceThreshold = confidenceThreshold,
            maximumFeatures = maximumFeatures
        )
        return ObservationFilter.filter(
            decoded,
            frameWidth = frame.width,
            frameHeight = frame.height,
            maximumCount = maximumFeatures
        )
    }

    private fun fillInput(frame: TrackingFrame, target: ByteBuffer) {
        for (y in 0 until inputHeight) {
            val sourceY = ((y + 0.5f) * frame.height / inputHeight - 0.5f)
                .coerceIn(0f, (frame.height - 1).toFloat())
            val y0 = sourceY.toInt()
            val y1 = minOf(y0 + 1, frame.height - 1)
            val fy = sourceY - y0

            for (x in 0 until inputWidth) {
                val sourceX = ((x + 0.5f) * frame.width / inputWidth - 0.5f)
                    .coerceIn(0f, (frame.width - 1).toFloat())
                val x0 = sourceX.toInt()
                val x1 = minOf(x0 + 1, frame.width - 1)
                val fx = sourceX - x0
                val top = gray(frame, x0, y0) * (1f - fx) + gray(frame, x1, y0) * fx
                val bottom = gray(frame, x0, y1) * (1f - fx) + gray(frame, x1, y1) * fx
                target.putFloat((top * (1f - fy) + bottom * fy) / 255f)
            }
        }
    }

    private fun gray(frame: TrackingFrame, x: Int, y: Int): Float {
        val offset = (y * frame.width + x) * frame.channels
        val red = frame.pixels[offset].toInt() and 0xff
        if (frame.channels <= 2) return red.toFloat()
        val green = frame.pixels[offset + 1].toInt() and 0xff
        val blue = frame.pixels[offset + 2].toInt() and 0xff
        return 0.299f * red + 0.587f * green + 0.114f * blue
    }

    @Synchronized
    override fun close() {
        interpreter.close()
    }
}
