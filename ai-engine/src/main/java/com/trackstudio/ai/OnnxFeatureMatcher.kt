package com.trackstudio.ai

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.exp

class OnnxFeatureMatcher(
    context: Context,
    modelAssetPath: String = "ai/matcher.onnx",
    private val maximumFeatures: Int = 1400,
    private val minimumMatchConfidence: Float = 0.1f,
    private val descriptorSize: Int = 64,
    threadCount: Int = 2
) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        require(maximumFeatures > 0)
        require(minimumMatchConfidence in 0f..1f)
        require(descriptorSize > 0)
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
    fun matchRaw(
        firstKeypoints: FloatArray,
        firstDescriptors: FloatArray,
        firstCount: Int,
        firstWidth: Int,
        firstHeight: Int,
        secondKeypoints: FloatArray,
        secondDescriptors: FloatArray,
        secondCount: Int,
        secondWidth: Int,
        secondHeight: Int
    ): List<FeatureMatch> {
        if (firstCount == 0 || secondCount == 0) return emptyList()
        require(firstCount <= maximumFeatures && secondCount <= maximumFeatures)
        require(firstKeypoints.size == firstCount * 2)
        require(secondKeypoints.size == secondCount * 2)
        require(firstDescriptors.size == firstCount * descriptorSize)
        require(secondDescriptors.size == secondCount * descriptorSize)
        require(firstWidth > 0 && firstHeight > 0 && secondWidth > 0 && secondHeight > 0)

        val inputs = LinkedHashMap<String, OnnxTensor>(6)
        try {
            inputs["keypoints0"] = tensor(firstKeypoints, 1L, firstCount.toLong(), 2L)
            inputs["descriptors0"] = tensor(firstDescriptors, 1L, firstCount.toLong(), descriptorSize.toLong())
            inputs["image_size0"] = tensor(floatArrayOf(firstWidth.toFloat(), firstHeight.toFloat()), 1L, 2L)
            inputs["keypoints1"] = tensor(secondKeypoints, 1L, secondCount.toLong(), 2L)
            inputs["descriptors1"] = tensor(secondDescriptors, 1L, secondCount.toLong(), descriptorSize.toLong())
            inputs["image_size1"] = tensor(floatArrayOf(secondWidth.toFloat(), secondHeight.toFloat()), 1L, 2L)

            session.run(inputs).use { result ->
                val assignmentTensor = result.get("log_assignment").get() as OnnxTensor
                val assignmentBuffer = assignmentTensor.floatBuffer
                val assignment = FloatArray(assignmentBuffer.remaining())
                assignmentBuffer.get(assignment)
                return AssignmentDecoder.decode(
                    assignment = assignment,
                    firstCount = firstCount,
                    secondCount = secondCount,
                    minimumConfidence = minimumMatchConfidence
                )
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    private fun tensor(values: FloatArray, vararg shape: Long): OnnxTensor =
        OnnxTensor.createTensor(environment, floatBuffer(values), shape)

    private fun floatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .apply {
                put(values)
                rewind()
            }

    override fun close() {
        session.close()
    }
}

object AssignmentDecoder {
    fun decode(
        assignment: FloatArray,
        firstCount: Int,
        secondCount: Int,
        minimumConfidence: Float
    ): List<FeatureMatch> {
        require(firstCount >= 0 && secondCount >= 0)
        require(minimumConfidence in 0f..1f)
        val columns = secondCount + 1
        require(assignment.size == (firstCount + 1) * columns)

        val bestSecond = IntArray(firstCount) { -1 }
        val bestSecondScore = FloatArray(firstCount) { Float.NEGATIVE_INFINITY }
        val bestFirst = IntArray(secondCount) { -1 }
        val bestFirstScore = FloatArray(secondCount) { Float.NEGATIVE_INFINITY }

        for (first in 0 until firstCount) {
            for (second in 0 until secondCount) {
                val score = assignment[first * columns + second]
                if (!score.isFinite()) continue
                if (score > bestSecondScore[first]) {
                    bestSecondScore[first] = score
                    bestSecond[first] = second
                }
                if (score > bestFirstScore[second]) {
                    bestFirstScore[second] = score
                    bestFirst[second] = first
                }
            }
        }

        return buildList {
            for (first in 0 until firstCount) {
                val second = bestSecond[first]
                if (second < 0 || bestFirst[second] != first) continue
                val confidence = exp(bestSecondScore[first].toDouble()).toFloat()
                if (confidence < minimumConfidence || !confidence.isFinite()) continue
                add(FeatureMatch(first, second, confidence.coerceIn(0f, 1f)))
            }
        }.sortedByDescending { it.confidence }
    }
}
