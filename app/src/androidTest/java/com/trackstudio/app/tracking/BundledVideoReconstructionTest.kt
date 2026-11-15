package com.trackstudio.app.tracking

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

@RunWith(AndroidJUnit4::class)
class BundledVideoReconstructionTest {
    @Test
    fun bundledVideoDecodesAndRunsThroughGlobalEngine() {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        assertTrue("Native feature library is unavailable", NativeSiftExtractor.isAvailable)

        val sequence = decodeSampledFrames()
        val cameraMatrix = cameraMatrix(sequence.width, sequence.height)
        try {
            val result = runBlocking {
                CameraReconstructionEngine(
                    mode = ReconstructionMode.GLOBAL,
                    options = CameraReconstructionOptions(
                        globalMaxFeatures = 900,
                        ratioThreshold = 0.8f,
                        maximumPairCount = sequence.frames.size * 2
                    )
                ).solve(
                    frameCount = sequence.frames.size,
                    cameraMatrix = cameraMatrix,
                    frameProvider = { index -> sequence.frames[index].clone() }
                )
            }

            val pairwiseInlierRatio = result.totalInliers.toDouble() / result.totalMatches.coerceAtLeast(1)
            val averageFeatureCount = result.diagnostics.map { it.detectedFeatures }.average()
            val meanLandmarkError = result.landmarks.map { it.meanReprojectionErrorPx }.average()
            val json = CameraSolveExporter.toJson(result)
            val summary = JSONObject(json).getJSONObject("summary")
            val bundleState = if (result.globalBundleAdjustmentApplied) {
                "applied:${result.globalBundleAdjustmentIterations} iterations, " +
                    "${result.globalBundleAdjustmentInitialCost}->${result.globalBundleAdjustmentFinalCost}"
            } else {
                result.globalBundleAdjustmentStatus
            }

            Log.i(
                LOG_TAG,
                "VIDEO_TRACKING_RESULT asset=$VIDEO_ASSET durationMs=${sequence.durationMs} " +
                    "sampleTimestampsMs=${sequence.timestampsMs.joinToString(",")} " +
                    "sampledFrames=${result.processedFrames}/${sequence.frames.size} " +
                    "dimensions=${sequence.width}x${sequence.height} " +
                    "averageFeatures=${"%.1f".format(averageFeatureCount)} " +
                    "featuresPerFrame=${result.diagnostics.joinToString(",") { it.detectedFeatures.toString() }} " +
                    "solvedPairs=${result.solvedFramePairs} matches=${result.totalMatches} " +
                    "inliers=${result.totalInliers} aggregateInlierRatio=${"%.4f".format(pairwiseInlierRatio)} " +
                    "meanPairInlierRatio=${"%.4f".format(result.inlierRatio)} " +
                    "tracks=${result.featureTracks.size} landmarks=${result.landmarks.size} " +
                    "meanLandmarkReprojectionErrorPx=${"%.3f".format(meanLandmarkError)} " +
                    "positioning=${result.globalPositioningStatus} " +
                    "positionCost=${result.globalPositioningInitialCost}->${result.globalPositioningFinalCost} " +
                    "positionIterations=${result.globalPositioningIterations} bundleAdjustment=$bundleState"
            )

            assertEquals(sequence.frames.size, result.processedFrames)
            assertTrue("No frame features were detected", result.diagnostics.all { it.detectedFeatures > 0 })
            assertTrue("Only ${result.registeredCameraCount} cameras registered", result.registeredCameraCount >= 3)
            assertTrue("No image pairs passed verification", result.solvedFramePairs > 0)
            assertTrue("The verified view graph is too small", result.viewGraphAcceptedEdgeCount >= 2)
            assertTrue("No verified correspondences were retained", result.totalInliers > 0)
            assertTrue("Only ${result.featureTracks.size} multi-view tracks were built", result.featureTracks.size >= 20)
            assertTrue("No finite 3D landmarks were produced", result.landmarks.isNotEmpty())
            assertTrue("The global position solve was not applied", result.globalPositioningApplied)
            assertTrue(result.globalPositioningInitialCost.isFinite())
            assertTrue(result.globalPositioningFinalCost.isFinite())
            assertTrue(
                "Global positioning objective increased: ${result.globalPositioningInitialCost} -> ${result.globalPositioningFinalCost}",
                SolverObjective.didNotIncrease(result.globalPositioningInitialCost, result.globalPositioningFinalCost)
            )
            assertTrue(result.inlierRatio.isFinite() && result.inlierRatio in 0.0..1.0)
            if (result.globalBundleAdjustmentApplied) {
                assertTrue(result.globalBundleAdjustmentInitialCost.isFinite())
                assertTrue(result.globalBundleAdjustmentFinalCost.isFinite())
                assertTrue(
                    "Bundle-adjustment objective increased",
                    SolverObjective.didNotIncrease(
                        result.globalBundleAdjustmentInitialCost,
                        result.globalBundleAdjustmentFinalCost
                    )
                )
            }
            assertEquals(result.processedFrames, summary.getInt("processedFrames"))
            assertEquals(result.registeredCameraCount, summary.getInt("registeredCameras"))
            assertEquals(result.totalInliers, summary.getInt("totalInliers"))
            assertFalse("Export contains a non-finite numeric token", json.contains("NaN") || json.contains("Infinity"))
        } finally {
            cameraMatrix.release()
            sequence.frames.forEach(Mat::release)
        }
    }

    private fun decodeSampledFrames(): DecodedSequence {
        val context = InstrumentationRegistry.getInstrumentation().context
        val assetDescriptor = context.assets.openFd(VIDEO_ASSET)
        val retriever = MediaMetadataRetriever()
        val frames = mutableListOf<Mat>()
        try {
            retriever.setDataSource(
                assetDescriptor.fileDescriptor,
                assetDescriptor.startOffset,
                assetDescriptor.length
            )
            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?: throw AssertionError("Video duration metadata is missing")
            assertTrue("Video duration is too short: ${durationMs}ms", durationMs >= 5_000L)

            val timestampsMs = SAMPLE_FRACTIONS.map { fraction ->
                (durationMs * fraction).roundToInt().toLong()
            }
            var outputWidth = 0
            var outputHeight = 0
            timestampsMs.forEach { timestampMs ->
                val bitmap = retriever.getFrameAtTime(
                    timestampMs * 1_000L,
                    MediaMetadataRetriever.OPTION_CLOSEST
                ) ?: throw AssertionError("Could not decode video frame at ${timestampMs}ms")
                try {
                    if (outputWidth == 0) {
                        val scale = minOf(1.0, MAX_FRAME_EDGE.toDouble() / max(bitmap.width, bitmap.height))
                        outputWidth = (bitmap.width * scale).roundToInt().coerceAtLeast(1)
                        outputHeight = (bitmap.height * scale).roundToInt().coerceAtLeast(1)
                    }
                    assertTrue("Decoded video frame has invalid dimensions", bitmap.width > 0 && bitmap.height > 0)
                    val rgba = Mat()
                    try {
                        Utils.bitmapToMat(bitmap, rgba)
                        val frame = Mat()
                        if (rgba.cols() == outputWidth && rgba.rows() == outputHeight) {
                            rgba.copyTo(frame)
                        } else {
                            Imgproc.resize(
                                rgba,
                                frame,
                                Size(outputWidth.toDouble(), outputHeight.toDouble()),
                                0.0,
                                0.0,
                                Imgproc.INTER_AREA
                            )
                        }
                        frames += frame
                    } finally {
                        rgba.release()
                    }
                } finally {
                    bitmap.recycle()
                }
            }
            return DecodedSequence(frames, outputWidth, outputHeight, durationMs, timestampsMs)
        } catch (failure: Throwable) {
            frames.forEach(Mat::release)
            throw failure
        } finally {
            retriever.release()
            assetDescriptor.close()
        }
    }

    private fun cameraMatrix(width: Int, height: Int): Mat {
        val focalLength = width * NOMINAL_FOCAL_FRACTION
        return Mat(3, 3, CvType.CV_64F).also { matrix ->
            val values = doubleArrayOf(
                focalLength, 0.0, width / 2.0,
                0.0, focalLength, height / 2.0,
                0.0, 0.0, 1.0
            )
            check(matrix.put(0, 0, *values) == values.size) {
                "Could not initialize the nominal pinhole camera matrix"
            }
        }
    }

    private data class DecodedSequence(
        val frames: List<Mat>,
        val width: Int,
        val height: Int,
        val durationMs: Long,
        val timestampsMs: List<Long>
    )

    private companion object {
        const val VIDEO_ASSET = "test-video.mp4"
        const val LOG_TAG = "VideoReconstruction"
        const val MAX_FRAME_EDGE = 960
        const val NOMINAL_FOCAL_FRACTION = 0.75
        val SAMPLE_FRACTIONS = listOf(0.03, 0.18, 0.33, 0.48, 0.63, 0.78, 0.93)
    }
}
