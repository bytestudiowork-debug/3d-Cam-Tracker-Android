package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.util.Random
import kotlin.math.roundToInt

@RunWith(AndroidJUnit4::class)
class CameraTrackingEngineEndToEndTest {
    @Test
    fun nativeSparseL1BridgePreservesFrameAndAxisLayout() {
        assertTrue("Native sparse solver library is unavailable", NativeL1Solver.isAvailable)
        NativeL1Solver.resetSuccessfulInvocationsForTest()

        val actual = NativeL1Solver.solve(
            frameCount = 3,
            firstFrames = intArrayOf(0, 1, 0),
            secondFrames = intArrayOf(1, 2, 2),
            residuals = doubleArrayOf(
                0.1, -0.2, 0.3,
                -0.5, 0.7, -0.9,
                -0.4, 0.5, -0.6
            ),
            anchorResidual = doubleArrayOf(0.02, -0.03, 0.04),
            maxIterations = 1000
        ) ?: throw AssertionError("Native sparse solver rejected the bridge fixture")
        val expected = doubleArrayOf(
            0.02, -0.03, 0.04,
            0.12, -0.23, 0.34,
            -0.38, 0.47, -0.56
        )

        assertEquals("native bridge output size", expected.size, actual.size)
        expected.indices.forEach { index ->
            assertEquals("native bridge value at $index", expected[index], actual[index], 1e-10)
        }
        assertTrue("Native sparse solver did not complete a call", NativeL1Solver.successfulInvocations > 0)
    }

    @Test
    fun cleanAndDisplacedOutlierSequencesRunThroughNativeSolveAndExport() {
        assertTrue("Native sparse solver library is unavailable", NativeL1Solver.isAvailable)
        NativeL1Solver.resetSuccessfulInvocationsForTest()
        val cleanFrames = SyntheticSequence.render()
        val cleanResult = try {
            solve(cleanFrames)
        } finally {
            cleanFrames.forEach(Mat::release)
        }
        assertSuccessfulGlobalResult(cleanResult, "clean", requireBundleAdjustmentApplied = true)
        assertTrue(
            "Global rotation averaging did not invoke the native sparse solver",
            NativeL1Solver.successfulInvocations > 0
        )

        val cleanJson = StringBuilder().also { CameraSolveExporter.writeJson(cleanResult, it) }.toString()
        val cleanRoot = JSONObject(cleanJson)
        val cleanSummary = cleanRoot.getJSONObject("summary")
        assertEquals(SyntheticSequence.FRAME_COUNT, cleanSummary.getInt("processedFrames"))
        assertTrue(cleanSummary.getBoolean("globalPositioningApplied"))
        assertEquals(cleanResult.globalBundleAdjustmentApplied, cleanSummary.getBoolean("globalBundleAdjustmentApplied"))
        assertEquals(cleanResult.globalBundleAdjustmentStatus, cleanSummary.getString("globalBundleAdjustmentStatus"))
        assertEquals(cleanResult.poses.size, cleanRoot.getJSONArray("cameraPoses").length())
        assertEquals(cleanResult.landmarks.size, cleanRoot.getJSONArray("landmarks").length())
        assertFalse(cleanJson.contains("NaN"))
        assertFalse(cleanJson.contains("Infinity"))

        val outlierFrames = SyntheticSequence.render(displacedOutliers = true)
        val outlierResult = try {
            solve(outlierFrames)
        } finally {
            outlierFrames.forEach(Mat::release)
        }
        assertSuccessfulGlobalResult(outlierResult, "outlier", requireBundleAdjustmentApplied = false)

        val outlierJson = JSONObject(CameraSolveExporter.toJson(outlierResult))
        val outlierSummary = outlierJson.getJSONObject("summary")
        assertTrue(outlierSummary.getBoolean("globalPositioningApplied"))
        assertEquals(outlierResult.globalBundleAdjustmentApplied, outlierSummary.getBoolean("globalBundleAdjustmentApplied"))
        assertEquals(outlierResult.globalBundleAdjustmentStatus, outlierSummary.getString("globalBundleAdjustmentStatus"))
        assertTrue(outlierJson.getJSONArray("landmarks").length() >= 20)
    }

    @Test
    fun insufficientFrameCountFailsBeforeCallingTheProvider() {
        val cameraMatrix = SyntheticSequence.cameraMatrix()
        try {
            val failure = assertThrows(IllegalArgumentException::class.java) {
                runBlocking {
                    engine().solve(2, cameraMatrix, frameProvider = {
                        throw AssertionError("The provider must not be called for an invalid frame count")
                    })
                }
            }
            assertTrue(failure.message.orEmpty().contains("at least three frames"))
        } finally {
            cameraMatrix.release()
        }
    }

    @Test
    fun missingFrameFailsWithItsOneBasedIndex() {
        val frames = SyntheticSequence.blankFrames()
        val cameraMatrix = SyntheticSequence.cameraMatrix()
        try {
            val failure = assertThrows(IllegalStateException::class.java) {
                runBlocking {
                    engine().solve(
                        SyntheticSequence.FRAME_COUNT,
                        cameraMatrix,
                        frameProvider = { index -> if (index == 2) null else frames[index].clone() }
                    )
                }
            }
            assertTrue(failure.message.orEmpty().contains("frame 3"))
        } finally {
            frames.forEach(Mat::release)
            cameraMatrix.release()
        }
    }

    @Test
    fun featurelessFramesFailAfterRealFeatureExtraction() {
        val frames = SyntheticSequence.blankFrames()
        val cameraMatrix = SyntheticSequence.cameraMatrix()
        try {
            val failure = assertThrows(IllegalStateException::class.java) {
                runBlocking {
                    engine().solve(
                        SyntheticSequence.FRAME_COUNT,
                        cameraMatrix,
                        frameProvider = { index -> frames[index].clone() }
                    )
                }
            }
            assertTrue(failure.message.orEmpty().contains("No image pairs passed geometric verification"))
        } finally {
            frames.forEach(Mat::release)
            cameraMatrix.release()
        }
    }

    private fun solve(frames: List<Mat>): CameraTrackingResult {
        val cameraMatrix = SyntheticSequence.cameraMatrix()
        try {
            return runBlocking {
                engine().solve(
                    frameCount = frames.size,
                    cameraMatrix = cameraMatrix,
                    frameProvider = { index -> frames[index].clone() }
                )
            }
        } finally {
            cameraMatrix.release()
        }
    }

    private fun engine() = CameraReconstructionEngine(
        mode = ReconstructionMode.GLOBAL,
        options = CameraReconstructionOptions(
            globalMaxFeatures = 1200,
            ratioThreshold = 0.8f,
            maximumPairCount = 10
        )
    )

    private fun assertSuccessfulGlobalResult(
        result: CameraTrackingResult,
        label: String,
        requireBundleAdjustmentApplied: Boolean
    ) {
        assertEquals("$label decoded frame count", SyntheticSequence.FRAME_COUNT, result.processedFrames)
        assertTrue("$label registered cameras=${result.registeredCameraCount}", result.registeredCameraCount >= 3)
        assertTrue("$label verified pairs=${result.solvedFramePairs}", result.solvedFramePairs >= 3)
        assertTrue("$label graph edges=${result.viewGraphEdgeCount}", result.viewGraphEdgeCount >= 3)
        assertTrue("$label matches=${result.totalMatches}", result.totalMatches >= 60)
        assertTrue("$label inliers=${result.totalInliers}", result.totalInliers >= 40)
        assertTrue(result.inlierRatio.isFinite() && result.inlierRatio in 0.15..1.0)
        assertTrue("$label tracks=${result.featureTracks.size}", result.featureTracks.size >= 20)
        assertTrue("$label landmarks=${result.landmarks.size}", result.landmarks.size >= 20)
        assertTrue("$label native positioning not applied", result.globalPositioningApplied)
        assertTrue(result.globalPositioningInitialCost.isFinite())
        assertTrue(result.globalPositioningFinalCost.isFinite())
        assertTrue(SolverObjective.didNotIncrease(result.globalPositioningInitialCost, result.globalPositioningFinalCost))
        if (result.globalBundleAdjustmentApplied) {
            assertEquals("Applied", result.globalBundleAdjustmentStatus)
            assertTrue(result.globalBundleAdjustmentIterations > 0)
            assertTrue(result.globalBundleAdjustmentInitialCost.isFinite())
            assertTrue(result.globalBundleAdjustmentFinalCost.isFinite())
            assertTrue(SolverObjective.didNotIncrease(result.globalBundleAdjustmentInitialCost, result.globalBundleAdjustmentFinalCost))
            assertTrue(result.globalBundleAdjustmentPositiveDepthRatio >= 0.9)
            assertNotNull(result.refinedCameraIntrinsics)
        } else {
            assertTrue(
                "$label native bundle adjustment was unavailable or rejected unexpectedly",
                !requireBundleAdjustmentApplied &&
                    result.globalBundleAdjustmentStatus == "Objective increased; retained global-positioning result"
            )
        }
        assertTrue(result.poses.all { pose ->
            pose.rotation.size == 9 && pose.translation.size == 3 &&
                pose.rotation.all(Double::isFinite) && pose.translation.all(Double::isFinite)
        })
        assertTrue(result.landmarks.all { landmark ->
            landmark.position.x.isFinite() && landmark.position.y.isFinite() && landmark.position.z.isFinite() &&
                landmark.meanReprojectionErrorPx.isFinite()
        })
        assertTrue("$label reprojection error is too high", result.landmarks.map { it.meanReprojectionErrorPx }.average() < 8.0)
    }
}

private object SyntheticSequence {
    const val WIDTH = 640
    const val HEIGHT = 480
    const val FRAME_COUNT = 5

    private const val FX = 700.0
    private const val FY = 700.0
    private const val CX = 320.0
    private const val CY = 240.0
    private const val CAMERA_STEP_WORLD = 0.2
    private const val PATCH_WORLD_SIZE = 0.24
    private const val COLUMNS = 9
    private const val ROWS = 8

    private data class Patch(
        val worldX: Double,
        val worldY: Double,
        val depth: Int,
        val pixelSize: Int,
        val column: Int,
        val row: Int,
        val pixels: ByteArray
    )

    data class PixelBox(
        val left: Int,
        val top: Int,
        val right: Int,
        val bottom: Int
    ) {
        fun overlaps(other: PixelBox): Boolean =
            left < other.right && right > other.left && top < other.bottom && bottom > other.top
    }

    private data class Projection(
        val box: PixelBox
    )

    fun cameraMatrix(): Mat {
        check(OpenCvRuntime.ensureLoaded()) { "OpenCV could not be loaded in the emulator" }
        return Mat(3, 3, CvType.CV_64F).also { matrix ->
            val values = doubleArrayOf(
                FX, 0.0, CX,
                0.0, FY, CY,
                0.0, 0.0, 1.0
            )
            assertEquals(values.size, matrix.put(0, 0, *values))
        }
    }

    fun render(displacedOutliers: Boolean = false): List<Mat> {
        check(OpenCvRuntime.ensureLoaded()) { "OpenCV could not be loaded in the emulator" }
        val patches = buildPatches()
        return (0 until FRAME_COUNT).map { frameIndex ->
            val image = ByteArray(WIDTH * HEIGHT) { 127.toByte() }
            val depthBuffer = IntArray(WIDTH * HEIGHT) { Int.MAX_VALUE }
            val boxes = ArrayList<PixelBox>(patches.size)
            patches.forEach { patch ->
                val projection = project(patch, frameIndex, displacedOutliers)
                val box = projection.box
                check(box.left >= 0 && box.top >= 0 && box.right <= WIDTH && box.bottom <= HEIGHT) {
                    "Synthetic patch is clipped in frame $frameIndex"
                }
                check(boxes.none(box::overlaps)) { "Synthetic patches overlap in frame $frameIndex" }
                boxes += box
                paste(image, depthBuffer, patch.pixels, patch.pixelSize, box.left, box.top, patch.depth)
            }
            Mat(HEIGHT, WIDTH, CvType.CV_8UC1).also { mat ->
                val bytesWritten = mat.put(0, 0, image)
                if (bytesWritten != image.size) {
                    mat.release()
                    error("Could not fill synthetic frame $frameIndex")
                }
            }
        }
    }

    fun blankFrames(): List<Mat> {
        check(OpenCvRuntime.ensureLoaded()) { "OpenCV could not be loaded in the emulator" }
        val pixels = ByteArray(WIDTH * HEIGHT) { 127.toByte() }
        return List(FRAME_COUNT) { index ->
            Mat(HEIGHT, WIDTH, CvType.CV_8UC1).also { mat ->
                val bytesWritten = mat.put(0, 0, pixels)
                if (bytesWritten != pixels.size) {
                    mat.release()
                    error("Could not fill blank frame $index")
                }
            }
        }
    }

    private fun buildPatches(): List<Patch> {
        val depths = intArrayOf(4, 5, 7, 10)
        return buildList {
            for (row in 0 until ROWS) {
                for (column in 0 until COLUMNS) {
                    val id = row * COLUMNS + column
                    val depth = depths[row % depths.size]
                    val pixelX = 180.0 + column * 50.0
                    val pixelY = 60.0 + row * 55.0
                    val worldX = (pixelX - CX) * depth / FX
                    val worldY = (pixelY - CY) * depth / FY
                    val pixelSize = (FX * PATCH_WORLD_SIZE / depth).roundToInt()
                    add(
                        Patch(
                            worldX = worldX,
                            worldY = worldY,
                            depth = depth,
                            pixelSize = pixelSize,
                            column = column,
                            row = row,
                            pixels = makePattern(id, pixelSize)
                        )
                    )
                }
            }
        }
    }

    private fun project(patch: Patch, frameIndex: Int, displacedOutliers: Boolean): Projection {
        val cameraCenterX = frameIndex * CAMERA_STEP_WORLD
        var centerX = (CX + FX * (patch.worldX - cameraCenterX) / patch.depth).roundToInt()
        var centerY = (CY + FY * patch.worldY / patch.depth).roundToInt()
        if (displacedOutliers && frameIndex == 2 && isOutlier(patch)) {
            centerX += if ((patch.row / 2 + patch.column) % 2 == 0) 6 else -6
            centerY += if ((patch.column / 2 + patch.row / 2) % 2 == 0) 6 else -6
        }
        val left = centerX - patch.pixelSize / 2
        val top = centerY - patch.pixelSize / 2
        return Projection(PixelBox(left, top, left + patch.pixelSize, top + patch.pixelSize))
    }

    private fun isOutlier(patch: Patch): Boolean = patch.column % 2 == 1 && patch.row % 2 == 0

    private fun makePattern(id: Int, size: Int): ByteArray {
        val random = Random(0x5EEDL + id * 7919L)
        return ByteArray(size * size) { index ->
            val x = index % size
            val y = index / size
            if (x < 2 || y < 2 || x >= size - 2 || y >= size - 2) {
                127.toByte()
            } else if (random.nextBoolean()) {
                245.toByte()
            } else {
                9.toByte()
            }
        }
    }

    private fun paste(
        destination: ByteArray,
        depthBuffer: IntArray,
        patch: ByteArray,
        size: Int,
        left: Int,
        top: Int,
        depth: Int
    ) {
        for (y in 0 until size) {
            val targetY = top + y
            for (x in 0 until size) {
                val targetX = left + x
                val index = targetY * WIDTH + targetX
                if (depth < depthBuffer[index]) {
                    destination[index] = patch[y * size + x]
                    depthBuffer[index] = depth
                }
            }
        }
    }
}
