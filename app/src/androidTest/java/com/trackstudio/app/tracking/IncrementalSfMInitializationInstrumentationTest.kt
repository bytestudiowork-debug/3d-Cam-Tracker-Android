package com.trackstudio.app.tracking

import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.CvType
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.Point
import java.util.Random
import kotlin.math.hypot
import kotlin.math.roundToInt
import kotlin.math.sqrt

@RunWith(AndroidJUnit4::class)
class IncrementalSfMInitializationInstrumentationTest {
    @Test
    fun highParallaxHomographyPoseSeedsReturnedPositiveDepthMapThroughPublicEngine() {
        val fixture = SyntheticSeedScene.render(SceneKind.PLANAR_HIGH)
        val cameraMatrix = cameraMatrix()
        try {
            val oracle = inspectSeedPair(fixture, cameraMatrix)
            val geometry = oracle.geometry
            assertEquals(
                SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY,
                geometry.poseSource
            )
            assertTrue("H seed gates failed: angle=${Math.toDegrees(geometry.medianTriangulationAngleRadians)}",
                geometry.passesInitialPairGates)
            assertTrue(Math.toDegrees(geometry.medianTriangulationAngleRadians) > 16.0)
            assertTrue(oracle.evaluation.accepted)
            assertEquals(geometry.poseSource, oracle.evaluation.candidate.poseSource)

            val result = solvePublicEngine(fixture, cameraMatrix)
            assertConsistentSeedResult(result, oracle)
        } finally {
            fixture.release()
            cameraMatrix.release()
        }
    }

    @Test
    fun homographyPoseUsesHigherSupportFundamentalMaskForReturnedSeedTracks() {
        val fixture = SyntheticSeedScene.render(SceneKind.MIXED_HIGH)
        val cameraMatrix = cameraMatrix()
        try {
            val oracle = inspectSeedPair(fixture, cameraMatrix)
            val geometry = oracle.geometry
            assertEquals(SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY, geometry.poseSource)
            assertEquals(SourceFaithfulTwoViewEstimator.InlierModel.FUNDAMENTAL, geometry.selectedModel)
            assertTrue(
                "Expected higher F support than H: F=${geometry.fundamental.inlierCount}, " +
                    "H=${geometry.homography.inlierCount}",
                geometry.fundamental.inlierCount > geometry.homography.inlierCount
            )
            assertTrue(
                geometry.homography.inlierCount.toDouble() /
                    geometry.fundamental.inlierCount > 0.8
            )
            assertEquals(
                geometry.fundamental.inlierMask.toList(),
                geometry.selectedInlierMask.toList()
            )
            assertTrue(geometry.passesInitialPairGates)
            assertTrue(oracle.evaluation.accepted)
            assertEquals(geometry.poseSource, oracle.evaluation.candidate.poseSource)
            assertEquals(geometry.selectedModel, oracle.evaluation.candidate.selectedInlierModel)

            val positiveDepthFOnlyPairs = geometry.pose!!.inlierIndices
                .filter { !geometry.homography.inlierMask[it] }
                .mapTo(linkedSetOf()) { oracle.inputMatches[it] }
            assertTrue("Synthetic scene did not preserve positive-depth F-only correspondences",
                positiveDepthFOnlyPairs.isNotEmpty())

            val result = solvePublicEngine(fixture, cameraMatrix)
            assertConsistentSeedResult(result, oracle)
            val returnedSeedPairs = seedPairs(result)
            assertTrue(
                "The returned map dropped F-selected points outside the H consensus",
                returnedSeedPairs.any { it in positiveDepthFOnlyPairs }
            )
            assertTrue("No non-planar synthetic patch reached the returned map",
                result.featureTracks.any { track -> fixture.isNonPlanarPatchTrack(track) })
        } finally {
            fixture.release()
            cameraMatrix.release()
        }
    }

    @Test
    fun lowAngleHomographySeedIsRejectedByThePublicEngine() {
        val fixture = SyntheticSeedScene.render(SceneKind.PLANAR_LOW_ANGLE)
        val cameraMatrix = cameraMatrix()
        try {
            val oracle = inspectSeedPair(fixture, cameraMatrix)
            assertEquals(SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY, oracle.geometry.poseSource)
            assertFalse("Low-angle H seed unexpectedly passed its source-derived gates",
                oracle.geometry.passesInitialPairGates)
            assertTrue(Math.toDegrees(oracle.geometry.medianTriangulationAngleRadians) <= 16.0)
            assertFalse("Low-angle pair unexpectedly passed evaluation", oracle.evaluation.accepted)

            val failure = org.junit.Assert.assertThrows(IllegalStateException::class.java) {
                solvePublicEngine(fixture, cameraMatrix)
            }
            assertTrue(
                "Unexpected initialization failure: ${failure.message}",
                failure.message.orEmpty().contains("No reliable 3D initialization pair")
            )
        } finally {
            fixture.release()
            cameraMatrix.release()
        }
    }

    private fun inspectSeedPair(
        fixture: SyntheticSeedScene.Fixture,
        cameraMatrix: Mat
    ): PairOracle {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        val extractor = AndroidSiftFeatureExtractor(MAX_FEATURES)
        val matcher = SiftDescriptorMatcher(RATIO_THRESHOLD)
        val firstKeypointMat = MatOfKeyPoint()
        val secondKeypointMat = MatOfKeyPoint()
        val firstDescriptors = Mat()
        val secondDescriptors = Mat()
        val emptyMask = Mat()
        try {
            extractor.detectAndCompute(fixture.firstFrame, emptyMask, firstKeypointMat, firstDescriptors)
            extractor.detectAndCompute(fixture.secondFrame, emptyMask, secondKeypointMat, secondDescriptors)
            val firstKeypoints = firstKeypointMat.toArray()
            val secondKeypoints = secondKeypointMat.toArray()
            val inputMatches = matcher.match(firstDescriptors, secondDescriptors)
            val firstPixels = DoubleArray(inputMatches.size * 2)
            val secondPixels = DoubleArray(inputMatches.size * 2)
            inputMatches.forEachIndexed { index, (firstIndex, secondIndex) ->
                firstPixels[index * 2] = firstKeypoints[firstIndex].pt.x
                firstPixels[index * 2 + 1] = firstKeypoints[firstIndex].pt.y
                secondPixels[index * 2] = secondKeypoints[secondIndex].pt.x
                secondPixels[index * 2 + 1] = secondKeypoints[secondIndex].pt.y
            }
            val geometry = requireNotNull(
                SourceFaithfulTwoViewEstimator.estimateInitialPair(
                    firstPixelsXY = firstPixels,
                    secondPixelsXY = secondPixels,
                    cameraMatrixRowMajor = CAMERA_VALUES,
                    bothCamerasHaveFocalPriors = false,
                    maxErrorPixels = RANSAC_THRESHOLD_PX,
                    minimumEssentialToFundamentalRatio =
                        InitialPairPolicy.initialization().minimumEssentialToFundamentalInlierRatio,
                    maximumHomographyRatio = InitialPairPolicy.initialization().planarDominanceRatio
                )
            ) { "Native two-view geometry estimation failed for the deterministic synthetic scene" }
            val evaluation = OpenCvInitialPairEvaluator.evaluate(
                firstFrame = FIRST_FRAME,
                secondFrame = SECOND_FRAME,
                firstKeypoints = firstKeypoints,
                firstDescriptors = firstDescriptors,
                secondKeypoints = secondKeypoints,
                secondDescriptors = secondDescriptors,
                cameraMatrix = cameraMatrix,
                matcher = matcher,
                frameWidth = SyntheticSeedScene.WIDTH,
                frameHeight = SyntheticSeedScene.HEIGHT,
                policy = InitialPairPolicy.initialization(),
                cameraFocalLengthHasPrior = false,
                ransacThresholdPx = RANSAC_THRESHOLD_PX
            )
            if (evaluation.accepted) {
                assertEquals(geometry.selectedModel, evaluation.candidate.selectedInlierModel)
                assertEquals(geometry.poseSource, evaluation.candidate.poseSource)
                assertEquals(
                    geometry.pose!!.inlierIndices.map(inputMatches::get),
                    evaluation.candidate.verifiedMatches
                )
            }
            return PairOracle(
                evaluation = evaluation,
                geometry = geometry,
                inputMatches = inputMatches,
                firstKeypoints = firstKeypoints,
                secondKeypoints = secondKeypoints
            )
        } finally {
            matcher.clear()
            extractor.clear()
            firstKeypointMat.release()
            secondKeypointMat.release()
            firstDescriptors.release()
            secondDescriptors.release()
            emptyMask.release()
        }
    }

    private fun solvePublicEngine(
        fixture: SyntheticSeedScene.Fixture,
        cameraMatrix: Mat
    ): CameraTrackingResult = runBlocking {
        CameraTrackingEngine(
            maxFeatures = MAX_FEATURES,
            ratioThreshold = RATIO_THRESHOLD,
            cameraFocalLengthHasPrior = false
        ).solve(
            frameCount = SyntheticSeedScene.FRAME_COUNT,
            cameraMatrix = cameraMatrix,
            frameProvider = { index -> fixture.frames[index].clone() },
            onProgress = {}
        )
    }

    private fun assertConsistentSeedResult(result: CameraTrackingResult, oracle: PairOracle) {
        assertEquals(FIRST_FRAME to SECOND_FRAME, result.initialPairFrames)
        assertEquals(2, result.registeredCameraCount)
        assertEquals(CameraPoseStatus.REFERENCE, result.poses[FIRST_FRAME].status)
        assertTrue(
            "Seed camera was not returned as registered: ${result.poses[SECOND_FRAME].status}",
            result.poses[SECOND_FRAME].status == CameraPoseStatus.RELATIVE_ESTIMATE ||
                result.poses[SECOND_FRAME].status == CameraPoseStatus.PNP_REGISTERED
        )
        assertTrue(result.featureTracks.size >= 30)
        assertTrue(result.landmarks.size >= 30)

        val expectedPose = requireNotNull(oracle.evaluation.candidate.relativeRotation)
        val actualPose = result.poses[SECOND_FRAME]
        val maximumRotationDifference = expectedPose.indices.maxOf { index ->
            kotlin.math.abs(expectedPose[index] - actualPose.rotation[index])
        }
        assertTrue("Returned seed rotation diverged from the selected pose: " +
            "max element error=$maximumRotationDifference", maximumRotationDifference <= 0.03)
        val expectedTranslation = requireNotNull(
            oracle.evaluation.candidate.relativeTranslationDirection
        )
        val actualTranslationNorm = sqrt(actualPose.translation.sumOf { it * it })
        assertTrue("Returned seed translation is invalid", actualTranslationNorm > 1e-12)
        val translationDirectionAgreement = expectedTranslation.indices.sumOf { index ->
            expectedTranslation[index] * actualPose.translation[index] / actualTranslationNorm
        }
        assertTrue("Returned seed translation direction diverged: $translationDirectionAgreement",
            translationDirectionAgreement > 0.98)

        val expectedSeedPairs = oracle.evaluation.candidate.verifiedMatches.toSet()
        val returnedSeedPairs = seedPairs(result)
        assertTrue("Too few returned tracks observe both seed frames", returnedSeedPairs.size >= 30)
        assertEquals("Seed correspondence indices were duplicated",
            returnedSeedPairs.size, returnedSeedPairs.toSet().size)
        assertTrue(
            "A returned seed track does not preserve the original ordered feature-index pair",
            returnedSeedPairs.all { it in expectedSeedPairs }
        )
        val expectedOrder = oracle.evaluation.candidate.verifiedMatches
            .withIndex().associate { it.value to it.index }
        val returnedOrder = returnedSeedPairs.mapNotNull(expectedOrder::get)
        assertEquals("A returned seed track lost its source correspondence index",
            returnedSeedPairs.size, returnedOrder.size)
        assertTrue("Seed tracks no longer follow input correspondence order",
            returnedOrder.zipWithNext().all { (first, second) -> first < second })
        assertTrue(result.featureTracks.all { track ->
            track.observations.zipWithNext().all { (first, second) ->
                first.frameIndex < second.frameIndex &&
                    first.featureIndex >= 0 && second.featureIndex >= 0
            }
        })

        val pose = result.poses[SECOND_FRAME]
        for (landmark in result.landmarks) {
            val x = landmark.position.x
            val y = landmark.position.y
            val z = landmark.position.z
            val firstDepth = z
            val secondDepth = pose.rotation[6] * x + pose.rotation[7] * y +
                pose.rotation[8] * z + pose.translation[2]
            assertTrue("Landmark ${landmark.trackId} is not in front of the first camera", firstDepth > 0.0)
            assertTrue("Landmark ${landmark.trackId} is not in front of the second camera", secondDepth > 0.0)
            assertTrue("Landmark ${landmark.trackId} reprojection error is too high",
                landmark.meanReprojectionErrorPx <= 4.0)
        }
        assertNotNull(result.initialPairFrames)
    }

    private fun seedPairs(result: CameraTrackingResult): List<Pair<Int, Int>> =
        result.featureTracks.mapNotNull { track ->
            val first = track.observations.singleOrNull { it.frameIndex == FIRST_FRAME }
            val second = track.observations.singleOrNull { it.frameIndex == SECOND_FRAME }
            if (first == null || second == null || first.featureIndex < 0 || second.featureIndex < 0) {
                null
            } else {
                first.featureIndex to second.featureIndex
            }
        }

    private data class PairOracle(
        val evaluation: InitialPairEvaluation,
        val geometry: SourceFaithfulTwoViewEstimator.InitialPairEstimate,
        val inputMatches: List<Pair<Int, Int>>,
        val firstKeypoints: Array<KeyPoint>,
        val secondKeypoints: Array<KeyPoint>
    )

    private fun cameraMatrix(): Mat = Mat(3, 3, CvType.CV_64F).also { matrix ->
        check(matrix.put(0, 0, *CAMERA_VALUES) == CAMERA_VALUES.size)
    }

    private enum class SceneKind { PLANAR_HIGH, MIXED_HIGH, PLANAR_LOW_ANGLE }

    private object SyntheticSeedScene {
        const val WIDTH = 1280
        const val HEIGHT = 720
        const val FRAME_COUNT = 3
        private const val FIRST_FRAME = 0
        private const val SECOND_FRAME = 2

        private const val PATCH_SIZE = 32
        private const val FX = 800.0
        private const val FY = 800.0
        private const val CX = 640.0
        private const val CY = 360.0
        private const val HIGH_BASELINE = 1.5
        private const val LOW_BASELINE = 0.65
        private const val PLANE_DEPTH = 4.5
        private const val PLANAR_FEATURE_COUNT = 100
        private const val FEATURE_COUNT = 120

        data class Fixture(
            val frames: List<Mat>,
            val patchCentersFirst: List<Point>,
            val patchCentersSecond: List<Point>,
            val nonPlanarFeatureStart: Int
        ) {
            val firstFrame: Mat get() = frames[FIRST_FRAME]
            val secondFrame: Mat get() = frames[SECOND_FRAME]

            fun isNonPlanarPatchTrack(track: CameraFeatureTrack): Boolean {
                val first = track.observations.singleOrNull { it.frameIndex == FIRST_FRAME } ?: return false
                val second = track.observations.singleOrNull { it.frameIndex == SECOND_FRAME } ?: return false
                val firstPatch = nearestPatch(first.x, first.y, patchCentersFirst) ?: return false
                val secondPatch = nearestPatch(second.x, second.y, patchCentersSecond) ?: return false
                return firstPatch == secondPatch && firstPatch >= nonPlanarFeatureStart
            }

            fun release() = frames.forEach(Mat::release)

            private fun nearestPatch(x: Double, y: Double, centers: List<Point>): Int? {
                val best = centers.indices.minByOrNull { index ->
                    val center = centers[index]
                    val dx = x - center.x
                    val dy = y - center.y
                    dx * dx + dy * dy
                } ?: return null
                val center = centers[best]
                return best.takeIf { hypot(x - center.x, y - center.y) <= PATCH_SIZE * 0.75 }
            }
        }

        fun render(kind: SceneKind): Fixture {
            check(OpenCvRuntime.ensureLoaded()) { "OpenCV could not be loaded in the emulator" }
            val baseline = if (kind == SceneKind.PLANAR_LOW_ANGLE) LOW_BASELINE else HIGH_BASELINE
            val patches = (0 until FEATURE_COUNT).map { index ->
                val nonPlanar = kind == SceneKind.MIXED_HIGH && index >= PLANAR_FEATURE_COUNT
                val (column, row, verticalOffset) = if (kind == SceneKind.MIXED_HIGH && nonPlanar) {
                    Triple(10 + (index - PLANAR_FEATURE_COUNT) % 2,
                        (index - PLANAR_FEATURE_COUNT) / 2, 32.0)
                } else if (kind == SceneKind.PLANAR_HIGH || kind == SceneKind.PLANAR_LOW_ANGLE) {
                    Triple(index % 12, index / 12, 0.0)
                } else {
                    Triple(index % 10, index / 10, 0.0)
                }
                val depth = if (nonPlanar) {
                    2.2 + 0.71 * ((index - PLANAR_FEATURE_COUNT) % 6)
                } else {
                    PLANE_DEPTH
                }
                val pixelX = 300.0 + column * 80.0
                val pixelY = 50.0 + row * 65.0 + verticalOffset
                val worldX = (pixelX - CX) * depth / FX
                val worldY = (pixelY - CY) * depth / FY
                Patch(index, worldX, worldY, depth, makePattern(index))
            }
            val first = renderView(patches, cameraCenterX = 0.0)
            val second = renderView(patches, cameraCenterX = baseline)
            val blank = Mat(HEIGHT, WIDTH, CvType.CV_8UC1).also { image ->
                val pixels = ByteArray(WIDTH * HEIGHT) { 127.toByte() }
                check(image.put(0, 0, pixels) == pixels.size)
            }
            val firstCenters = patches.map { project(it, 0.0) }
            val secondCenters = patches.map { project(it, baseline) }
            return Fixture(
                frames = listOf(first, blank, second),
                patchCentersFirst = firstCenters,
                patchCentersSecond = secondCenters,
                nonPlanarFeatureStart = if (kind == SceneKind.MIXED_HIGH) PLANAR_FEATURE_COUNT else FEATURE_COUNT
            )
        }

        private data class Patch(
            val id: Int,
            val worldX: Double,
            val worldY: Double,
            val depth: Double,
            val pixels: ByteArray
        )

        private fun renderView(patches: List<Patch>, cameraCenterX: Double): Mat {
            val image = ByteArray(WIDTH * HEIGHT) { 127.toByte() }
            val occupied = mutableListOf<Box>()
            for (patch in patches) {
                val center = project(patch, cameraCenterX)
                val left = center.x.roundToInt() - PATCH_SIZE / 2
                val top = center.y.roundToInt() - PATCH_SIZE / 2
                val box = Box(left, top, left + PATCH_SIZE, top + PATCH_SIZE)
                check(box.left >= 0 && box.top >= 0 && box.right <= WIDTH && box.bottom <= HEIGHT) {
                    "Synthetic patch ${patch.id} is clipped at baseline $cameraCenterX"
                }
                check(occupied.none(box::overlaps)) {
                    "Synthetic patches overlap at baseline $cameraCenterX"
                }
                occupied += box
                for (y in 0 until PATCH_SIZE) {
                    val target = (top + y) * WIDTH + left
                    System.arraycopy(patch.pixels, y * PATCH_SIZE, image, target, PATCH_SIZE)
                }
            }
            return Mat(HEIGHT, WIDTH, CvType.CV_8UC1).also { mat ->
                check(mat.put(0, 0, image) == image.size)
            }
        }

        private data class Box(val left: Int, val top: Int, val right: Int, val bottom: Int) {
            fun overlaps(other: Box): Boolean =
                left < other.right && right > other.left && top < other.bottom && bottom > other.top
        }

        private fun project(patch: Patch, cameraCenterX: Double): Point = Point(
            (CX + FX * (patch.worldX - cameraCenterX) / patch.depth).roundToInt().toDouble(),
            (CY + FY * patch.worldY / patch.depth).roundToInt().toDouble()
        )

        private fun makePattern(id: Int): ByteArray {
            val random = Random(0x5EEDL + id * 7919L)
            return ByteArray(PATCH_SIZE * PATCH_SIZE) { index ->
                val x = index % PATCH_SIZE
                val y = index / PATCH_SIZE
                if (x < 2 || y < 2 || x >= PATCH_SIZE - 2 || y >= PATCH_SIZE - 2) {
                    127.toByte()
                } else if (random.nextBoolean()) {
                    245.toByte()
                } else {
                    9.toByte()
                }
            }
        }
    }

    private companion object {
        const val FIRST_FRAME = 0
        const val SECOND_FRAME = 2
        const val MAX_FEATURES = 1400
        const val RATIO_THRESHOLD = 0.8f
        const val RANSAC_THRESHOLD_PX = 4.0
        val CAMERA_VALUES = doubleArrayOf(
            800.0, 0.0, 640.0,
            0.0, 800.0, 360.0,
            0.0, 0.0, 1.0
        )
    }
}
