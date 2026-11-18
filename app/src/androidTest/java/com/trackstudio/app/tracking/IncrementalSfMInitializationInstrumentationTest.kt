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
import org.opencv.core.Point3
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

    @Test
    fun homographySeedRegistersThirdCameraFromMapAndPreservesThreeViewTracks() {
        val fixture = SyntheticSeedScene.render(
            SceneKind.PLANAR_HIGH,
            includeThirdCamera = true
        )
        val cameraMatrix = cameraMatrix()
        try {
            val oracle = inspectSeedPair(fixture, cameraMatrix, cameraFocalLengthHasPrior = true)
            assertEquals(SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY, oracle.geometry.poseSource)
            assertTrue(oracle.geometry.passesInitialPairGates)

            val result = solvePublicEngine(
                fixture,
                cameraMatrix,
                cameraFocalLengthHasPrior = true
            )
            assertConsistentSeedResult(result, oracle, expectedRegisteredCameraCount = 3)
            assertEquals(FIRST_FRAME to SECOND_FRAME, result.initialPairFrames)
            assertEquals(fixture.frames.size, result.poses.size)
            assertEquals(CameraPoseStatus.PNP_REGISTERED, result.poses[THIRD_FRAME].status)
            assertTrue(result.diagnostics[THIRD_FRAME].status.contains("PnP"))

            val expectedSeedPairs = oracle.evaluation.candidate.verifiedMatches.toSet()
            val keypointsByFrame = mapOf(
                FIRST_FRAME to oracle.firstKeypoints,
                SECOND_FRAME to oracle.secondKeypoints,
                THIRD_FRAME to detectKeypoints(fixture.frames[THIRD_FRAME])
            )
            val threeViewTracks = result.featureTracks.filter { track ->
                listOf(FIRST_FRAME, SECOND_FRAME, THIRD_FRAME).all { frameIndex ->
                    track.observations.any { it.frameIndex == frameIndex }
                }
            }
            assertTrue("Only ${threeViewTracks.size} tracks retained all three registered views",
                threeViewTracks.size >= 30)
            threeViewTracks.forEach { track ->
                val observations = track.observations.associateBy { it.frameIndex }
                assertEquals(listOf(FIRST_FRAME, SECOND_FRAME, THIRD_FRAME),
                    observations.keys.sorted())
                val seedPair = observations.getValue(FIRST_FRAME).featureIndex to
                    observations.getValue(SECOND_FRAME).featureIndex
                assertTrue("A three-view track lost its ordered H-seed correspondence", seedPair in expectedSeedPairs)
                val patchIds = observations.values.map { observation ->
                    val keypoint = keypointsByFrame.getValue(observation.frameIndex)
                        .getOrNull(observation.featureIndex)
                    assertNotNull("Feature index is invalid in frame ${observation.frameIndex}", keypoint)
                    val observedKeypoint = requireNotNull(keypoint)
                    assertEquals(observedKeypoint.pt.x, observation.x, 1e-9)
                    assertEquals(observedKeypoint.pt.y, observation.y, 1e-9)
                    fixture.patchIndex(observation.frameIndex, observation)
                }
                assertTrue("A track joined different synthetic patches: $patchIds",
                    patchIds.all { it != null } && patchIds.distinct().size == 1)
            }
            for (frameIndex in listOf(FIRST_FRAME, SECOND_FRAME, THIRD_FRAME)) {
                val indices = threeViewTracks.map { track ->
                    track.observations.single { it.frameIndex == frameIndex }.featureIndex
                }
                assertEquals("Frame $frameIndex feature was assigned to multiple tracks",
                    indices.size, indices.toSet().size)
            }

            val tracksById = result.featureTracks.associateBy { it.id }
            val landmarksById = result.landmarks.associateBy { it.trackId }
            assertEquals(tracksById.keys, landmarksById.keys)
            assertTrue(result.landmarks.isNotEmpty())
            val fx = CAMERA_VALUES[0]
            val fy = CAMERA_VALUES[4]
            val cx = CAMERA_VALUES[2]
            val cy = CAMERA_VALUES[5]
            for ((trackId, track) in tracksById) {
                val landmark = landmarksById.getValue(trackId)
                assertEquals("Landmark observation count diverged for track $trackId",
                    track.observations.size, landmark.observationCount)
                val point = doubleArrayOf(
                    landmark.position.x, landmark.position.y, landmark.position.z
                )
                for (observation in track.observations) {
                    val cameraPoint = transform(result.poses[observation.frameIndex], point)
                    assertTrue("Track $trackId is behind frame ${observation.frameIndex}",
                        cameraPoint[2] > 0.0)
                    val dx = fx * cameraPoint[0] / cameraPoint[2] + cx - observation.x
                    val dy = fy * cameraPoint[1] / cameraPoint[2] + cy - observation.y
                    assertTrue("Track $trackId reprojection error exceeds tolerance",
                        hypot(dx, dy) <= 4.25)
                }
                assertTrue("Track $trackId landmark error exceeds triangulation policy",
                    landmark.meanReprojectionErrorPx <= 4.0)
            }

            val seedCenter = cameraCenter(result.poses[SECOND_FRAME])
            val thirdCenter = cameraCenter(result.poses[THIRD_FRAME])
            val seedNorm = sqrt(seedCenter.sumOf { it * it })
            val thirdNorm = sqrt(thirdCenter.sumOf { it * it })
            val centerRatio = thirdNorm / seedNorm
            val expectedRatio = SyntheticSeedScene.THIRD_CAMERA_CENTER_X /
                SyntheticSeedScene.HIGH_BASELINE
            assertEquals("Third camera changed the seed map's scale gauge",
                expectedRatio, centerRatio, 0.18)
            val centerError = sqrt(seedCenter.indices.sumOf { index ->
                val difference = thirdCenter[index] - expectedRatio * seedCenter[index]
                difference * difference
            })
            assertTrue("Third pose left the seed baseline direction or scale: error=$centerError",
                centerError / seedNorm < 0.18)
        } finally {
            fixture.release()
            cameraMatrix.release()
        }
    }

    @Test
    fun poorOverlapThirdCameraRemainsHeldWithoutCorruptingTheSeedMap() {
        val fixture = SyntheticSeedScene.render(
            SceneKind.PLANAR_HIGH,
            includeThirdCamera = true,
            thirdCameraHasTexture = false
        )
        val cameraMatrix = cameraMatrix()
        try {
            val oracle = inspectSeedPair(fixture, cameraMatrix, cameraFocalLengthHasPrior = true)
            val result = solvePublicEngine(
                fixture,
                cameraMatrix,
                cameraFocalLengthHasPrior = true
            )
            assertConsistentSeedResult(result, oracle)
            assertEquals(CameraPoseStatus.HELD_LAST_VALID, result.poses[THIRD_FRAME].status)
            assertTrue(result.featureTracks.none { track ->
                track.observations.any { it.frameIndex == THIRD_FRAME }
            })
            assertTrue("Third frame should report a failed registration attempt",
                result.diagnostics[THIRD_FRAME].status.isNotBlank() &&
                    !result.diagnostics[THIRD_FRAME].status.contains("PnP registered"))
        } finally {
            fixture.release()
            cameraMatrix.release()
        }
    }

    @Test
    fun seededPixelNoiseOutliersAndPartialThirdViewOcclusionRegisterThroughPublicEngine() {
        val fixture = SyntheticSeedScene.render(
            SceneKind.PLANAR_HIGH,
            includeThirdCamera = true,
            pixelNoiseSigmaPx = 0.8,
            secondViewOutlierFraction = 0.08,
            thirdViewOutlierFraction = 0.10,
            thirdViewVisibleFraction = 0.70
        )
        val cameraMatrix = cameraMatrix()
        try {
            val oracle = inspectSeedPair(fixture, cameraMatrix, cameraFocalLengthHasPrior = true)
            assertTrue("Seed noise/outliers lost the 100-inlier initialization support: " +
                "${oracle.geometry.numInliers}", oracle.geometry.numInliers >= 100)
            assertTrue("Seed noise/outliers crossed the source-derived initialization gates: " +
                "${oracle.evaluation.reason}", oracle.geometry.passesInitialPairGates && oracle.evaluation.accepted)
            assertEquals(oracle.geometry.selectedInlierMask.count { it }, oracle.geometry.numInliers)
            val mismatchedPatchMatches = oracle.inputMatches.indices.filter { matchIndex ->
                val (firstIndex, secondIndex) = oracle.inputMatches[matchIndex]
                val firstPoint = oracle.firstKeypoints[firstIndex].pt
                val secondPoint = oracle.secondKeypoints[secondIndex].pt
                val firstPatch = fixture.patchIndex(
                    FIRST_FRAME,
                    CameraFeatureObservation(FIRST_FRAME, firstPoint.x, firstPoint.y, firstIndex)
                )
                val secondPatch = fixture.patchIndex(
                    SECOND_FRAME,
                    CameraFeatureObservation(SECOND_FRAME, secondPoint.x, secondPoint.y, secondIndex)
                )
                firstPatch != null && secondPatch != null && firstPatch != secondPatch
            }
            assertTrue("Rendered texture swaps produced no detectable outlier correspondences",
                mismatchedPatchMatches.isNotEmpty())
            assertTrue("Selected seed mask retained too many wrong patch identities",
                mismatchedPatchMatches.count { oracle.geometry.selectedInlierMask[it] } <=
                    maxOf(1, mismatchedPatchMatches.size / 10))

            val result = solvePublicEngine(
                fixture,
                cameraMatrix,
                cameraFocalLengthHasPrior = true
            )
            assertConsistentSeedResult(result, oracle, expectedRegisteredCameraCount = 3)
            assertEquals(CameraPoseStatus.PNP_REGISTERED, result.poses[THIRD_FRAME].status)
            assertTrue(result.diagnostics[THIRD_FRAME].status.contains("PnP"))
            assertTrue("Occlusion unexpectedly removed all synthetic overlap",
                fixture.visiblePatchIds(THIRD_FRAME).size in 70..90)

            val keypointsByFrame = mapOf(
                FIRST_FRAME to oracle.firstKeypoints,
                SECOND_FRAME to oracle.secondKeypoints,
                THIRD_FRAME to detectKeypoints(fixture.frames[THIRD_FRAME])
            )
            val threeViewTracks = result.featureTracks.filter { track ->
                listOf(FIRST_FRAME, SECOND_FRAME, THIRD_FRAME).all { frameIndex ->
                    track.observations.any { it.frameIndex == frameIndex }
                }
            }
            assertTrue("Only ${threeViewTracks.size} tracks survived all three perturbed views",
                threeViewTracks.size >= 30)
            threeViewTracks.forEach { track ->
                val observations = track.observations.associateBy { it.frameIndex }
                assertEquals(listOf(FIRST_FRAME, SECOND_FRAME, THIRD_FRAME), observations.keys.sorted())
                val patchIds = observations.map { (frameIndex, observation) ->
                    val keypoint = keypointsByFrame.getValue(frameIndex).getOrNull(observation.featureIndex)
                    assertNotNull("Invalid feature index in frame $frameIndex", keypoint)
                    val point = requireNotNull(keypoint).pt
                    assertEquals(point.x, observation.x, 1e-9)
                    assertEquals(point.y, observation.y, 1e-9)
                    fixture.patchIndex(frameIndex, observation)
                }
                assertTrue("Outlier matching joined different physical patches: $patchIds",
                    patchIds.all { it != null } && patchIds.distinct().size == 1)
                assertTrue("A track continued through an occluded third-view feature",
                    fixture.isPatchVisible(THIRD_FRAME, requireNotNull(patchIds.last())))
            }
            for (frameIndex in listOf(FIRST_FRAME, SECOND_FRAME, THIRD_FRAME)) {
                val indices = threeViewTracks.map { track ->
                    track.observations.single { it.frameIndex == frameIndex }.featureIndex
                }
                assertEquals("Frame $frameIndex feature was assigned to multiple tracks",
                    indices.size, indices.toSet().size)
            }

            val tracksById = result.featureTracks.associateBy { it.id }
            val landmarksById = result.landmarks.associateBy { it.trackId }
            assertEquals(tracksById.keys, landmarksById.keys)
            for ((trackId, track) in tracksById) {
                val landmark = landmarksById.getValue(trackId)
                assertEquals(track.observations.size, landmark.observationCount)
                assertTrue("Track $trackId fell below the 1.5 degree triangulation parallax floor",
                    landmark.triangulationAngleDegrees >= 1.5)
                val point = doubleArrayOf(landmark.position.x, landmark.position.y, landmark.position.z)
                for (observation in track.observations) {
                    val cameraPoint = transform(result.poses[observation.frameIndex], point)
                    assertTrue("Track $trackId is behind frame ${observation.frameIndex}", cameraPoint[2] > 0.0)
                    val dx = CAMERA_VALUES[0] * cameraPoint[0] / cameraPoint[2] + CAMERA_VALUES[2] - observation.x
                    val dy = CAMERA_VALUES[4] * cameraPoint[1] / cameraPoint[2] + CAMERA_VALUES[5] - observation.y
                    assertTrue("Track $trackId reprojection error exceeds 4.25 px", hypot(dx, dy) <= 4.25)
                }
                assertTrue(landmark.meanReprojectionErrorPx <= 4.0)
            }
        } finally {
            fixture.release()
            cameraMatrix.release()
        }
    }

    @Test
    fun severeThirdViewOcclusionIsHeldByExistingRegistrationGates() {
        val fixture = SyntheticSeedScene.render(
            SceneKind.PLANAR_HIGH,
            includeThirdCamera = true,
            thirdViewVisibleFraction = 0.01
        )
        val cameraMatrix = cameraMatrix()
        try {
            val oracle = inspectSeedPair(fixture, cameraMatrix, cameraFocalLengthHasPrior = true)
            val result = solvePublicEngine(fixture, cameraMatrix, cameraFocalLengthHasPrior = true)
            assertConsistentSeedResult(result, oracle)
            assertEquals(1, fixture.visiblePatchIds(THIRD_FRAME).size)
            assertEquals(CameraPoseStatus.HELD_LAST_VALID, result.poses[THIRD_FRAME].status)
            assertTrue(result.featureTracks.none { track ->
                track.observations.any { it.frameIndex == THIRD_FRAME }
            })
            assertFalse(result.diagnostics[THIRD_FRAME].status.contains("PnP registered"))
            assertTrue(result.diagnostics[THIRD_FRAME].status.isNotBlank())
        } finally {
            fixture.release()
            cameraMatrix.release()
        }
    }

    @Test
    fun sourceFaithfulTwoViewAndPnpMasksRejectSeededOutliersAtExistingQualityGates() {
        val pair = syntheticSeedCorrespondences(
            pixelNoiseSigmaPx = 0.8,
            outlierFraction = 0.10
        )
        val pairEstimate = requireNotNull(estimateSyntheticSeedPair(pair)) {
            "Source-faithful estimator did not return a model for the in-range synthetic perturbation"
        }
        assertTrue(pairEstimate.passesInitialPairGates)
        assertTrue(pairEstimate.numInliers >= InitialPairPolicy.initialization().minimumInliers)
        assertEquals(pairEstimate.numInliers, pairEstimate.selectedInlierMask.count { it })
        val selectedOutliers = pair.outlierIndices.count { pairEstimate.selectedInlierMask[it] }
        assertEquals("Selected seed mask retained a deliberately swapped correspondence", 0, selectedOutliers)
        assertTrue(pairEstimate.pose!!.inlierIndices.none { it in pair.outlierIndices })

        val camera = cameraMatrix()
        try {
            val random = Random(0x51A7E5L)
            val worldPoints = (0 until SyntheticSeedScene.FEATURE_COUNT).map { index ->
                val column = index % 12
                val row = index / 12
                val depth = 2.8 + (index % 7) * 0.43
                val pixelX = 300.0 + column * 80.0
                val pixelY = 50.0 + row * 65.0
                Point3((pixelX - 640.0) * depth / 800.0,
                    (pixelY - 360.0) * depth / 800.0, depth)
            }
            val trueImagePoints = worldPoints.map { point ->
                Point(
                    800.0 * (point.x - 2.0 / 3.0) / point.z + 640.0,
                    800.0 * point.y / point.z + 360.0
                )
            }
            val belowMinimumCount = CameraRegistrationPolicy.MINIMUM_CORRESPONDENCES - 1
            val belowMinimumTracks = (0 until belowMinimumCount).map { index ->
                val observed = trueImagePoints[index]
                CameraFeatureTrack(index, listOf(
                    CameraFeatureObservation(THIRD_FRAME, observed.x, observed.y, index)
                ))
            }
            val belowMinimumLandmarks = worldPoints.take(belowMinimumCount).mapIndexed { index, point ->
                CameraLandmark(index, point, observationCount = 1, meanReprojectionErrorPx = 0.0)
            }
            val belowMinimumResult = IncrementalCameraRegistrar.register(
                frameIndex = THIRD_FRAME,
                frameWidth = SyntheticSeedScene.WIDTH,
                frameHeight = SyntheticSeedScene.HEIGHT,
                featureTracks = belowMinimumTracks,
                landmarks = belowMinimumLandmarks,
                cameraMatrix = camera
            )
            assertFalse(belowMinimumResult.accepted)
            assertEquals(belowMinimumCount, belowMinimumResult.correspondenceCount)
            assertTrue(belowMinimumResult.reason.contains("Not enough 2D-3D correspondences"))

            val outlierIndices = shuffledIndices(worldPoints.size, 0xBADC0DEL)
                .take((worldPoints.size * 0.15).roundToInt()).toSet()
            val outlierOrder = outlierIndices.sorted()
            val imagePoints = trueImagePoints.mapIndexed { index, _ ->
                val sourceIndex = if (index in outlierIndices) {
                    outlierOrder[(outlierOrder.indexOf(index) + 1) % outlierOrder.size]
                } else index
                val source = trueImagePoints[sourceIndex]
                Point(
                    source.x + random.nextGaussian() * 1.0,
                    source.y + random.nextGaussian() * 1.0
                )
            }
            val estimate = requireNotNull(AbsolutePoseLORansacEstimator.estimate(
                worldPoints.toTypedArray(), imagePoints.toTypedArray(), camera,
                CameraRegistrationPolicy.RANSAC_REPROJECTION_ERROR_PX,
                randomSeed = 0xC0FFEE
            )) { "Seeded PnP fixture did not produce a robust pose" }
            val pnpMask = estimate.inlierIndices.toSet()
            assertTrue("PnP consensus did not meet the 30-correspondence registration minimum",
                pnpMask.size >= CameraRegistrationPolicy.MINIMUM_CORRESPONDENCES)
            assertTrue("PnP consensus fell below the 25% registration inlier ratio",
                pnpMask.size.toDouble() / worldPoints.size >= CameraRegistrationPolicy.MINIMUM_INLIER_RATIO)
            assertEquals("PnP consensus retained a deliberately swapped 2D-3D association",
                0, outlierIndices.count { it in pnpMask })
            val positiveDepth = estimate.inlierIndices.count { index ->
                val point = worldPoints[index]
                estimate.rotation[6] * point.x + estimate.rotation[7] * point.y +
                    estimate.rotation[8] * point.z + estimate.translation[2] > 1e-6
            }
            assertTrue("PnP consensus failed the 90% positive-depth registration gate",
                positiveDepth.toDouble() / pnpMask.size >= CameraRegistrationPolicy.MINIMUM_CHEIRALITY_RATIO)
            val reprojectionErrors = estimate.inlierIndices.map { index ->
                val point = worldPoints[index]
                val observed = imagePoints[index]
                val r = estimate.rotation
                val t = estimate.translation
                val x = r[0] * point.x + r[1] * point.y + r[2] * point.z + t[0]
                val y = r[3] * point.x + r[4] * point.y + r[5] * point.z + t[1]
                val z = r[6] * point.x + r[7] * point.y + r[8] * point.z + t[2]
                hypot(800.0 * x / z + 640.0 - observed.x, 800.0 * y / z + 360.0 - observed.y)
            }
            val sortedErrors = reprojectionErrors.sorted()
            val medianError = sortedErrors[sortedErrors.size / 2]
            val rmsError = sqrt(reprojectionErrors.sumOf { it * it } / reprojectionErrors.size)
            assertTrue("PnP median error exceeds the 4 px final registration gate", medianError <=
                CameraRegistrationPolicy.FINAL_REPROJECTION_ERROR_PX)
            assertTrue("PnP RMS error exceeds 1.5x the final registration limit", rmsError <=
                CameraRegistrationPolicy.FINAL_REPROJECTION_ERROR_PX * 1.5)
            val occupiedCells = estimate.inlierIndices.map { index ->
                val observed = imagePoints[index]
                val column = ((observed.x / SyntheticSeedScene.WIDTH) * 4.0).toInt().coerceIn(0, 3)
                val row = ((observed.y / SyntheticSeedScene.HEIGHT) * 4.0).toInt().coerceIn(0, 3)
                row * 4 + column
            }.toSet().size
            assertTrue("PnP inliers fail the four-cell coverage gate",
                occupiedCells >= CameraRegistrationPolicy.MINIMUM_OCCUPIED_GRID_CELLS)
        } finally {
            camera.release()
        }
    }

    @Test
    fun seedGateRejectsExcessivePixelNoiseAndInsufficientVisibleCorrespondences() {
        val noisyPair = syntheticSeedCorrespondences(pixelNoiseSigmaPx = 6.0)
        val noisyEstimate = estimateSyntheticSeedPair(noisyPair)
        assertTrue("Excessive pixel noise unexpectedly passed the 4 px/100-inlier seed gates",
            noisyEstimate == null || !noisyEstimate.passesInitialPairGates ||
                noisyEstimate.numInliers < InitialPairPolicy.initialization().minimumInliers)

        val lowOverlapPair = syntheticSeedCorrespondences(visibleFraction = 0.75)
        assertEquals(90, lowOverlapPair.firstPixelsXY.size / 2)
        val lowOverlapEstimate = requireNotNull(estimateSyntheticSeedPair(lowOverlapPair))
        assertTrue("A 90-correspondence seed passed the existing 100-inlier initialization minimum",
            lowOverlapEstimate.numInliers < InitialPairPolicy.initialization().minimumInliers &&
                !lowOverlapEstimate.passesInitialPairGates)
    }

    private data class SyntheticSeedCorrespondences(
        val firstPixelsXY: DoubleArray,
        val secondPixelsXY: DoubleArray,
        val outlierIndices: Set<Int>
    )

    private fun syntheticSeedCorrespondences(
        pixelNoiseSigmaPx: Double = 0.0,
        outlierFraction: Double = 0.0,
        visibleFraction: Double = 1.0
    ): SyntheticSeedCorrespondences {
        require(pixelNoiseSigmaPx.isFinite() && pixelNoiseSigmaPx >= 0.0)
        require(outlierFraction in 0.0..1.0 && visibleFraction in 0.0..1.0)
        val allIndices = (0 until 120).toList()
        val retained = shuffledIndices(allIndices.size, 0xA11CE).take(
            (allIndices.size * visibleFraction).roundToInt()
        ).sorted()
        val outlierCount = (retained.size * outlierFraction).roundToInt()
        val localOutliers = shuffledIndices(retained.size, 0x0BADC0DE)
            .take(outlierCount).toSet()
        val outlierOrder = localOutliers.sorted()
        val firstPixels = DoubleArray(retained.size * 2)
        val secondPixels = DoubleArray(retained.size * 2)
        retained.forEachIndexed { localIndex, featureId ->
            val column = featureId % 12
            val row = featureId / 12
            val pixelX = 300.0 + column * 80.0
            val pixelY = 50.0 + row * 65.0
            val sourceFeatureId = if (localIndex in localOutliers && outlierOrder.size > 1) {
                retained[outlierOrder[(outlierOrder.indexOf(localIndex) + 1) % outlierOrder.size]]
            } else featureId
            val sourceColumn = sourceFeatureId % 12
            val sourceRow = sourceFeatureId / 12
            val sourcePixelX = 300.0 + sourceColumn * 80.0
            val sourcePixelY = 50.0 + sourceRow * 65.0
            val firstNoise = Random(0x1F1F1L + featureId * 104729L)
            val secondNoise = Random(0x2E2E2L + localIndex * 130363L)
            val offset = localIndex * 2
            firstPixels[offset] = pixelX + firstNoise.nextGaussian() * pixelNoiseSigmaPx
            firstPixels[offset + 1] = pixelY + firstNoise.nextGaussian() * pixelNoiseSigmaPx
            secondPixels[offset] = sourcePixelX - 800.0 * 1.5 / 4.5 +
                secondNoise.nextGaussian() * pixelNoiseSigmaPx
            secondPixels[offset + 1] = sourcePixelY +
                secondNoise.nextGaussian() * pixelNoiseSigmaPx
        }
        return SyntheticSeedCorrespondences(firstPixels, secondPixels, localOutliers)
    }

    private fun estimateSyntheticSeedPair(
        fixture: SyntheticSeedCorrespondences
    ): SourceFaithfulTwoViewEstimator.InitialPairEstimate? {
        val policy = InitialPairPolicy.initialization()
        return SourceFaithfulTwoViewEstimator.estimateInitialPair(
            firstPixelsXY = fixture.firstPixelsXY,
            secondPixelsXY = fixture.secondPixelsXY,
            cameraMatrixRowMajor = CAMERA_VALUES,
            bothCamerasHaveFocalPriors = true,
            maxErrorPixels = RANSAC_THRESHOLD_PX,
            minimumInliers = policy.minimumInliers,
            minimumEssentialToFundamentalRatio = policy.minimumEssentialToFundamentalInlierRatio,
            maximumHomographyRatio = policy.planarDominanceRatio
        )
    }

    private fun shuffledIndices(size: Int, seed: Long): List<Int> =
        (0 until size).toMutableList().apply {
            java.util.Collections.shuffle(this, Random(seed))
        }

    private fun inspectSeedPair(
        fixture: SyntheticSeedScene.Fixture,
        cameraMatrix: Mat,
        cameraFocalLengthHasPrior: Boolean = false
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
                    bothCamerasHaveFocalPriors = cameraFocalLengthHasPrior,
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
                cameraFocalLengthHasPrior = cameraFocalLengthHasPrior,
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
        cameraMatrix: Mat,
        cameraFocalLengthHasPrior: Boolean = false
    ): CameraTrackingResult = runBlocking {
        CameraTrackingEngine(
            maxFeatures = MAX_FEATURES,
            ratioThreshold = RATIO_THRESHOLD,
            cameraFocalLengthHasPrior = cameraFocalLengthHasPrior
        ).solve(
            frameCount = fixture.frames.size,
            cameraMatrix = cameraMatrix,
            frameProvider = { index -> fixture.frames[index].clone() },
            onProgress = {}
        )
    }

    private fun assertConsistentSeedResult(
        result: CameraTrackingResult,
        oracle: PairOracle,
        expectedRegisteredCameraCount: Int = 2
    ) {
        assertEquals(FIRST_FRAME to SECOND_FRAME, result.initialPairFrames)
        assertEquals(expectedRegisteredCameraCount, result.registeredCameraCount)
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

    private fun detectKeypoints(frame: Mat): Array<KeyPoint> {
        assertTrue("OpenCV runtime is unavailable", OpenCvRuntime.ensureLoaded())
        val extractor = AndroidSiftFeatureExtractor(MAX_FEATURES)
        val keypointMat = MatOfKeyPoint()
        val descriptors = Mat()
        val emptyMask = Mat()
        try {
            extractor.detectAndCompute(frame, emptyMask, keypointMat, descriptors)
            return keypointMat.toArray()
        } finally {
            extractor.clear()
            keypointMat.release()
            descriptors.release()
            emptyMask.release()
        }
    }

    private fun cameraCenter(pose: CameraPose): DoubleArray {
        val rotation = pose.rotation
        val translation = pose.translation
        return doubleArrayOf(
            -(rotation[0] * translation[0] + rotation[3] * translation[1] + rotation[6] * translation[2]),
            -(rotation[1] * translation[0] + rotation[4] * translation[1] + rotation[7] * translation[2]),
            -(rotation[2] * translation[0] + rotation[5] * translation[1] + rotation[8] * translation[2])
        )
    }

    private fun transform(pose: CameraPose, point: DoubleArray): DoubleArray {
        val rotation = pose.rotation
        val translation = pose.translation
        return doubleArrayOf(
            rotation[0] * point[0] + rotation[1] * point[1] + rotation[2] * point[2] + translation[0],
            rotation[3] * point[0] + rotation[4] * point[1] + rotation[5] * point[2] + translation[1],
            rotation[6] * point[0] + rotation[7] * point[1] + rotation[8] * point[2] + translation[2]
        )
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
        private const val FIRST_FRAME = 0
        private const val SECOND_FRAME = 2
        const val THIRD_FRAME = 3

        private const val PATCH_SIZE = 32
        private const val FX = 800.0
        private const val FY = 800.0
        private const val CX = 640.0
        private const val CY = 360.0
        const val HIGH_BASELINE = 1.5
        const val THIRD_CAMERA_CENTER_X = 1.0
        private const val LOW_BASELINE = 0.65
        private const val PLANE_DEPTH = 4.5
        private const val PLANAR_FEATURE_COUNT = 100
        const val FEATURE_COUNT = 120

        data class Fixture(
            val frames: List<Mat>,
            val patchCentersFirst: List<Point>,
            val patchCentersSecond: List<Point>,
            val patchCentersThird: List<Point>,
            private val visiblePatchIdsByFrame: Map<Int, Set<Int>>,
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

            fun patchIndex(frameIndex: Int, observation: CameraFeatureObservation): Int? {
                val centers = when (frameIndex) {
                    FIRST_FRAME -> patchCentersFirst
                    SECOND_FRAME -> patchCentersSecond
                    THIRD_FRAME -> patchCentersThird
                    else -> return null
                }
                return nearestPatch(observation.x, observation.y, centers)
            }

            fun visiblePatchIds(frameIndex: Int): Set<Int> =
                visiblePatchIdsByFrame[frameIndex].orEmpty()

            fun isPatchVisible(frameIndex: Int, patchId: Int): Boolean =
                patchId in visiblePatchIds(frameIndex)

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

        fun render(
            kind: SceneKind,
            includeThirdCamera: Boolean = false,
            thirdCameraHasTexture: Boolean = true,
            pixelNoiseSigmaPx: Double = 0.0,
            secondViewOutlierFraction: Double = 0.0,
            thirdViewOutlierFraction: Double = 0.0,
            secondViewVisibleFraction: Double = 1.0,
            thirdViewVisibleFraction: Double = 1.0
        ): Fixture {
            check(OpenCvRuntime.ensureLoaded()) { "OpenCV could not be loaded in the emulator" }
            require(pixelNoiseSigmaPx.isFinite() && pixelNoiseSigmaPx in 0.0..8.0)
            require(secondViewOutlierFraction in 0.0..1.0 && thirdViewOutlierFraction in 0.0..1.0)
            require(secondViewVisibleFraction in 0.0..1.0 && thirdViewVisibleFraction in 0.0..1.0)
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
            val firstCenters = jitteredCenters(patches, 0.0, FIRST_FRAME, pixelNoiseSigmaPx)
            val secondCenters = jitteredCenters(patches, baseline, SECOND_FRAME, pixelNoiseSigmaPx)
            val blank = blankView()
            val secondVisiblePatchIds = visiblePatchSubset(patches.size, secondViewVisibleFraction, 0x2B02L)
            val thirdVisiblePatchIds = if (includeThirdCamera && thirdCameraHasTexture) {
                visiblePatchSubset(patches.size, thirdViewVisibleFraction, 0x3B03L)
            } else emptySet()
            val first = renderView(patches, firstCenters)
            val second = renderView(
                patches,
                secondCenters,
                textureSourceByPatch = texturePermutation(patches.size, secondViewOutlierFraction, 0x2A02L),
                visiblePatchIds = secondVisiblePatchIds
            )
            val third = if (includeThirdCamera) {
                if (thirdCameraHasTexture) {
                    val thirdCenters = jitteredCenters(
                        patches, THIRD_CAMERA_CENTER_X, THIRD_FRAME, pixelNoiseSigmaPx
                    )
                    renderView(
                        patches,
                        thirdCenters,
                        textureSourceByPatch = texturePermutation(
                            patches.size, thirdViewOutlierFraction, 0x3A03L
                        ),
                        visiblePatchIds = thirdVisiblePatchIds
                    )
                }
                else blankView()
            } else {
                null
            }
            val thirdCenters = if (third != null && thirdCameraHasTexture) {
                jitteredCenters(patches, THIRD_CAMERA_CENTER_X, THIRD_FRAME, pixelNoiseSigmaPx)
            } else {
                emptyList()
            }
            return Fixture(
                frames = listOfNotNull(first, blank, second, third),
                patchCentersFirst = firstCenters,
                patchCentersSecond = secondCenters,
                patchCentersThird = thirdCenters,
                visiblePatchIdsByFrame = mapOf(
                    FIRST_FRAME to patches.indices.toSet(),
                    SECOND_FRAME to secondVisiblePatchIds,
                    THIRD_FRAME to thirdVisiblePatchIds
                ),
                nonPlanarFeatureStart = if (kind == SceneKind.MIXED_HIGH) PLANAR_FEATURE_COUNT else FEATURE_COUNT
            )
        }

        private fun jitteredCenters(
            patches: List<Patch>,
            cameraCenterX: Double,
            viewId: Int,
            pixelNoiseSigmaPx: Double
        ): List<Point> = patches.map { patch ->
            val ideal = project(patch, cameraCenterX)
            if (pixelNoiseSigmaPx == 0.0) return@map ideal
            val random = Random(0x5EEDC0DEL + patch.id * 7919L + viewId * 104729L)
            val maximumOffset = pixelNoiseSigmaPx * 3.0
            val dx = (random.nextGaussian() * pixelNoiseSigmaPx)
                .coerceIn(-maximumOffset, maximumOffset).roundToInt()
            val dy = (random.nextGaussian() * pixelNoiseSigmaPx)
                .coerceIn(-maximumOffset, maximumOffset).roundToInt()
            Point(ideal.x + dx, ideal.y + dy)
        }

        private fun visiblePatchSubset(
            patchCount: Int,
            visibleFraction: Double,
            seed: Long
        ): Set<Int> = shuffledIndices(patchCount, seed)
            .take((patchCount * visibleFraction).roundToInt())
            .toSet()

        private fun texturePermutation(
            patchCount: Int,
            outlierFraction: Double,
            seed: Long
        ): IntArray {
            val permutation = IntArray(patchCount) { it }
            val swapCount = ((patchCount * outlierFraction).roundToInt().coerceAtMost(patchCount)) / 2 * 2
            val selected = shuffledIndices(patchCount, seed).take(swapCount)
            selected.chunked(2).forEach { pair ->
                if (pair.size == 2) {
                    permutation[pair[0]] = pair[1]
                    permutation[pair[1]] = pair[0]
                }
            }
            return permutation
        }

        private fun shuffledIndices(size: Int, seed: Long): List<Int> =
            (0 until size).toMutableList().apply {
                java.util.Collections.shuffle(this, Random(seed))
            }

        private fun blankView(): Mat {
            val pixels = ByteArray(WIDTH * HEIGHT) { 127.toByte() }
            return Mat(HEIGHT, WIDTH, CvType.CV_8UC1).also { image ->
                check(image.put(0, 0, pixels) == pixels.size)
            }
        }

        private data class Patch(
            val id: Int,
            val worldX: Double,
            val worldY: Double,
            val depth: Double,
            val pixels: ByteArray
        )

        private fun renderView(
            patches: List<Patch>,
            centers: List<Point>,
            textureSourceByPatch: IntArray = IntArray(patches.size) { it },
            visiblePatchIds: Set<Int> = patches.indices.toSet()
        ): Mat {
            val image = ByteArray(WIDTH * HEIGHT) { 127.toByte() }
            val occupied = mutableListOf<Box>()
            for (patch in patches) {
                if (patch.id !in visiblePatchIds) continue
                val center = centers[patch.id]
                val left = center.x.roundToInt() - PATCH_SIZE / 2
                val top = center.y.roundToInt() - PATCH_SIZE / 2
                val box = Box(left, top, left + PATCH_SIZE, top + PATCH_SIZE)
                check(box.left >= 0 && box.top >= 0 && box.right <= WIDTH && box.bottom <= HEIGHT) {
                    "Synthetic patch ${patch.id} is clipped at center $center"
                }
                check(occupied.none(box::overlaps)) {
                    "Synthetic patches overlap at center $center"
                }
                occupied += box
                val appearance = patches[textureSourceByPatch[patch.id]]
                for (y in 0 until PATCH_SIZE) {
                    val target = (top + y) * WIDTH + left
                    System.arraycopy(appearance.pixels, y * PATCH_SIZE, image, target, PATCH_SIZE)
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
        const val THIRD_FRAME = 3
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
