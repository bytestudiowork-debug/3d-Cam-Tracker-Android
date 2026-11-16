package com.trackstudio.app.tracking

import com.trackstudio.app.tracking.colmap.ColmapTrackCompleter
import com.trackstudio.app.tracking.glomap.GlomapGlobalPositioningInitializer
import com.trackstudio.app.tracking.glomap.GlomapGlobalPositioningProblemBuilder
import com.trackstudio.app.tracking.glomap.GlomapGlobalBundleAdjuster
import com.trackstudio.app.tracking.glomap.GlomapNativeSolver
import com.trackstudio.app.tracking.glomap.GlomapTrackFilter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.CvType
import org.opencv.core.Point3
import org.opencv.core.MatOfPoint2f
import org.opencv.core.KeyPoint
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import kotlin.math.sqrt
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * Independent global reconstruction path.
 *
 * Unlike IncrementalSfMSolver, this engine does not consume an incremental
 * camera map, seed poses, or landmarks. It extracts features for the complete
 * sequence, verifies a multi-view image-pair graph, estimates all camera
 * rotations jointly, establishes multi-view tracks, and solves camera centres
 * and points together with the global direction-constraint positioner.
 *
 * The final sparse points are retriangulated and robustly refined with the
 * mobile landmark block optimizer. Joint bundle adjustment refines poses and
 * focal lengths, and releases principal point only with broad coverage,
 * translation parallax, and multi-axis rotation; skew and metric scale remain fixed.
 */
class GlobalReconstructionEngine(
    private val maxFeatures: Int = 1400,
    private val ratioThreshold: Float = 0.8f,
    private val maximumPairCount: Int = 420
) : ReconstructionEngine {
    init {
        require(maxFeatures in 500..4000)
        require(ratioThreshold.isFinite() && ratioThreshold > 0f && ratioThreshold < 1f)
        require(maximumPairCount > 0)
    }

    internal data class FrameFeatures(
        val keypoints: Array<KeyPoint>,
        val descriptors: Mat,
        val width: Int,
        val height: Int
    )

    private data class VerifiedPair(
        val first: Int,
        val second: Int,
        val rotation: DoubleArray,
        val translationDirection: DoubleArray,
        val inlierMatches: List<Pair<Int, Int>>,
        val matchCount: Int
    )

    override suspend fun solve(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit
    ): CameraTrackingResult = withContext(Dispatchers.Default) {
        require(frameCount >= 3) { "Global reconstruction needs at least three frames." }
        require(!cameraMatrix.empty() && cameraMatrix.rows() == 3 && cameraMatrix.cols() == 3) {
            "A valid 3x3 camera intrinsic matrix is required."
        }
        check(OpenCvRuntime.ensureLoaded()) { "OpenCV could not be initialized on this device." }

        // Bound per-frame descriptor memory on long clips while retaining a
        // sufficiently rich feature set for geometric verification.
        val heapMb = Runtime.getRuntime().maxMemory() / (1024L * 1024L)
        val featureBudget = when {
            frameCount >= 500 -> if (heapMb <= 256L) 800 else 1000
            frameCount >= 250 -> if (heapMb <= 256L) 950 else 1200
            else -> maxFeatures.coerceIn(500, 4000)
        }
        val featureExtractor = AndroidSiftFeatureExtractor(featureBudget)
        val frames = arrayOfNulls<FrameFeatures>(frameCount)
        var refinedCameraMatrix: Mat? = null
        var decodedFrames = 0
        val solveContext = currentCoroutineContext()
        fun publishProgress(phase: String, fraction: Float) {
            // Native optimizer iteration callbacks are synchronous. Bridge them
            // to the suspend progress handler so Pause really stops at the next
            // optimizer iteration instead of merely updating the UI.
            runBlocking {
                solveContext.ensureActive()
                onProgress(CameraTrackingProgress(frameCount, frameCount, phase, fraction, decodedFrames, frameCount))
            }
        }
        try {
            // Global reconstruction starts from the entire sequence, not an
            // initialization pair selected by the Custom engine.
            for (frameIndex in 0 until frameCount) {
                currentCoroutineContext().ensureActive()
                val source = frameProvider(frameIndex)
                    ?: throw IllegalStateException("Unable to decode frame ${frameIndex + 1}.")
                try {
                    val gray = toGray(source)
                    val keypoints = org.opencv.core.MatOfKeyPoint()
                    val descriptors = Mat()
                    try {
                        val mask = Mat()
                        try {
                            featureExtractor.detectAndCompute(gray, mask, keypoints, descriptors)
                        } finally {
                            mask.release()
                        }
                        frames[frameIndex] = FrameFeatures(
                            keypoints.toArray(), descriptors, source.cols(), source.rows()
                        )
                    } catch (failure: Throwable) {
                        descriptors.release()
                        throw failure
                    } finally {
                        gray.release()
                        keypoints.release()
                    }
                } finally {
                    source.release()
                }
                decodedFrames++
                onProgress(
                    CameraTrackingProgress(
                        frameIndex = frameIndex + 1,
                        frameCount = frameCount,
                        phase = "Global feature extraction",
                        overallFraction = 0.25f * decodedFrames / frameCount,
                        completedFrames = decodedFrames,
                        totalFrames = frameCount
                    )
                )
            }

            val frameSizes = frames.filterNotNull().map { it.width to it.height }.distinct()
            require(frameSizes.size == 1 && frameSizes.single().first > 0 && frameSizes.single().second > 0) {
                "All frames in a reconstruction must use the same nonzero image dimensions."
            }

            // Long clips need both a connected temporal backbone and wider
            // baselines. The default 420-pair cap must not collapse a 588-frame
            // sequence into adjacent-only comparisons.
            val candidatePairs = buildPairSchedule(frameCount, maxOf(maximumPairCount, frameCount * 2))
            val completedPairs = AtomicInteger(0)
            val progressMutex = Mutex()
            val workerCount = minOf(Runtime.getRuntime().availableProcessors().coerceAtLeast(1), 2, candidatePairs.size.coerceAtLeast(1))
            val chunks = candidatePairs.chunked((candidatePairs.size + workerCount - 1) / workerCount)
            val verifiedPairs = coroutineScope {
                chunks.map { chunk ->
                    async(Dispatchers.Default) {
                        val workerMatcher = SiftDescriptorMatcher(ratioThreshold)
                        try {
                            chunk.mapNotNull { pair ->
                                currentCoroutineContext().ensureActive()
                                val first = frames[pair.first] ?: return@mapNotNull null
                                val second = frames[pair.second] ?: return@mapNotNull null
                                val verified = verifyPair(pair.first, pair.second, first, second, cameraMatrix, workerMatcher)
                                val completed = completedPairs.incrementAndGet()
                                if (completed % 2 == 0 || completed == candidatePairs.size) {
                                    progressMutex.withLock {
                                        val current = completedPairs.get()
                                        onProgress(CameraTrackingProgress(
                                            frameCount, frameCount,
                                            "Verifying image pairs " + current + "/" + candidatePairs.size,
                                            0.25f + 0.30f * current / candidatePairs.size.coerceAtLeast(1),
                                            decodedFrames, frameCount
                                        ))
                                    }
                                }
                                verified
                            }
                        } finally {
                            workerMatcher.clear()
                        }
                    }
                }.awaitAll().flatten()
            }
            // Descriptor Mats are no longer needed after pair verification.
            // Release them before constructing the multi-view graph to lower
            // peak native memory on long sequences.
            frames.filterNotNull().forEach { it.descriptors.release() }
            val totalMatches = verifiedPairs.sumOf { it.matchCount }
            val totalInliers = verifiedPairs.sumOf { it.inlierMatches.size }
            if (verifiedPairs.isEmpty()) {
                throw IllegalStateException("No image pairs passed geometric verification. Use textured footage with visible camera translation.")
            }

            val rotationEdges = verifiedPairs.map {
                GlobalRotationAverager.Edge(
                    firstFrame = it.first,
                    secondFrame = it.second,
                    relativeRotation = it.rotation,
                    inlierCount = it.inlierMatches.size
                )
            }
            val firstRotationPass = GlobalRotationAverager.solve(
                frameIndices = verifiedPairs.flatMap { listOf(it.first, it.second) }.distinct(),
                edges = rotationEdges
            ) ?: throw IllegalStateException("The verified view graph is disconnected or has insufficient rotation constraints.")
            val rotationPass = GlobalRotationAverager.solve(
                frameIndices = firstRotationPass.rotationsByFrame.keys.toList(),
                edges = firstRotationPass.acceptedEdges
            ) ?: throw IllegalStateException("Global rotation refinement failed.")
            val rotations = rotationPass.rotationsByFrame
            val activeFrames = rotations.keys.sorted()
            val acceptedPairKeys = rotationPass.acceptedEdges.mapTo(HashSet()) {
                (it.firstFrame.toLong() shl 32) xor (it.secondFrame.toLong() and 0xffffffffL)
            }
            val activePairs = verifiedPairs.filter {
                it.first in rotations && it.second in rotations &&
                    ((it.first.toLong() shl 32) xor (it.second.toLong() and 0xffffffffL)) in acceptedPairKeys
            }
            if (activeFrames.size < 3 || activePairs.isEmpty()) {
                throw IllegalStateException("The global view graph contains fewer than three connected cameras.")
            }

            onProgress(
                CameraTrackingProgress(
                    frameIndex = frameCount,
                    frameCount = frameCount,
                    phase = "Establishing global feature tracks",
                    overallFraction = 0.62f,
                    completedFrames = decodedFrames,
                    totalFrames = frameCount
                )
            )
            val trackBuilder = GlobalTrackGraphBuilder()
            activePairs.forEach { pair ->
                trackBuilder.addVerifiedMatches(pair.first, pair.second, pair.inlierMatches)
            }
            val tracks = trackBuilder.build(frames, rotations.keys)
            if (tracks.size < 20) {
                throw IllegalStateException("Global reconstruction found only ${tracks.size} multi-view tracks; at least 20 are needed for a stable solve.")
            }

            val images = activeFrames.map { frame ->
                GlomapGlobalPositioningInitializer.ImageInput(
                    imageId = frame,
                    frameId = frame,
                    isRegistered = true
                )
            }
            val initializer = GlomapGlobalPositioningInitializer.initialize(
                frames = activeFrames.map { frame ->
                    GlomapGlobalPositioningInitializer.FrameInput(
                        frameId = frame,
                        cameraCenterFromPose = doubleArrayOf(0.0, 0.0, 0.0),
                        existingTranslation = doubleArrayOf(0.0, 0.0, 0.0)
                    )
                },
                images = images,
                imagePairs = activePairs.map {
                    GlomapGlobalPositioningInitializer.ImagePairInput(
                        firstImageId = it.first,
                        secondImageId = it.second,
                        isValid = true
                    )
                },
                tracks = tracks.map { track ->
                    GlomapGlobalPositioningInitializer.TrackInput(
                        trackId = track.id,
                        observationImageIds = track.observations.map { it.frameIndex }.distinct()
                    )
                }
            )

            val fx = cameraMatrix.get(0, 0)[0]
            val skew = cameraMatrix.get(0, 1)[0]
            val cx = cameraMatrix.get(0, 2)[0]
            val fy = cameraMatrix.get(1, 1)[0]
            val cy = cameraMatrix.get(1, 2)[0]
            require(fx.isFinite() && fy.isFinite() && fx > 0.0 && fy > 0.0) {
                "Camera intrinsics contain invalid focal lengths."
            }
            val initialIntrinsics = CameraIntrinsics(
                fx = fx, skew = skew, cx = cx, fy = fy, cy = cy
            )

            val positionTracks = tracks.map { track ->
                val observations = track.observations.mapNotNull { observation ->
                    val rotation = rotations[observation.frameIndex] ?: return@mapNotNull null
                    val center = initializer.frameTranslationsById[observation.frameIndex]
                        ?: return@mapNotNull null
                    val normalizedY = (observation.y - cy) / fy
                    val normalizedX = (observation.x - cx - skew * normalizedY) / fx
                    GlomapGlobalPositioningProblemBuilder.CameraObservation(
                        frameIndex = observation.frameIndex,
                        rotation = rotation,
                        cameraCenter = center,
                        undistortedFeatureDirection = doubleArrayOf(normalizedX, normalizedY, 1.0),
                        isCalibrated = true
                    )
                }
                GlomapGlobalPositioningProblemBuilder.Track(
                    trackId = track.id,
                    observations = observations,
                    initializedPoint = initializer.pointsByTrack[track.id]
                )
            }
            val positioningProblem = GlomapGlobalPositioningProblemBuilder.build(positionTracks)
            if (positioningProblem.constraints.size < 60) {
                throw IllegalStateException("Global positioning has too few valid track observations (${positioningProblem.constraints.size}).")
            }

            onProgress(
                CameraTrackingProgress(
                    frameIndex = frameCount,
                    frameCount = frameCount,
                    phase = "Solving global camera positions and 3D points",
                    overallFraction = 0.78f,
                    completedFrames = decodedFrames,
                    totalFrames = frameCount
                )
            )
            val positionResult = try {
                GlomapNativeSolver().solve(
                    problem = positioningProblem,
                    frameCentersById = initializer.frameTranslationsById,
                    pointsByTrackId = initializer.pointsByTrack.mapNotNull { (id, point) ->
                        point?.let { id to it }
                    }.toMap(),
                    maxIterations = 100,
                    numThreads = if (activeFrames.size >= 300) 2 else Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
                    onIteration = { iteration, total -> publishProgress("Global position solve " + iteration + "/" + total, 0.78f + 0.10f * iteration.toFloat() / total.coerceAtLeast(1)) }
                )
            } catch (failure: LinkageError) {
                throw IllegalStateException("The Android global-positioning library could not be loaded.", failure)
            } ?: throw IllegalStateException("The global camera-position and point solve did not converge to a usable solution.")
            if (!SolverObjective.didNotIncrease(positionResult.initialCost, positionResult.finalCost)) {
                throw IllegalStateException("The global camera-position solve increased its objective; no result was published.")
            }
            var reconstructionTracks = GlomapTrackFilter.filterAfterPositioning(
                tracks = tracks,
                cameraCentersByFrame = positionResult.frameCentersById,
                rotationsByFrame = rotations,
                pointsByTrackId = positionResult.pointsByTrackId,
                intrinsics = initialIntrinsics,
                hasPriorFocalLength = true
            )

            onProgress(
                CameraTrackingProgress(
                    frameIndex = frameCount,
                    frameCount = frameCount,
                    phase = "Joint global bundle adjustment",
                    overallFraction = 0.88f,
                    completedFrames = decodedFrames,
                    totalFrames = frameCount
                )
            )
            val bundleCandidate = try {
                GlomapGlobalBundleAdjuster().solve(
                    frameCentersById = positionResult.frameCentersById,
                    rotationsByFrame = rotations,
                    pointsByTrackId = positionResult.pointsByTrackId,
                    tracks = reconstructionTracks,
                    cameraMatrix = cameraMatrix,
                    imageWidth = frameSizes.single().first,
                    imageHeight = frameSizes.single().second,
                    maxIterations = 45,
                    numThreads = if (activeFrames.size >= 300) 2 else Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
                    onIteration = { iteration, total -> publishProgress("Global bundle adjustment " + iteration + "/" + total, 0.88f + 0.05f * iteration.toFloat() / total.coerceAtLeast(1)) }
                )
            } catch (_: LinkageError) {
                null
            }
            val bundleResult = bundleCandidate?.takeIf {
                SolverObjective.didNotIncrease(it.initialCost, it.finalCost)
            }
            val bundleStatus = when {
                bundleCandidate == null -> "Unavailable; retained global-positioning result"
                bundleResult == null -> "Objective increased; retained global-positioning result"
                else -> "Applied"
            }
            val finalCenters = bundleResult?.frameCentersById ?: positionResult.frameCentersById
            val finalRotations = bundleResult?.rotationsByFrame ?: rotations
            val finalPoints = bundleResult?.pointsByTrackId ?: positionResult.pointsByTrackId
            val refinedIntrinsics = bundleResult?.intrinsics
            if (bundleResult != null) {
                reconstructionTracks = GlomapTrackFilter.filterAfterBundleAdjustment(
                    tracks = reconstructionTracks,
                    cameraCentersByFrame = finalCenters,
                    rotationsByFrame = finalRotations,
                    pointsByTrackId = finalPoints,
                    intrinsics = refinedIntrinsics ?: initialIntrinsics
                )
            }
            val resultCameraMatrix = if (refinedIntrinsics != null) {
                Mat(3, 3, CvType.CV_64F).also { matrix ->
                    refinedCameraMatrix = matrix
                    matrix.put(0, 0, *refinedIntrinsics.toRowMajorMatrix())
                }
            } else cameraMatrix

            val poses = activeFrames.mapNotNull { frame ->
                val rotation = finalRotations[frame] ?: return@mapNotNull null
                val center = finalCenters[frame] ?: return@mapNotNull null
                val translation = doubleArrayOf(
                    -(rotation[0] * center[0] + rotation[1] * center[1] + rotation[2] * center[2]),
                    -(rotation[3] * center[0] + rotation[4] * center[1] + rotation[5] * center[2]),
                    -(rotation[6] * center[0] + rotation[7] * center[1] + rotation[8] * center[2])
                )
                CameraPose(frame, rotation.copyOf(), translation, CameraPoseStatus.RELATIVE_ESTIMATE)
            }

            onProgress(
                CameraTrackingProgress(
                    frameIndex = frameCount,
                    frameCount = frameCount,
                    phase = "Retriangulating and refining global landmarks",
                    overallFraction = 0.94f,
                    completedFrames = decodedFrames,
                    totalFrames = frameCount
                )
            )
            val poseByFrame = poses.associateBy { it.frameIndex }
            val projectionIntrinsics = refinedIntrinsics?.let {
                PinholeIntrinsics(it.fx, it.fy, it.skew, it.cx, it.cy)
            } ?: PinholeIntrinsics(
                fx = resultCameraMatrix.get(0, 0)[0],
                fy = resultCameraMatrix.get(1, 1)[0],
                skew = resultCameraMatrix.get(0, 1)[0],
                cx = resultCameraMatrix.get(0, 2)[0],
                cy = resultCameraMatrix.get(1, 2)[0]
            )
            var triangulationPass = 0
            val triangulationProgress: (Int, Int) -> Unit = { completed, total ->
                val progressStart = if (triangulationPass == 0) 0.94f else 0.955f
                publishProgress(
                    "Triangulating landmarks " + completed + "/" + total,
                    progressStart + 0.015f * completed.toFloat() / total.coerceAtLeast(1)
                )
            }
            var triangulated = SfMTriangulator.triangulate(
                reconstructionTracks, poses, resultCameraMatrix,
                onProgress = triangulationProgress
            )
            val completion = ColmapTrackCompleter.completeTracks(
                tracks = reconstructionTracks,
                landmarks = triangulated.landmarks,
                pairMatches = activePairs.map { pair ->
                    ColmapTrackCompleter.PairMatches(pair.first, pair.second, pair.inlierMatches)
                },
                activeFrames = activeFrames.toSet(),
                poses = poses,
                intrinsics = projectionIntrinsics,
                featureAt = { frame, feature ->
                    val keypoint = frames.getOrNull(frame)?.keypoints?.getOrNull(feature)
                    keypoint?.let {
                        CameraFeatureObservation(frame, it.pt.x, it.pt.y, feature)
                    }
                }
            )
            if (completion.addedObservations > 0) {
                reconstructionTracks = GlomapTrackFilter.filterAfterBundleAdjustment(
                    tracks = completion.tracks,
                    cameraCentersByFrame = finalCenters,
                    rotationsByFrame = finalRotations,
                    pointsByTrackId = finalPoints,
                    intrinsics = refinedIntrinsics ?: initialIntrinsics
                ).tracks
                triangulationPass = 1
                triangulated = SfMTriangulator.triangulate(
                    reconstructionTracks, poses, resultCameraMatrix,
                    onProgress = triangulationProgress
                )
            }
            val bundleLandmarks = reconstructionTracks.mapNotNull trackLoop@ { track ->
                val point = finalPoints[track.id] ?: return@trackLoop null
                val errors = track.observations.mapNotNull observationLoop@ { observation ->
                    val pose = poseByFrame[observation.frameIndex] ?: return@observationLoop null
                    reprojectionError(projectionIntrinsics, pose, point, observation)
                }
                if (errors.size < 2) return@trackLoop null
                CameraLandmark(
                    trackId = track.id,
                    position = Point3(point[0], point[1], point[2]),
                    observationCount = errors.size,
                    meanReprojectionErrorPx = errors.average()
                )
            }
            val mergedLandmarks = linkedMapOf<Int, CameraLandmark>()
            bundleLandmarks.forEach { mergedLandmarks[it.trackId] = it }
            triangulated.landmarks.forEach { candidate ->
                val previous = mergedLandmarks[candidate.trackId]
                if (previous == null ||
                    candidate.meanReprojectionErrorPx < previous.meanReprojectionErrorPx
                ) mergedLandmarks[candidate.trackId] = candidate
            }
            val refinedLandmarks = MobileBundleAdjuster.optimizeLandmarks(
                featureTracks = reconstructionTracks,
                landmarks = mergedLandmarks.values.toList(),
                poses = poses,
                cameraMatrix = resultCameraMatrix,
                onProgress = { completed, total ->
                    publishProgress("Refining landmarks " + completed + "/" + total, 0.96f + 0.03f * completed.toFloat() / total.coerceAtLeast(1))
                }
            )
            publishProgress("Finalizing camera diagnostics", 0.995f)
            // Accumulate counts in one pass. The previous per-frame nested scan
            // was O(frames × tracks × observations), which became a long silent
            // tail on dense 99-frame reconstructions.
            val frameTrackCounts = IntArray(frameCount)
            reconstructionTracks.forEachIndexed { trackIndex, track ->
                if (trackIndex % 512 == 0) {
                    publishProgress(
                        "Finalizing camera diagnostics " + trackIndex + "/" + reconstructionTracks.size,
                        0.995f + 0.004f * trackIndex.toFloat() / reconstructionTracks.size.coerceAtLeast(1)
                    )
                }
                track.observations.asSequence().map { it.frameIndex }.distinct().forEach { frame ->
                    if (frame in frameTrackCounts.indices) frameTrackCounts[frame]++
                }
            }
            val frameMatchedCounts = IntArray(frameCount)
            activePairs.forEach { pair ->
                val inliers = pair.inlierMatches.size
                frameMatchedCounts[pair.first] += inliers
                frameMatchedCounts[pair.second] += inliers
            }
            val activeFrameSet = activeFrames.toSet()
            val diagnostics = (0 until frameCount).map { frame ->
                FrameTrackingDiagnostic(
                    frameIndex = frame,
                    detectedFeatures = frames[frame]?.keypoints?.size ?: 0,
                    matchedFeatures = frameMatchedCounts[frame],
                    inlierFeatures = frameTrackCounts[frame],
                    status = if (frame in activeFrameSet) "Global reconstruction registered"
                    else "Unregistered: outside the solved view-graph component"
                )
            }
            val averageInlierRatio = verifiedPairs.map {
                it.inlierMatches.size.toDouble() / it.matchCount.coerceAtLeast(1)
            }.average().takeIf { it.isFinite() } ?: 0.0
            onProgress(
                CameraTrackingProgress(
                    frameIndex = frameCount,
                    frameCount = frameCount,
                    phase = "Global reconstruction complete",
                    overallFraction = 1f,
                    completedFrames = decodedFrames,
                    totalFrames = frameCount
                )
            )

            CameraTrackingResult(
                poses = poses,
                diagnostics = diagnostics,
                processedFrames = decodedFrames,
                solvedFramePairs = verifiedPairs.size,
                totalMatches = totalMatches,
                totalInliers = totalInliers,
                inlierRatio = averageInlierRatio,
                isCameraMotionEstimated = poses.size >= 3,
                featureTracks = reconstructionTracks,
                landmarks = refinedLandmarks,
                triangulationStats = triangulated.stats,
                initialPairFrames = activePairs.maxByOrNull { it.inlierMatches.size }
                    ?.let { it.first to it.second },
                registeredCameraCount = poses.size,
                globalRotationsByFrame = finalRotations,
                viewGraphEdgeCount = verifiedPairs.size,
                viewGraphAcceptedEdgeCount = rotationPass.acceptedEdges.size,
                globalRotationIterations = firstRotationPass.iterations + rotationPass.iterations,
                globalPositioningApplied = true,
                globalPositioningInitialCost = positionResult.initialCost,
                globalPositioningFinalCost = positionResult.finalCost,
                globalPositioningIterations = positionResult.iterations,
                globalPositioningStatus = "Global camera positions and points solved",
                globalBundleAdjustmentApplied = bundleResult != null,
                globalBundleAdjustmentInitialCost = bundleResult?.initialCost ?: Double.NaN,
                globalBundleAdjustmentFinalCost = bundleResult?.finalCost ?: Double.NaN,
                globalBundleAdjustmentIterations = bundleResult?.iterations ?: 0,
                globalBundleAdjustmentStatus = bundleStatus,
                globalBundleAdjustmentPositiveDepthRatio =
                    bundleResult?.positiveDepthRatio ?: Double.NaN,
                globalBundleAdjustmentPrincipalPointRefinementEnabled =
                    bundleResult?.principalPointRefinementEnabled ?: false,
                jointBundleAdjustmentApplied = bundleResult != null,
                jointBundleAdjustmentInitialCost = bundleResult?.initialCost ?: Double.NaN,
                jointBundleAdjustmentFinalCost = bundleResult?.finalCost ?: Double.NaN,
                jointBundleAdjustmentIterations = bundleResult?.iterations ?: 0,
                jointBundleAdjustmentStatus = bundleStatus,
                jointBundleAdjustmentPositiveDepthRatio =
                    bundleResult?.positiveDepthRatio ?: Double.NaN,
                jointBundleAdjustmentPrincipalPointRefinementEnabled =
                    bundleResult?.principalPointRefinementEnabled ?: false,
                refinedCameraIntrinsics = refinedIntrinsics
            )
        } finally {
            frames.filterNotNull().forEach { it.descriptors.release() }
            refinedCameraMatrix?.release()
            featureExtractor.clear()
        }
    }

    internal fun buildPairSchedule(frameCount: Int, maximumPairs: Int): List<Pair<Int, Int>> {
        if (frameCount < 2) return emptyList()
        val pairs = linkedSetOf<Pair<Int, Int>>()
        for (first in 0 until frameCount - 1) pairs += first to first + 1
        val pairBudget = maxOf(frameCount - 1, minOf(maximumPairs, frameCount * 2))
        val offsets = ArrayList<Int>()
        var offset = 2
        while (offset < frameCount) {
            offsets += offset
            if (offset > Int.MAX_VALUE / 2) break
            offset *= 2
        }

        var remaining = pairBudget - pairs.size
        for ((offsetIndex, baseline) in offsets.withIndex()) {
            if (remaining <= 0) break
            val offsetsLeft = offsets.size - offsetIndex
            val candidates = frameCount - baseline
            val requested = (remaining + offsetsLeft - 1) / offsetsLeft
            val sampleCount = minOf(candidates, requested)
            for (sample in 0 until sampleCount) {
                // Midpoint sampling spreads each baseline across the full clip
                // rather than consuming the budget from the opening frames.
                val first = (((2L * sample + 1L) * candidates) /
                    (2L * sampleCount)).toInt()
                pairs += first to first + baseline
            }
            remaining -= sampleCount
        }
        if (pairs.size < pairBudget) {
            pairLoop@ for (baseline in 2 until frameCount) {
                for (first in 0 until frameCount - baseline) {
                    pairs += first to first + baseline
                    if (pairs.size >= pairBudget) break@pairLoop
                }
            }
        }
        return pairs.toList()
    }

    private fun verifyPair(
        firstIndex: Int,
        secondIndex: Int,
        first: FrameFeatures,
        second: FrameFeatures,
        cameraMatrix: Mat,
        matcher: SiftDescriptorMatcher
    ): VerifiedPair? {
        if (first.descriptors.empty() || second.descriptors.empty() ||
            first.keypoints.isEmpty() || second.keypoints.isEmpty()
        ) return null

        val matched = matcher.match(first.descriptors, second.descriptors)
            .filter { (query, train) ->
                query in first.keypoints.indices && train in second.keypoints.indices
            }
        if (matched.size < 30) return null

        val points1 = MatOfPoint2f()
        val points2 = MatOfPoint2f()
        val essentialMask = Mat()
        val bestMask = Mat()
        val bestRotation = Mat()
        val bestTranslation = Mat()
        val bestEssential = Mat()
        return try {
            points1.fromList(matched.map { first.keypoints[it.first].pt })
            points2.fromList(matched.map { second.keypoints[it.second].pt })
            val essential = Calib3d.findEssentialMat(
                points1, points2, cameraMatrix, Calib3d.RANSAC,
                0.999, 1.0, 900, essentialMask
            )
            try {
                if (essential.empty() || essential.cols() != 3 ||
                    essential.rows() < 3 || essential.rows() % 3 != 0
                ) return null

                var bestInliers = 0
                for (candidateIndex in 0 until essential.rows() / 3) {
                    val candidate = essential.rowRange(candidateIndex * 3, candidateIndex * 3 + 3).clone()
                    val mask = essentialMask.clone()
                    val rotation = Mat()
                    val translation = Mat()
                    try {
                        val inliers = Calib3d.recoverPose(
                            candidate, points1, points2, cameraMatrix,
                            rotation, translation, mask
                        )
                        if (inliers > bestInliers) {
                            bestInliers = inliers
                            mask.copyTo(bestMask)
                            rotation.copyTo(bestRotation)
                            translation.copyTo(bestTranslation)
                            candidate.copyTo(bestEssential)
                        }
                    } finally {
                        candidate.release()
                        mask.release()
                        rotation.release()
                        translation.release()
                    }
                }
                if (bestInliers > 0 && !bestEssential.empty()) {
                    val initialEssential = DoubleArray(9)
                    if (bestEssential.get(0, 0, initialEssential) ==
                        initialEssential.size * java.lang.Double.BYTES
                    ) {
                        val polished = OpenCvTwoViewLocalOptimizer.refine(
                            matched.map { first.keypoints[it.first].pt }.toTypedArray(),
                            matched.map { second.keypoints[it.second].pt }.toTypedArray(),
                            cameraMatrix,
                            initialEssential,
                            OpenCvTwoViewLocalOptimizer.inliersFromMask(bestMask, matched.size),
                            1.0,
                            0.999,
                            900
                        )
                        if (polished != null && polished.inlierIndices.size >= bestInliers) {
                            val polishedMask = OpenCvTwoViewLocalOptimizer.maskFromIndices(
                                polished.inlierIndices, matched.size
                            )
                            val polishedRotation = Mat(3, 3, CvType.CV_64F)
                            val polishedTranslation = Mat(3, 1, CvType.CV_64F)
                            try {
                                if (polishedRotation.put(0, 0, *polished.rotation) == polished.rotation.size &&
                                    polishedTranslation.put(0, 0, *polished.translation) == polished.translation.size
                                ) {
                                    polishedMask.copyTo(bestMask)
                                    polishedRotation.copyTo(bestRotation)
                                    polishedTranslation.copyTo(bestTranslation)
                                    bestInliers = polished.inlierIndices.size
                                }
                            } finally {
                                polishedMask.release()
                                polishedRotation.release()
                                polishedTranslation.release()
                            }
                        }
                    }
                }
                if (bestInliers < 30 || bestRotation.empty() || bestTranslation.empty()) return null
                val rotationValues = OpenCvMatrixReadback.readCv64F(bestRotation, 3, 3) ?: return null
                val translationValues = OpenCvMatrixReadback.readCv64F(bestTranslation, 3, 1) ?: return null
                val translationNorm = sqrt(translationValues.sumOf { it * it })
                if (!translationNorm.isFinite() || translationNorm < 1e-9) return null
                for (i in translationValues.indices) translationValues[i] /= translationNorm

                val inlierMatches = (0 until bestMask.rows()).mapNotNull { index ->
                    val accepted = bestMask.get(index, 0)?.firstOrNull() ?: 0.0
                    if (accepted != 0.0) matched[index] else null
                }
                if (inlierMatches.size != bestInliers || inlierMatches.size < 30 ||
                    inlierMatches.size.toDouble() / matched.size < 0.20
                ) return null
                VerifiedPair(
                    firstIndex, secondIndex, rotationValues,
                    translationValues, inlierMatches, matched.size
                )
            } finally {
                essential.release()
            }
        } finally {
            points1.release()
            points2.release()
            essentialMask.release()
            bestMask.release()
            bestRotation.release()
            bestTranslation.release()
            bestEssential.release()
        }
    }

    private fun reprojectionError(
        intrinsics: PinholeIntrinsics,
        pose: CameraPose,
        point: DoubleArray,
        observation: CameraFeatureObservation
    ): Double? {
        val r = pose.rotation
        val t = pose.translation
        val x = r[0] * point[0] + r[1] * point[1] + r[2] * point[2] + t[0]
        val y = r[3] * point[0] + r[4] * point[1] + r[5] * point[2] + t[1]
        val z = r[6] * point[0] + r[7] * point[1] + r[8] * point[2] + t[2]
        return CameraProjectionMath.reprojectionError(
            x, y, z, observation.x, observation.y, intrinsics
        )
    }

    private fun toGray(source: Mat): Mat {
        if (source.channels() == 1) return source.clone()
        val gray = Mat()
        when (source.channels()) {
            4 -> Imgproc.cvtColor(source, gray, Imgproc.COLOR_RGBA2GRAY)
            3 -> Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY)
            else -> {
                gray.release()
                throw IllegalArgumentException("Unsupported frame channel count: ${source.channels()}")
            }
        }
        return gray
    }
}

/**
 * Multi-view track graph. Each node is a detected feature in one frame.
 * Verified edges are joined first; same-frame consistency is evaluated on the
 * completed component before a landmark track is returned.
 */
internal class GlobalTrackGraphBuilder(
    private val minimumObservations: Int = 3,
    private val maximumObservations: Int = 100,
    private val inconsistencyThresholdPx: Double = 10.0
) {
    init {
        require(minimumObservations >= 2)
        require(maximumObservations >= minimumObservations)
        require(inconsistencyThresholdPx.isFinite() && inconsistencyThresholdPx >= 0.0)
    }

    private val nodeIndexByKey = HashMap<Long, Int>()
    private var parent = IntArray(1024)
    private var rank = ByteArray(1024)
    private var minimumNodeByRoot = LongArray(1024)
    private var nodeCount = 0

    private data class TrackCandidate(
        val stableId: Long,
        val observations: List<CameraFeatureObservation>
    )

    fun addVerifiedMatches(
        firstFrame: Int,
        secondFrame: Int,
        matches: List<Pair<Int, Int>>
    ) {
        require(firstFrame >= 0 && secondFrame >= 0)
        require(firstFrame != secondFrame) { "Verified matches must connect distinct frames." }
        for ((firstFeature, secondFeature) in matches) {
            require(firstFeature >= 0 && secondFeature >= 0)
            val a = node(firstFrame, firstFeature)
            val b = node(secondFrame, secondFeature)
            union(a, b)
        }
    }

    fun build(
        frames: Array<GlobalReconstructionEngine.FrameFeatures?>,
        activeFrames: Set<Int>
    ): List<CameraFeatureTrack> = build(activeFrames) { frame, feature ->
        val keypoint = frames.getOrNull(frame)?.keypoints?.getOrNull(feature)
        if (keypoint == null) null
        else CameraFeatureObservation(frame, keypoint.pt.x, keypoint.pt.y, feature)
    }

    fun buildFromLocations(
        featureLocationsByFrame: Map<Int, List<CameraFeatureObservation>>,
        activeFrames: Set<Int>
    ): List<CameraFeatureTrack> = build(activeFrames) { frame, feature ->
        featureLocationsByFrame[frame]?.getOrNull(feature)?.copy(
            frameIndex = frame,
            featureIndex = feature
        )
    }

    private fun build(
        activeFrames: Set<Int>,
        observationForNode: (Int, Int) -> CameraFeatureObservation?
    ): List<CameraFeatureTrack> {
        val observations = HashMap<Int, MutableList<CameraFeatureObservation>>()
        for ((node, nodeIndex) in nodeIndexByKey) {
            val root = find(nodeIndex)
            val frame = (node ushr 32).toInt()
            if (frame !in activeFrames) continue
            val featureIndex = node.toInt()
            val observation = observationForNode(frame, featureIndex) ?: continue
            observations.getOrPut(root) { ArrayList() }.add(
                if (observation.frameIndex == frame) observation
                else observation.copy(frameIndex = frame)
            )
        }

        val candidates = observations.mapNotNull { (root, values) ->
            if (values.any { !it.x.isFinite() || !it.y.isFinite() }) {
                return@mapNotNull null
            }
            val ordered = values.sortedWith(
                compareBy<CameraFeatureObservation>({ it.frameIndex }, { it.x }, { it.y })
            )
            val consolidated = consolidateFrameObservations(ordered) ?: return@mapNotNull null
            if (consolidated.size !in minimumObservations..maximumObservations) {
                return@mapNotNull null
            }
            TrackCandidate(minimumNodeByRoot[root], consolidated)
        }.sortedWith(
            compareByDescending<TrackCandidate> { it.observations.size }
                .thenByDescending { it.stableId }
        )

        return candidates.mapIndexed { index, candidate ->
            CameraFeatureTrack(index, candidate.observations)
        }
    }

    /** A feature track can contribute at most one measurement from each camera. */
    private fun consolidateFrameObservations(
        observations: List<CameraFeatureObservation>
    ): List<CameraFeatureObservation>? {
        val thresholdSquared = inconsistencyThresholdPx * inconsistencyThresholdPx
        val consolidated = ArrayList<CameraFeatureObservation>()
        var groupStart = 0
        while (groupStart < observations.size) {
            val frameIndex = observations[groupStart].frameIndex
            var groupEnd = groupStart + 1
            while (groupEnd < observations.size && observations[groupEnd].frameIndex == frameIndex) {
                groupEnd++
            }
            for (firstIndex in groupStart until groupEnd) {
                for (secondIndex in firstIndex + 1 until groupEnd) {
                    val dx = observations[firstIndex].x - observations[secondIndex].x
                    val dy = observations[firstIndex].y - observations[secondIndex].y
                    if (dx * dx + dy * dy > thresholdSquared) return null
                }
            }
            val frameObservations = observations.subList(groupStart, groupEnd)
            val medianX = median(frameObservations.map { it.x })
            val medianY = median(frameObservations.map { it.y })
            // Keep an observed keypoint rather than inventing an averaged pixel.
            // The input ordering makes equal-distance ties deterministic.
            val representative = frameObservations.minByOrNull { observation ->
                val dx = observation.x - medianX
                val dy = observation.y - medianY
                dx * dx + dy * dy
            } ?: return null
            consolidated += representative
            groupStart = groupEnd
        }
        return consolidated
    }

    private fun median(values: List<Double>): Double {
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 1) {
            sorted[middle]
        } else {
            sorted[middle - 1] * 0.5 + sorted[middle] * 0.5
        }
    }

    private fun node(frame: Int, feature: Int): Long =
        (frame.toLong() shl 32) or (feature.toLong() and 0xffffffffL)

    private fun ensure(value: Long): Int {
        nodeIndexByKey[value]?.let { return it }
        if (nodeCount == parent.size) {
            check(parent.size <= Int.MAX_VALUE / 2) { "Feature graph contains too many nodes." }
            val newCapacity = parent.size * 2
            parent = parent.copyOf(newCapacity)
            rank = rank.copyOf(newCapacity)
            minimumNodeByRoot = minimumNodeByRoot.copyOf(newCapacity)
        }
        val index = nodeCount++
        parent[index] = index
        rank[index] = 0
        minimumNodeByRoot[index] = value
        nodeIndexByKey[value] = index
        return index
    }

    private fun find(value: Int): Int {
        var current = value
        while (parent[current] != current) {
            parent[current] = parent[parent[current]]
            current = parent[current]
        }
        return current
    }

    private fun union(first: Long, second: Long) {
        var a = find(ensure(first))
        var b = find(ensure(second))
        if (a == b) return
        if (rank[a] < rank[b]) {
            val swap = a
            a = b
            b = swap
        }
        parent[b] = a
        minimumNodeByRoot[a] = minOf(minimumNodeByRoot[a], minimumNodeByRoot[b])
        if (rank[a] == rank[b]) rank[a] = (rank[a] + 1).toByte()
    }
}
