package com.trackstudio.app.tracking

import com.trackstudio.app.tracking.colmap.ColmapTrackCompleter
import com.trackstudio.app.tracking.colmap.ColmapTrackCreator
import com.trackstudio.app.tracking.colmap.ColmapTrackContinuation
import com.trackstudio.app.tracking.colmap.ColmapTrackMerger
import com.trackstudio.app.tracking.colmap.ColmapTrackRetriangulator
import com.trackstudio.app.tracking.glomap.GlomapGlobalPositioningInitializer
import com.trackstudio.app.tracking.glomap.GlomapGlobalPositioningProblemBuilder
import com.trackstudio.app.tracking.glomap.GlomapGlobalBundleAdjuster
import com.trackstudio.app.tracking.glomap.GlomapViewGraphCalibrator
import com.trackstudio.app.tracking.glomap.GlomapBundleAdjustmentSchedule
import com.trackstudio.app.tracking.glomap.GlomapNativeSolver
import com.trackstudio.app.tracking.glomap.GlomapTrackFilter
import com.trackstudio.app.tracking.glomap.GlomapPostRetriangulationRefinement
import com.trackstudio.app.tracking.glomap.GlomapReconstructionNormalizer
import com.trackstudio.app.tracking.glomap.GlomapCovisibilityPruner
import com.trackstudio.app.tracking.glomap.GlomapTrackQuotaSelector
import com.trackstudio.app.tracking.glomap.GlomapImagePairInlierScorer
import com.trackstudio.app.tracking.glomap.GlomapRelativePoseEstimator
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
    private val maximumPairCount: Int = 420,
    // GLOMAP's corresponding skip_pruning option defaults to true.
    private val enableWeakConnectivityPruning: Boolean = false,
    // Negative follows the intended unlimited sentinel; see the selector's pinned-source note.
    private val minTracksPerView: Int = GlomapTrackQuotaSelector.DEFAULT_MIN_TRACKS_PER_VIEW,
    private val maximumTracks: Int = GlomapTrackQuotaSelector.DEFAULT_MAX_TRACKS
) : ReconstructionEngine {
    init {
        require(maxFeatures in 500..4000)
        require(ratioThreshold.isFinite() && ratioThreshold > 0f && ratioThreshold < 1f)
        require(maximumPairCount > 0)
        require(maximumTracks >= 0)
    }

    internal data class FrameFeatures(
        val keypoints: Array<KeyPoint>,
        val descriptors: Mat,
        val width: Int,
        val height: Int
    )

    private data class FundamentalPair(
        val first: Int,
        val second: Int,
        val fundamentalInlierIndices: IntArray,
        val allMatches: MutableList<Pair<Int, Int>>,
        val fundamentalMatrix: DoubleArray,
        val matchCount: Int
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
        val workingCameraMatrix = cameraMatrix.clone()
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

            val fx = workingCameraMatrix.get(0, 0)[0]
            val skew = workingCameraMatrix.get(0, 1)[0]
            val cx = workingCameraMatrix.get(0, 2)[0]
            val fy = workingCameraMatrix.get(1, 1)[0]
            val cy = workingCameraMatrix.get(1, 2)[0]
            require(listOf(fx, skew, cx, fy, cy).all { it.isFinite() } && fx > 0.0 && fy > 0.0) {
                "Camera intrinsics contain invalid focal lengths or principal point."
            }
            val initialIntrinsics = CameraIntrinsics(
                fx = fx, skew = skew, cx = cx, fy = fy, cy = cy
            )
            var effectiveIntrinsics = initialIntrinsics

            // Long clips need both a connected temporal backbone and wider
            // baselines. The default 420-pair cap must not collapse a 588-frame
            // sequence into adjacent-only comparisons.
            val candidatePairs = buildPairSchedule(frameCount, maxOf(maximumPairCount, frameCount * 2))
            val completedFundamentals = AtomicInteger(0)
            val progressMutex = Mutex()
            val workerCount = minOf(Runtime.getRuntime().availableProcessors().coerceAtLeast(1), 2, candidatePairs.size.coerceAtLeast(1))
            val chunks = candidatePairs.chunked((candidatePairs.size + workerCount - 1) / workerCount)
            val fundamentalPairs = coroutineScope {
                chunks.map { chunk ->
                    async(Dispatchers.Default) {
                        val workerMatcher = SiftDescriptorMatcher(ratioThreshold)
                        try {
                            chunk.mapNotNull { pair ->
                                currentCoroutineContext().ensureActive()
                                val first = frames[pair.first] ?: return@mapNotNull null
                                val second = frames[pair.second] ?: return@mapNotNull null
                                val candidate = estimateFundamentalPair(
                                    pair.first, pair.second, first, second,
                                    initialIntrinsics, workerMatcher
                                )
                                val completed = completedFundamentals.incrementAndGet()
                                if (completed % 2 == 0 || completed == candidatePairs.size) {
                                    progressMutex.withLock {
                                        val current = completedFundamentals.get()
                                        onProgress(CameraTrackingProgress(
                                            frameCount, frameCount,
                                            "Estimating view-graph fundamentals " + current + "/" + candidatePairs.size,
                                            0.25f + 0.13f * current / candidatePairs.size.coerceAtLeast(1),
                                            decodedFrames, frameCount
                                        ))
                                    }
                                }
                                candidate
                            }
                        } finally {
                            workerMatcher.clear()
                        }
                    }
                }.awaitAll().flatten()
            }
            // Descriptor Mats are no longer needed after fundamental estimation.
            frames.filterNotNull().forEach { it.descriptors.release() }

            val calibrationResult = try {
                GlomapViewGraphCalibrator().calibrate(
                    fundamentals = fundamentalPairs.map { it.fundamentalMatrix },
                    initialIntrinsics = initialIntrinsics
                )
            } catch (_: LinkageError) {
                null
            }
            if (calibrationResult != null) {
                effectiveIntrinsics = calibrationResult.intrinsics
                workingCameraMatrix.put(0, 0, *effectiveIntrinsics.toRowMajorMatrix())
            }
            val pairsForPoseEstimation = if (calibrationResult == null) {
                fundamentalPairs
            } else {
                fundamentalPairs.filterIndexed { index, _ ->
                    index in calibrationResult.acceptedPairIndices
                }
            }
            onProgress(CameraTrackingProgress(
                frameCount, frameCount, "Calibrating view-graph focal length",
                0.40f, decodedFrames, frameCount
            ))

            val completedVerifications = AtomicInteger(0)
            val verificationChunks = pairsForPoseEstimation.chunked(
                ((pairsForPoseEstimation.size + workerCount - 1) / workerCount).coerceAtLeast(1)
            )
            val verifiedPairs = coroutineScope {
                verificationChunks.map { chunk ->
                    async(Dispatchers.Default) {
                        chunk.mapNotNull { pair ->
                            currentCoroutineContext().ensureActive()
                            val first = frames[pair.first] ?: return@mapNotNull null
                            val second = frames[pair.second] ?: return@mapNotNull null
                            val verified = verifyPair(pair, first, second, workingCameraMatrix)
                            val completed = completedVerifications.incrementAndGet()
                            if (completed % 2 == 0 || completed == pairsForPoseEstimation.size) {
                                progressMutex.withLock {
                                    val current = completedVerifications.get()
                                    onProgress(CameraTrackingProgress(
                                        frameCount, frameCount,
                                        "Verifying calibrated image pairs " + current + "/" + pairsForPoseEstimation.size,
                                        0.42f + 0.13f * current / pairsForPoseEstimation.size.coerceAtLeast(1),
                                        decodedFrames, frameCount
                                    ))
                                }
                            }
                            verified
                        }
                    }
                }.awaitAll().flatten()
            }
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
            // The builder already orders by descending length and stable representative ID,
            // matching GLOMAP's track priority before dense Android IDs are assigned.
            val priorityOrderedTracks = trackBuilder.build(frames, rotations.keys)
            val tracks = GlomapTrackQuotaSelector.select(
                priorityOrderedTracks = priorityOrderedTracks,
                registeredFrameIds = activeFrames.toSet(),
                minTracksPerView = minTracksPerView,
                maxNumTracks = maximumTracks
            ).selectedTracks
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

            val positionTracks = tracks.map { track ->
                val observations = track.observations.mapNotNull { observation ->
                    val rotation = rotations[observation.frameIndex] ?: return@mapNotNull null
                    val center = initializer.frameTranslationsById[observation.frameIndex]
                        ?: return@mapNotNull null
                    val normalizedY = (observation.y - effectiveIntrinsics.cy) / effectiveIntrinsics.fy
                    val normalizedX = (observation.x - effectiveIntrinsics.cx -
                        effectiveIntrinsics.skew * normalizedY) / effectiveIntrinsics.fx
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
            val rawPositionResult = try {
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
            if (!SolverObjective.didNotIncrease(rawPositionResult.initialCost, rawPositionResult.finalCost)) {
                throw IllegalStateException("The global camera-position solve increased its objective; no result was published.")
            }
            var reconstructionTracks = GlomapTrackFilter.filterAfterPositioning(
                tracks = tracks,
                cameraCentersByFrame = rawPositionResult.frameCentersById,
                rotationsByFrame = rotations,
                pointsByTrackId = rawPositionResult.pointsByTrackId,
                intrinsics = effectiveIntrinsics,
                hasPriorFocalLength = true
            )
            val positionNormalization = GlomapReconstructionNormalizer.normalize(
                frameCentersById = rawPositionResult.frameCentersById,
                pointsByTrackId = rawPositionResult.pointsByTrackId
            )
            val positionResult = rawPositionResult.copy(
                frameCentersById = positionNormalization.frameCentersById,
                pointsByTrackId = positionNormalization.pointsByTrackId
            )

            onProgress(
                CameraTrackingProgress(
                    frameIndex = frameCount,
                    frameCount = frameCount,
                    phase = "Staged global bundle adjustment",
                    overallFraction = 0.88f,
                    completedFrames = decodedFrames,
                    totalFrames = frameCount
                )
            )
            val bundleCandidate = try {
                val bundleAdjuster = GlomapGlobalBundleAdjuster()
                GlomapBundleAdjustmentSchedule.solve(
                    initialState = GlomapBundleAdjustmentSchedule.State(
                        frameCentersById = positionResult.frameCentersById,
                        rotationsByFrame = rotations,
                        pointsByTrackId = positionResult.pointsByTrackId,
                        tracks = reconstructionTracks,
                        intrinsics = effectiveIntrinsics
                    ),
                    stageSolver = { state, optimizeRotations, stageIndex ->
                        val stageCameraMatrix = Mat(3, 3, CvType.CV_64F)
                        try {
                            stageCameraMatrix.put(0, 0, *state.intrinsics.toRowMajorMatrix())
                            bundleAdjuster.solve(
                                frameCentersById = state.frameCentersById,
                                rotationsByFrame = state.rotationsByFrame,
                                pointsByTrackId = state.pointsByTrackId,
                                tracks = state.tracks,
                                cameraMatrix = stageCameraMatrix,
                                imageWidth = frameSizes.single().first,
                                imageHeight = frameSizes.single().second,
                                optimizeRotations = optimizeRotations,
                                optimizeTranslation = true,
                                maxIterations = 45,
                                numThreads = if (activeFrames.size >= 300) 2 else Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
                                onIteration = { iteration, total ->
                                    val stageFraction =
                                        (stageIndex.toFloat() + iteration.toFloat() / total.coerceAtLeast(1)) /
                                            GlomapBundleAdjustmentSchedule.MAX_STAGE_COUNT
                                    val stageName = if (optimizeRotations) "rotation" else "position"
                                    publishProgress(
                                        "Global bundle adjustment " + stageName + " stage " +
                                            (stageIndex + 1) + "/" + GlomapBundleAdjustmentSchedule.MAX_STAGE_COUNT,
                                        0.88f + 0.05f * stageFraction.coerceIn(0f, 1f)
                                    )
                                }
                            )
                        } finally {
                            stageCameraMatrix.release()
                        }
                    }
                )
            } catch (_: LinkageError) {
                null
            }
            val bundleResult = bundleCandidate?.bundleResult?.takeIf {
                SolverObjective.didNotIncrease(it.initialCost, it.finalCost)
            }
            val bundleStatus = when {
                bundleCandidate == null -> "Unavailable; retained global-positioning result"
                bundleResult == null -> "Objective increased; retained global-positioning result"
                else -> "Applied"
            }
            var finalCenters = bundleResult?.frameCentersById ?: positionResult.frameCentersById
            var finalRotations = bundleResult?.rotationsByFrame ?: rotations
            val finalPoints = bundleResult?.pointsByTrackId ?: positionResult.pointsByTrackId
            var finalPointsForLandmarks = finalPoints
            var refinedIntrinsics = bundleResult?.intrinsics
            if (bundleResult != null) {
                reconstructionTracks = bundleCandidate?.tracks ?: reconstructionTracks
            }
            val initialRefinedIntrinsics = refinedIntrinsics
            val resultCameraMatrix = if (initialRefinedIntrinsics != null) {
                Mat(3, 3, CvType.CV_64F).also { matrix ->
                    refinedCameraMatrix = matrix
                    matrix.put(0, 0, *initialRefinedIntrinsics.toRowMajorMatrix())
                }
            } else workingCameraMatrix

            var poses = activeFrames.mapNotNull { frame ->
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
            var poseByFrame = poses.associateBy { it.frameIndex }
            var projectionIntrinsics = refinedIntrinsics?.let {
                PinholeIntrinsics(it.fx, it.fy, it.skew, it.cx, it.cy)
            } ?: PinholeIntrinsics(
                fx = resultCameraMatrix.get(0, 0)[0],
                fy = resultCameraMatrix.get(1, 1)[0],
                skew = resultCameraMatrix.get(0, 1)[0],
                cx = resultCameraMatrix.get(0, 2)[0],
                cy = resultCameraMatrix.get(1, 2)[0]
            )
            var triangulationPass = 0
            var unassignedTracks = emptyList<CameraFeatureTrack>()
            val pairMatchesForContinuation = activePairs.map { pair ->
                ColmapTrackContinuation.PairMatches(pair.first, pair.second, pair.inlierMatches)
            }
            val triangulationProgress: (Int, Int) -> Unit = { completed, total ->
                val progressStart = 0.94f + 0.0125f * triangulationPass
                publishProgress(
                    "Triangulating landmarks " + completed + "/" + total,
                    progressStart + 0.0125f * completed.toFloat() / total.coerceAtLeast(1)
                )
            }
            fun triangulateWithColmapCreation(): ColmapTrackCreator.Result {
                val continuedBeforeCreate = ColmapTrackContinuation.continueTracks(
                    tracks = reconstructionTracks + unassignedTracks,
                    pointsByTrackId = finalPointsForLandmarks,
                    pairMatches = pairMatchesForContinuation,
                    poses = poses,
                    intrinsics = projectionIntrinsics
                )
                val result = ColmapTrackCreator.create(
                    tracks = continuedBeforeCreate.tracks,
                    poses = poses,
                    intrinsics = projectionIntrinsics,
                    onProgress = triangulationProgress
                )
                val pointTrackIds = result.tracks.mapTo(HashSet<Int>()) { it.id }
                val updatedPoints = finalPointsForLandmarks.filterKeys { it in pointTrackIds }
                    .toMutableMap()
                result.landmarks.forEach { landmark ->
                    if (landmark.trackId !in updatedPoints) {
                        updatedPoints[landmark.trackId] = doubleArrayOf(
                            landmark.position.x,
                            landmark.position.y,
                            landmark.position.z
                        )
                    }
                }
                val continuedAfterCreate = ColmapTrackContinuation.continueTracks(
                    tracks = result.tracks + result.unassignedTracks,
                    pointsByTrackId = updatedPoints,
                    pairMatches = pairMatchesForContinuation,
                    poses = poses,
                    intrinsics = projectionIntrinsics
                )
                val continuedPointTracks = continuedAfterCreate.tracks.filter { it.id in pointTrackIds }
                val continuedUnassignedTracks = continuedAfterCreate.tracks.filterNot { it.id in pointTrackIds }
                val continuedPointTracksById = continuedPointTracks.associateBy { it.id }
                val updatedLandmarks = result.landmarks.map { landmark ->
                    val point = updatedPoints[landmark.trackId] ?: return@map landmark
                    val track = continuedPointTracksById[landmark.trackId] ?: return@map landmark
                    val errors = track.observations.mapNotNull { observation ->
                        val pose = poseByFrame[observation.frameIndex] ?: return@mapNotNull null
                        reprojectionError(projectionIntrinsics, pose, point, observation)
                    }
                    if (errors.isEmpty()) landmark else landmark.copy(
                        position = Point3(point[0], point[1], point[2]),
                        observationCount = errors.size,
                        meanReprojectionErrorPx = errors.average()
                    )
                }
                reconstructionTracks = continuedPointTracks
                unassignedTracks = continuedUnassignedTracks
                finalPointsForLandmarks = updatedPoints.filterKeys { it in pointTrackIds }
                val updatedStats = result.stats.copy(
                    meanReprojectionErrorPx = updatedLandmarks.map { it.meanReprojectionErrorPx }
                        .takeIf { it.isNotEmpty() }?.average() ?: 0.0
                )
                return result.copy(
                    tracks = continuedPointTracks,
                    landmarks = updatedLandmarks,
                    stats = updatedStats,
                    unassignedTracks = continuedUnassignedTracks
                )
            }
            var triangulated = triangulateWithColmapCreation()
            val featureAt: (Int, Int) -> CameraFeatureObservation? = { frame, feature ->
                val keypoint = frames.getOrNull(frame)?.keypoints?.getOrNull(feature)
                keypoint?.let {
                    CameraFeatureObservation(frame, it.pt.x, it.pt.y, feature)
                }
            }
            val pairMatchesForCompletion = activePairs.map { pair ->
                ColmapTrackCompleter.PairMatches(pair.first, pair.second, pair.inlierMatches)
            }
            val completion = ColmapTrackCompleter.completeTracks(
                tracks = reconstructionTracks,
                landmarks = triangulated.landmarks,
                pairMatches = pairMatchesForCompletion,
                activeFrames = activeFrames.toSet(),
                poses = poses,
                intrinsics = projectionIntrinsics,
                featureAt = featureAt
            )
            if (completion.addedObservations > 0) {
                reconstructionTracks = GlomapTrackFilter.filterAfterBundleAdjustment(
                    tracks = completion.tracks,
                    cameraCentersByFrame = finalCenters,
                    rotationsByFrame = finalRotations,
                    pointsByTrackId = finalPointsForLandmarks,
                    intrinsics = refinedIntrinsics ?: effectiveIntrinsics
                )
                triangulationPass++
                triangulated = triangulateWithColmapCreation()
            }
            val retriangulation = ColmapTrackRetriangulator.retriangulate(
                tracks = reconstructionTracks,
                landmarks = triangulated.landmarks,
                pairMatches = activePairs.map { pair ->
                    ColmapTrackRetriangulator.PairMatches(
                        pair.first, pair.second, pair.inlierMatches
                    )
                },
                poses = poses,
                intrinsics = projectionIntrinsics,
                featureAt = featureAt
            )
            if (retriangulation.addedObservations > 0) {
                reconstructionTracks = retriangulation.tracks
                val observationsAlreadyInPoints = reconstructionTracks.asSequence()
                    .flatMap { track -> track.observations.asSequence() }
                    .map { it.frameIndex to it.featureIndex }
                    .toHashSet()
                var nextUnassignedTrackId =
                    (reconstructionTracks.maxOfOrNull { it.id }?.toLong() ?: -1L) + 1L
                unassignedTracks = unassignedTracks.mapNotNull { track ->
                    val residual = track.observations.filterNot {
                        (it.frameIndex to it.featureIndex) in observationsAlreadyInPoints
                    }
                    if (residual.isEmpty() || nextUnassignedTrackId > Int.MAX_VALUE.toLong()) {
                        null
                    } else {
                        track.copy(id = nextUnassignedTrackId.toInt(), observations = residual)
                            .also { nextUnassignedTrackId++ }
                    }
                }
                triangulationPass++
                triangulated = triangulateWithColmapCreation()
                val completionAfterRetriangulation = ColmapTrackCompleter.completeTracks(
                    tracks = reconstructionTracks,
                    landmarks = triangulated.landmarks,
                    pairMatches = pairMatchesForCompletion,
                    activeFrames = activeFrames.toSet(),
                    poses = poses,
                    intrinsics = projectionIntrinsics,
                    featureAt = featureAt
                )
                if (completionAfterRetriangulation.addedObservations > 0) {
                    reconstructionTracks = completionAfterRetriangulation.tracks
                    triangulationPass++
                    triangulated = triangulateWithColmapCreation()
                }
            }
            val trackMerging = ColmapTrackMerger.mergeTracks(
                tracks = reconstructionTracks,
                landmarks = triangulated.landmarks,
                pairMatches = activePairs.map { pair ->
                    ColmapTrackMerger.PairMatches(pair.first, pair.second, pair.inlierMatches)
                },
                poses = poses,
                intrinsics = projectionIntrinsics
            )
            if (trackMerging.mergedTrackCount > 0) {
                reconstructionTracks = trackMerging.tracks
                triangulated = triangulated.copy(
                    tracks = trackMerging.tracks,
                    landmarks = trackMerging.landmarks,
                    stats = triangulated.stats.withFinalLandmarks(trackMerging.landmarks)
                )
                val mergedPoints = finalPointsForLandmarks.toMutableMap()
                trackMerging.removedTrackIds.forEach { mergedPoints.remove(it) }
                trackMerging.mergedLandmarks.forEach { landmark ->
                    mergedPoints[landmark.trackId] = doubleArrayOf(
                        landmark.position.x,
                        landmark.position.y,
                        landmark.position.z
                    )
                }
                finalPointsForLandmarks = mergedPoints
            }
            val postRetriangulationInitialState =
                GlomapPostRetriangulationRefinement.State(
                    frameCentersById = finalCenters,
                    rotationsByFrame = finalRotations,
                    pointsByTrackId = finalPointsForLandmarks,
                    tracks = reconstructionTracks,
                    intrinsics = refinedIntrinsics ?: effectiveIntrinsics
                )
            val postRetriangulationProgressStart = maxOf(
                0.96f,
                0.94f + 0.0125f * (triangulationPass + 1)
            ).coerceAtMost(0.99f)
            val postRetriangulationProgressEnd = postRetriangulationProgressStart +
                (0.995f - postRetriangulationProgressStart) * 0.4f
            onProgress(
                CameraTrackingProgress(
                    frameIndex = frameCount,
                    frameCount = frameCount,
                    phase = "Refining reconstruction after retriangulation",
                    overallFraction = postRetriangulationProgressStart,
                    completedFrames = decodedFrames,
                    totalFrames = frameCount
                )
            )
            val postRetriangulationRefinement = try {
                val postRetriangulationBundleAdjuster = GlomapGlobalBundleAdjuster()
                GlomapPostRetriangulationRefinement.refine(
                    initialState = postRetriangulationInitialState,
                    solveBundleAdjustment = { state, solveIndex ->
                        try {
                            val stageCameraMatrix = Mat(3, 3, CvType.CV_64F)
                            try {
                                stageCameraMatrix.put(0, 0, *state.intrinsics.toRowMajorMatrix())
                                postRetriangulationBundleAdjuster.solve(
                                    frameCentersById = state.frameCentersById,
                                    rotationsByFrame = state.rotationsByFrame,
                                    pointsByTrackId = state.pointsByTrackId,
                                    tracks = state.tracks,
                                    cameraMatrix = stageCameraMatrix,
                                    imageWidth = frameSizes.single().first,
                                    imageHeight = frameSizes.single().second,
                                    optimizeRotations = true,
                                    optimizeTranslation = true,
                                    maxIterations = 45,
                                    numThreads = if (activeFrames.size >= 300) 2 else Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
                                    onIteration = { iteration, total ->
                                        val solveFraction = (
                                            solveIndex.toFloat() +
                                                iteration.toFloat() / total.coerceAtLeast(1)
                                            ) / 2f
                                        publishProgress(
                                            "Post-retriangulation bundle adjustment " +
                                                (solveIndex + 1) + "/2",
                                            postRetriangulationProgressStart +
                                                (postRetriangulationProgressEnd - postRetriangulationProgressStart) *
                                                solveFraction.coerceIn(0f, 1f)
                                        )
                                    }
                                )
                            } finally {
                                stageCameraMatrix.release()
                            }
                        } catch (_: LinkageError) {
                            null
                        }
                    }
                )
            } catch (_: LinkageError) {
                GlomapPostRetriangulationRefinement.refine(
                    initialState = postRetriangulationInitialState,
                    solveBundleAdjustment = { _, _ -> null }
                )
            }
            val postFirstAccepted = postRetriangulationRefinement.firstBundleAdjustment
                ?.takeIf { postRetriangulationRefinement.firstBundleAdjustmentAccepted }
            val postSecondAccepted = postRetriangulationRefinement.secondBundleAdjustment
                ?.takeIf { postRetriangulationRefinement.secondBundleAdjustmentAccepted }
            val postLastAccepted = postRetriangulationRefinement.lastAcceptedBundleAdjustment
            val postRetriangulationBundleAdjustmentStatus = when {
                postRetriangulationRefinement.firstBundleAdjustment == null ->
                    "Unavailable; retained retriangulated state"
                !postRetriangulationRefinement.firstBundleAdjustmentAccepted ->
                    "Objective increased; retained retriangulated state"
                postRetriangulationRefinement.secondBundleAdjustment == null ->
                    "Second solve unavailable; retained first adjustment"
                !postRetriangulationRefinement.secondBundleAdjustmentAccepted ->
                    "Second objective increased; retained first adjustment"
                else -> "Applied"
            }
            val postRetriangulationBundleAdjustmentIterations =
                (postFirstAccepted?.iterations ?: 0) + (postSecondAccepted?.iterations ?: 0)
            val postRetriangulationState = postRetriangulationRefinement.state
            reconstructionTracks = postRetriangulationState.tracks
            finalCenters = postRetriangulationState.frameCentersById
            finalRotations = postRetriangulationState.rotationsByFrame
            finalPointsForLandmarks = postRetriangulationState.pointsByTrackId
            val postNormalization = postRetriangulationRefinement.normalization
            if (postNormalization.applied) {
                triangulated = triangulated.copy(
                    landmarks = triangulated.landmarks.map { landmark ->
                        val point = postNormalization.transformPoint(
                            doubleArrayOf(
                                landmark.position.x,
                                landmark.position.y,
                                landmark.position.z
                            )
                        )
                        landmark.copy(position = Point3(point[0], point[1], point[2]))
                    }
                )
            }
            projectionIntrinsics = PinholeIntrinsics(
                fx = postRetriangulationState.intrinsics.fx,
                fy = postRetriangulationState.intrinsics.fy,
                skew = postRetriangulationState.intrinsics.skew,
                cx = postRetriangulationState.intrinsics.cx,
                cy = postRetriangulationState.intrinsics.cy
            )
            if (postLastAccepted != null) {
                refinedIntrinsics = postRetriangulationState.intrinsics
            }
            resultCameraMatrix.put(0, 0, *postRetriangulationState.intrinsics.toRowMajorMatrix())
            poses = activeFrames.mapNotNull { frame ->
                val rotation = finalRotations[frame] ?: return@mapNotNull null
                val center = finalCenters[frame] ?: return@mapNotNull null
                val translation = doubleArrayOf(
                    -(rotation[0] * center[0] + rotation[1] * center[1] + rotation[2] * center[2]),
                    -(rotation[3] * center[0] + rotation[4] * center[1] + rotation[5] * center[2]),
                    -(rotation[6] * center[0] + rotation[7] * center[1] + rotation[8] * center[2])
                )
                CameraPose(frame, rotation.copyOf(), translation, CameraPoseStatus.RELATIVE_ESTIMATE)
            }
            poseByFrame = poses.associateBy { it.frameIndex }
            val bundleLandmarks = reconstructionTracks.mapNotNull trackLoop@ { track ->
                val point = finalPointsForLandmarks[track.id] ?: return@trackLoop null
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
            val landmarkRefinementStart = maxOf(
                postRetriangulationProgressEnd,
                0.96f,
                0.94f + 0.0125f * (triangulationPass + 1)
            ).coerceAtMost(0.99f)
            val mergedLandmarks = linkedMapOf<Int, CameraLandmark>()
            bundleLandmarks.forEach { mergedLandmarks[it.trackId] = it }
            if (postRetriangulationRefinement.acceptedBundleAdjustmentCount == 0) {
                val survivingTrackIds = reconstructionTracks.asSequence()
                    .filter { it.observations.size >= 2 }
                    .map { it.id }
                    .toSet()
                triangulated.landmarks.filter { it.trackId in survivingTrackIds }.forEach { candidate ->
                    val previous = mergedLandmarks[candidate.trackId]
                    if (previous == null ||
                        candidate.meanReprojectionErrorPx < previous.meanReprojectionErrorPx
                    ) mergedLandmarks[candidate.trackId] = candidate
                }
            }
            val refinedLandmarks = MobileBundleAdjuster.optimizeLandmarks(
                featureTracks = reconstructionTracks,
                landmarks = mergedLandmarks.values.toList(),
                poses = poses,
                cameraMatrix = resultCameraMatrix,
                onProgress = { completed, total ->
                    publishProgress(
                        "Refining landmarks " + completed + "/" + total,
                        landmarkRefinementStart +
                            (0.995f - landmarkRefinementStart) *
                            completed.toFloat() / total.coerceAtLeast(1)
                    )
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
            val covisibilityPruning = if (enableWeakConnectivityPruning) {
                GlomapCovisibilityPruner.prune(
                    tracks = reconstructionTracks,
                    allFrameIds = (0 until frameCount).toSet(),
                    activeFrameIds = activeFrames.toSet()
                )
            } else {
                null
            }
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
                postRetriangulationBundleAdjustmentApplied = postLastAccepted != null,
                postRetriangulationBundleAdjustmentInitialCost =
                    postFirstAccepted?.initialCost ?: Double.NaN,
                postRetriangulationBundleAdjustmentFinalCost =
                    postLastAccepted?.finalCost ?: Double.NaN,
                postRetriangulationBundleAdjustmentIterations =
                    postRetriangulationBundleAdjustmentIterations,
                postRetriangulationBundleAdjustmentStatus =
                    postRetriangulationBundleAdjustmentStatus,
                globalViewGraphFocalCalibrationApplied = calibrationResult?.focalLengthUpdated ?: false,
                globalViewGraphInitialFocalLength = initialIntrinsics.fx * 0.5 + initialIntrinsics.fy * 0.5,
                globalViewGraphFocalLength = effectiveIntrinsics.fx * 0.5 + effectiveIntrinsics.fy * 0.5,
                globalViewGraphRejectedPairCount = calibrationResult?.rejectedPairIndices?.size ?: 0,
                globalViewGraphCalibrationStatus = when {
                    calibrationResult == null -> "Unavailable; retained input focal length"
                    calibrationResult.focalLengthUpdated -> "Applied"
                    else -> "Pairs filtered; retained input focal length"
                },
                refinedCameraIntrinsics = refinedIntrinsics
                    ?: effectiveIntrinsics.takeIf { calibrationResult?.focalLengthUpdated == true },
                globalCovisibilityPruningApplied = covisibilityPruning != null,
                globalCovisibilityClusterIdByFrame =
                    covisibilityPruning?.clusterIdByFrame ?: emptyMap(),
                globalCovisibilityClusterCount = covisibilityPruning?.clusterCount ?: 0,
                globalCovisibilityStrongThreshold =
                    covisibilityPruning?.strongThreshold ?: Double.NaN,
                globalCovisibilityEligiblePairCount =
                    covisibilityPruning?.eligiblePairWeights?.size ?: 0,
                globalCovisibilityBridgeIterations = covisibilityPruning?.bridgeIterations ?: 0
            )
        } finally {
            frames.filterNotNull().forEach { it.descriptors.release() }
            refinedCameraMatrix?.release()
            workingCameraMatrix.release()
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

    private fun estimateFundamentalPair(
        firstIndex: Int,
        secondIndex: Int,
        first: FrameFeatures,
        second: FrameFeatures,
        intrinsics: CameraIntrinsics,
        matcher: SiftDescriptorMatcher
    ): FundamentalPair? {
        if (first.descriptors.empty() || second.descriptors.empty() ||
            first.keypoints.isEmpty() || second.keypoints.isEmpty()
        ) return null

        val matched = matcher.match(first.descriptors, second.descriptors)
            .filter { (query, train) ->
                query in first.keypoints.indices && train in second.keypoints.indices
            }.toMutableList()
        if (matched.size < 30) return null

        val points1 = MatOfPoint2f()
        val points2 = MatOfPoint2f()
        val fundamentalMask = Mat()
        var fundamental = Mat()
        return try {
            points1.fromList(matched.map { zeroSkewPoint(first.keypoints[it.first].pt, intrinsics) })
            points2.fromList(matched.map { zeroSkewPoint(second.keypoints[it.second].pt, intrinsics) })
            fundamental = Calib3d.findFundamentalMat(
                points1, points2, Calib3d.FM_RANSAC, 1.0, 0.999, 900, fundamentalMask
            )
            if (fundamental.empty() || fundamental.rows() != 3 || fundamental.cols() != 3 ||
                fundamentalMask.empty() || fundamentalMask.total().toInt() != matched.size
            ) return null
            val fundamentalInlierIndices = IntArray(matched.size)
            var inlierCount = 0
            for (index in matched.indices) {
                val maskValue = if (fundamentalMask.rows() == 1) {
                    fundamentalMask.get(0, index)
                } else {
                    fundamentalMask.get(index, 0)
                }
                val accepted = maskValue?.firstOrNull() ?: 0.0
                if (accepted != 0.0) fundamentalInlierIndices[inlierCount++] = index
            }
            if (inlierCount < 30 || inlierCount.toDouble() / matched.size < 0.20) return null
            val fundamentalValues = OpenCvMatrixReadback.readCv64F(fundamental, 3, 3) ?: return null
            FundamentalPair(
                firstIndex, secondIndex, fundamentalInlierIndices.copyOf(inlierCount), matched,
                fundamentalValues, matched.size
            )
        } finally {
            points1.release()
            points2.release()
            fundamentalMask.release()
            fundamental.release()
        }
    }

    private fun zeroSkewPoint(point: Point, intrinsics: CameraIntrinsics): Point =
        Point(
            point.x - intrinsics.skew * ((point.y - intrinsics.cy) / intrinsics.fy),
            point.y
        )

    private fun verifyPair(
        pair: FundamentalPair,
        first: FrameFeatures,
        second: FrameFeatures,
        cameraMatrix: Mat
    ): VerifiedPair? {
        if (first.keypoints.isEmpty() || second.keypoints.isEmpty() ||
            pair.allMatches.size != pair.matchCount ||
            pair.fundamentalInlierIndices.any { it !in pair.allMatches.indices }
        ) {
            pair.allMatches.clear()
            return null
        }
        if (pair.fundamentalInlierIndices.size < 30) {
            pair.allMatches.clear()
            return null
        }

        return try {
            val cameraMatrixValues = OpenCvMatrixReadback.readCv64F(cameraMatrix, 3, 3) ?: return null
            val pinholeIntrinsics = PinholeIntrinsics(
                fx = cameraMatrixValues[0],
                fy = cameraMatrixValues[4],
                skew = cameraMatrixValues[1],
                cx = cameraMatrixValues[2],
                cy = cameraMatrixValues[5]
            )
            val firstPixels = DoubleArray(pair.allMatches.size * 2)
            val secondPixels = DoubleArray(pair.allMatches.size * 2)
            pair.allMatches.forEachIndexed { index, match ->
                val firstPoint = first.keypoints[match.first].pt
                val secondPoint = second.keypoints[match.second].pt
                firstPixels[index * 2] = firstPoint.x
                firstPixels[index * 2 + 1] = firstPoint.y
                secondPixels[index * 2] = secondPoint.x
                secondPixels[index * 2 + 1] = secondPoint.y
            }
            // GLOMAP estimates from the full pair match set; the prior
            // fundamental inliers remain a pair-quality gate for this adapter.
            val estimate = GlomapRelativePoseEstimator.estimate(
                firstPixelsXY = firstPixels,
                secondPixelsXY = secondPixels,
                firstWidth = first.width,
                firstHeight = first.height,
                secondWidth = second.width,
                secondHeight = second.height,
                firstIntrinsics = pinholeIntrinsics,
                secondIntrinsics = pinholeIntrinsics,
                maxIterations = GlomapRelativePoseEstimator.ANDROID_MAX_ITERATIONS
            ) ?: return null
            val rotationValues = estimate.rotation
            val translationValues = estimate.translation
            val rescored = GlomapImagePairInlierScorer.scoreEssential(
                firstPixels = pair.allMatches.map { match ->
                    val point = first.keypoints[match.first].pt
                    point.x to point.y
                },
                secondPixels = pair.allMatches.map { match ->
                    val point = second.keypoints[match.second].pt
                    point.x to point.y
                },
                firstIntrinsics = pinholeIntrinsics,
                secondIntrinsics = pinholeIntrinsics,
                rotation = rotationValues,
                translation = translationValues
            ) ?: return null
            val inlierMatches = rescored.inlierIndices.map { pair.allMatches[it] }
            if (inlierMatches.size < GlomapImagePairInlierScorer.MINIMUM_INLIERS ||
                inlierMatches.size.toDouble() / pair.matchCount <
                GlomapImagePairInlierScorer.MINIMUM_INLIER_RATIO
            ) return null
            VerifiedPair(
                pair.first, pair.second, rotationValues,
                translationValues, inlierMatches, pair.matchCount
            )
        } finally {
            pair.allMatches.clear()
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
