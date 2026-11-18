package com.trackstudio.app.tracking

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.KeyPoint
import org.opencv.core.Point
import org.opencv.imgproc.Imgproc
import org.opencv.core.Point3
import com.trackstudio.app.tracking.colmap.ColmapLocalBundleSelector
import com.trackstudio.app.tracking.colmap.ColmapLocalBundleTrackSelector
import com.trackstudio.app.tracking.colmap.ColmapNextImageSelector
import com.trackstudio.app.tracking.colmap.ColmapRegistrationTrialLedger
import com.trackstudio.app.tracking.colmap.ColmapIterativeLocalRefinement
import com.trackstudio.app.tracking.colmap.ColmapTrackCompleter
import com.trackstudio.app.tracking.colmap.ColmapTrackMerger
import com.trackstudio.app.tracking.glomap.GlomapGlobalPositioningProblemBuilder
import com.trackstudio.app.tracking.colmap.ColmapObservationFilter
import com.trackstudio.app.tracking.glomap.GlomapNativeSolver
import kotlin.math.hypot
import kotlin.math.max

internal fun <T> withSourceReleaseOnFailure(source: Mat, block: (Mat) -> T): T {
    try {
        return block(source)
    } catch (failure: Throwable) {
        source.release()
        throw failure
    }
}

/**
 * Incremental, map-based camera solve.
 *
 * Pass one evaluates a bounded set of temporal pairs and selects a geometrically
 * well-conditioned initialization. Pass two seeds persistent tracks from that
 * pair, registers later frames with PnP against the same 3D map, and periodically
 * adds triangulated landmarks. No per-frame recoverPose translation is accumulated.
 *
 * The sparse map uses the arbitrary unit baseline of the initial pair. It is
 * internally consistent, but has no metric scale until a user supplies scale.
 */
internal class IncrementalSfMSolver(
    private val maxFeatures: Int = 1800,
    private val ratioThreshold: Float = 0.8f,
    private val cameraFocalLengthHasPrior: Boolean = false
) {
    private companion object {
        const val INITIAL_PAIR_SAMPLE_FRAME_LIMIT = 24
        const val INITIAL_PAIR_COARSE_DESCRIPTOR_LIMIT = 48
        const val INITIAL_PAIR_COARSE_EDGE_LIMIT = 320
        const val INITIAL_PAIR_GEOMETRY_LIMIT = 36
        const val INITIAL_PAIR_CORRESPONDENCE_RANK_LIMIT = 24
    }

    private data class CachedFrame(
        val keypoints: Array<KeyPoint>,
        val descriptors: Mat,
        val width: Int,
        val height: Int
    )

    private data class PairGeometry(
        val matches: List<Pair<Int, Int>>,
        val rotation: DoubleArray,
        val translation: DoubleArray,
        val inlierCount: Int
    )

    suspend fun solve(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit
    ): CameraTrackingResult = withContext(Dispatchers.Default) {
        require(frameCount >= 3) { "Camera tracking needs at least three frames for initialization." }
        require(!cameraMatrix.empty() && cameraMatrix.rows() == 3 && cameraMatrix.cols() == 3) {
            "A valid 3x3 camera intrinsic matrix is required."
        }
        check(OpenCvRuntime.ensureLoaded()) { "OpenCV could not be initialized on this device." }

        val temporalPairCandidates = InitialPairSelector.temporalCandidates(
            frameCount = frameCount,
            anchorCount = 6,
            offsets = listOf(2, 4, 6, 9, 13, 18, 24, 32),
            maximumPairs = 36
        )
        val sampledFrameIndices = InitialPairSelector.uniformFrameSamples(
            frameCount, INITIAL_PAIR_SAMPLE_FRAME_LIMIT
        )
        val sampledPairCandidates = InitialPairSelector.sampledFramePairs(
            sampledFrameIndices, minimumFrameGap = 2
        )
        val coarsePairCandidates = (temporalPairCandidates + sampledPairCandidates)
            .distinct()
            .take(INITIAL_PAIR_COARSE_EDGE_LIMIT)
        require(coarsePairCandidates.isNotEmpty()) { "No initialization frame pairs are available." }
        val candidateFrames = coarsePairCandidates.flatMap { listOf(it.first, it.second) }.toSet()
        val processedFrameFlags = BooleanArray(frameCount)
        var processedFrameCount = 0
        fun markFrameProcessed(index: Int) {
            if (index in processedFrameFlags.indices && !processedFrameFlags[index]) {
                processedFrameFlags[index] = true
                processedFrameCount++
            }
        }
        val cache = linkedMapOf<Int, CachedFrame>()
        val coarseDescriptorCache = linkedMapOf<Int, Mat>()
        val featureExtractor = AndroidSiftFeatureExtractor(maxFeatures.coerceIn(400, 5000))
        val matcher = SiftDescriptorMatcher(ratioThreshold)
        var referenceFrameSize: Pair<Int, Int>? = null
        val originalCameraMatrix = cameraMatrix.clone()

        try {
            // Bounded first pass: retain descriptors only for frames participating
            // in the coarse pair schedule, not for every frame in a long sequence.
            val orderedCandidateFrames = candidateFrames.sorted()
            for ((sampleIndex, index) in orderedCandidateFrames.withIndex()) {
                currentCoroutineContext().ensureActive()
                val source = frameProvider(index)
                    ?: throw IllegalStateException("Unable to decode frame ${index + 1}.")
                try {
                    referenceFrameSize = TrackingFrameDimensions.validate(
                        referenceFrameSize, source.cols(), source.rows()
                    )
                    if (index in candidateFrames) {
                        val gray = toGray(source)
                        val keypoints = org.opencv.core.MatOfKeyPoint()
                        val descriptors = Mat()
                        try {
                            val detectionMask = Mat()
                            try {
                                featureExtractor.detectAndCompute(gray, detectionMask, keypoints, descriptors)
                            } finally {
                                detectionMask.release()
                            }
                            cache[index] = CachedFrame(
                                keypoints.toArray(), descriptors,
                                source.cols(), source.rows()
                            )
                        } catch (t: Throwable) {
                            descriptors.release()
                            throw t
                        } finally {
                            gray.release()
                            keypoints.release()
                        }
                    }
                } finally {
                    source.release()
                }
                markFrameProcessed(index)
                onProgress(
                    CameraTrackingProgress(
                        frameIndex = index + 1,
                        frameCount = frameCount,
                        phase = "Analyzing candidate frames",
                        overallFraction = 0.25f * (sampleIndex + 1) / orderedCandidateFrames.size,
                        completedFrames = processedFrameCount,
                        totalFrames = frameCount
                    )
                )
            }

            val coarseMatchSupport = linkedMapOf<Pair<Int, Int>, Int>()
            coarsePairCandidates.forEachIndexed { candidateIndex, pair ->
                currentCoroutineContext().ensureActive()
                val first = cache[pair.first]
                val second = cache[pair.second]
                if (first != null && second != null) {
                    val firstCoarse = coarseDescriptorCache.getOrPut(pair.first) {
                        coarseDescriptorSubset(first, INITIAL_PAIR_COARSE_DESCRIPTOR_LIMIT)
                    }
                    val secondCoarse = coarseDescriptorCache.getOrPut(pair.second) {
                        coarseDescriptorSubset(second, INITIAL_PAIR_COARSE_DESCRIPTOR_LIMIT)
                    }
                    coarseMatchSupport[pair] = matcher.match(firstCoarse, secondCoarse).size
                }
                if ((candidateIndex + 1) % 16 == 0 || candidateIndex + 1 == coarsePairCandidates.size) {
                    onProgress(
                        CameraTrackingProgress(
                            frameIndex = frameCount,
                            frameCount = frameCount,
                            phase = "Ranking seed pairs by coarse match support (${candidateIndex + 1}/${coarsePairCandidates.size})",
                            overallFraction = 0.25f + 0.02f * (candidateIndex + 1) / coarsePairCandidates.size,
                            completedFrames = processedFrameCount,
                            totalFrames = frameCount
                        )
                    )
                }
            }
            val pairCandidates = InitialPairSelector.correspondenceRankedCandidates(
                candidatePairs = coarsePairCandidates,
                temporalFallbackPairs = temporalPairCandidates,
                matchSupportByPair = coarseMatchSupport,
                maximumCandidates = INITIAL_PAIR_GEOMETRY_LIMIT,
                correspondenceRankBudget = INITIAL_PAIR_CORRESPONDENCE_RANK_LIMIT
            )

            val evaluations = mutableListOf<InitialPairEvaluation>()
            pairCandidates.forEachIndexed { candidateIndex, pair ->
                currentCoroutineContext().ensureActive()
                val first = cache[pair.first]
                val second = cache[pair.second]
                if (first != null && second != null) {
                    evaluations += OpenCvInitialPairEvaluator.evaluate(
                        firstFrame = pair.first,
                        secondFrame = pair.second,
                        firstKeypoints = first.keypoints,
                        firstDescriptors = first.descriptors,
                        secondKeypoints = second.keypoints,
                        secondDescriptors = second.descriptors,
                        cameraMatrix = cameraMatrix,
                        matcher = matcher,
                        frameWidth = minOf(first.width, second.width),
                        frameHeight = minOf(first.height, second.height)
                    )
                }
                onProgress(
                    CameraTrackingProgress(
                        frameIndex = frameCount,
                        frameCount = frameCount,
                        phase = "Evaluating camera-pair geometry (${candidateIndex + 1}/${pairCandidates.size})",
                        overallFraction = 0.27f + 0.085f * (candidateIndex + 1) / pairCandidates.size,
                        completedFrames = processedFrameCount,
                        totalFrames = frameCount
                    )
                )
            }

            // GLOMAP-style global rotation averaging over the verified temporal view graph.
            // Keep this global solution separate from incremental PnP poses until global
            // positioning can jointly estimate translations in the same coordinate gauge.
            val viewGraphRotation = GlobalViewGraphSolver.solve(
                evaluations = evaluations,
                policy = InitialPairPolicy()
            )
            onProgress(
                CameraTrackingProgress(
                    frameIndex = frameCount,
                    frameCount = frameCount,
                    phase = "Global rotation averaging · view graph",
                    overallFraction = 0.355f,
                    completedFrames = processedFrameCount,
                    totalFrames = frameCount
                )
            )

            val selection = InitialPairSelector.select(
                evaluations.map { it.candidate },
                policy = InitialPairPolicy.initialization()
            )
            val selected = selection.selected
                ?: throw IllegalStateException(
                    "No reliable 3D initialization pair: ${selection.reason}. Try footage with visible texture and sideways camera movement."
                )
            val firstFrame = cache[selected.firstFrame]
                ?: error("Selected first frame was not retained.")
            val secondFrame = cache[selected.secondFrame]
                ?: error("Selected second frame was not retained.")

            onProgress(
                CameraTrackingProgress(
                    selected.secondFrame + 1,
                    frameCount,
                    "Initializing 3D camera map",
                    overallFraction = 0.36f,
                    completedFrames = processedFrameCount,
                    totalFrames = frameCount
                )
            )
            val geometry = PairGeometry(
                matches = selected.verifiedMatches,
                rotation = selected.relativeRotation
                    ?: throw IllegalStateException("The selected pair has no recovered rotation."),
                translation = selected.relativeTranslationDirection
                    ?: throw IllegalStateException("The selected pair has no recovered translation."),
                inlierCount = selected.inlierCount
            )
            val matchedFirstIndices = HashSet<Int>()
            val matchedSecondIndices = HashSet<Int>()
            val validCorrespondences = geometry.matches.all { (firstIndex, secondIndex) ->
                firstIndex in firstFrame.keypoints.indices &&
                    secondIndex in secondFrame.keypoints.indices &&
                    matchedFirstIndices.add(firstIndex) && matchedSecondIndices.add(secondIndex)
            }
            val validRelativePose = geometry.rotation.size == 9 &&
                geometry.rotation.all { it.isFinite() } &&
                geometry.translation.size == 3 && geometry.translation.all { it.isFinite() } &&
                geometry.translation.any { kotlin.math.abs(it) > 1e-12 }
            if (geometry.matches.size != geometry.inlierCount || geometry.inlierCount < 100 ||
                !validCorrespondences || !validRelativePose
            ) {
                throw IllegalStateException("Initial pair failed correspondence or relative-pose validation.")
            }

            val trackBuilder = FeatureTrackBuilder()
            val verifiedMatchesByFramePair = linkedMapOf<Pair<Int, Int>, LinkedHashSet<Pair<Int, Int>>>()
            val matchedObservationsByNode = HashMap<Long, CameraFeatureObservation>()
            fun featureNode(frameIndex: Int, featureIndex: Int): Long =
                (frameIndex.toLong() shl 32) or (featureIndex.toLong() and 0xffffffffL)

            fun rememberVerifiedPair(
                firstFrame: Int,
                secondFrame: Int,
                firstKeypoints: Array<KeyPoint>,
                secondKeypoints: Array<KeyPoint>,
                matches: List<Pair<Int, Int>>
            ) {
                if (firstFrame == secondFrame) return
                val frames = minOf(firstFrame, secondFrame) to maxOf(firstFrame, secondFrame)
                val storedMatches = verifiedMatchesByFramePair.getOrPut(frames) { linkedSetOf() }
                val chronological = firstFrame < secondFrame
                for ((firstIndex, secondIndex) in matches) {
                    if (firstIndex !in firstKeypoints.indices || secondIndex !in secondKeypoints.indices) continue
                    val firstPoint = firstKeypoints[firstIndex].pt
                    val secondPoint = secondKeypoints[secondIndex].pt
                    matchedObservationsByNode[featureNode(firstFrame, firstIndex)] =
                        CameraFeatureObservation(firstFrame, firstPoint.x, firstPoint.y, firstIndex)
                    matchedObservationsByNode[featureNode(secondFrame, secondIndex)] =
                        CameraFeatureObservation(secondFrame, secondPoint.x, secondPoint.y, secondIndex)
                    storedMatches += if (chronological) firstIndex to secondIndex
                    else secondIndex to firstIndex
                }
            }

            fun localCompletionPairs(): List<ColmapTrackCompleter.PairMatches> =
                verifiedMatchesByFramePair.map { (frames, matches) ->
                    ColmapTrackCompleter.PairMatches(frames.first, frames.second, matches.toList())
                }

            fun localMergePairs(): List<ColmapTrackMerger.PairMatches> =
                verifiedMatchesByFramePair.map { (frames, matches) ->
                    ColmapTrackMerger.PairMatches(frames.first, frames.second, matches.toList())
                }

            trackBuilder.seedPair(
                selected.firstFrame,
                selected.secondFrame,
                firstFrame.keypoints,
                secondFrame.keypoints,
                geometry.matches
            )
            rememberVerifiedPair(
                selected.firstFrame, selected.secondFrame,
                firstFrame.keypoints, secondFrame.keypoints, geometry.matches
            )

            val poseByFrame = arrayOfNulls<CameraPose>(frameCount)
            val referencePose = CameraPose(
                selected.firstFrame, identity3(), doubleArrayOf(0.0, 0.0, 0.0),
                CameraPoseStatus.REFERENCE
            )
            val secondPose = CameraPose(
                selected.secondFrame, geometry.rotation, geometry.translation,
                CameraPoseStatus.RELATIVE_ESTIMATE
            )
            poseByFrame[selected.firstFrame] = referencePose
            poseByFrame[selected.secondFrame] = secondPose

            var tracks = trackBuilder.build()
            val initialReconstruction = SfMTriangulator.triangulate(
                tracks, listOf(referencePose, secondPose), cameraMatrix
            )
            require(initialReconstruction.landmarks.size >= 30) {
                "Initial 3D map is too small (${initialReconstruction.landmarks.size} landmarks). " +
                    "Use footage with stronger camera translation and more static scene texture."
            }
            val landmarksByTrack = linkedMapOf<Int, CameraLandmark>()
            initialReconstruction.landmarks.forEach { landmarksByTrack[it.trackId] = it }
            val modifiedPointTrackIds = initialReconstruction.landmarks
                .mapTo(linkedSetOf()) { it.trackId }

            val excludedObservationsByTrack = linkedMapOf<Int, MutableSet<CameraFeatureObservation>>()
            val deletedTrackObservationCountsByTrack = linkedMapOf<Int, Int>()
            fun filteredMapTrack(track: CameraFeatureTrack): CameraFeatureTrack? {
                val excluded = excludedObservationsByTrack[track.id].orEmpty()
                if (excluded.isEmpty()) return track
                val retained = track.observations.filterNot { it in excluded }
                return retained.takeIf { it.size >= 2 }?.let { track.copy(observations = it) }
            }

            fun currentMapTracks(): List<CameraFeatureTrack> =
                trackBuilder.build(trackIds = landmarksByTrack.keys).mapNotNull(::filteredMapTrack)

            fun recordModifiedTracksFromFrame(frameIndex: Int) {
                currentMapTracks().asSequence()
                    .filter { track -> track.observations.any { it.frameIndex == frameIndex } }
                    .mapTo(modifiedPointTrackIds) { it.id }
            }

            fun currentFeatureTracks(): List<CameraFeatureTrack> =
                trackBuilder.build().mapNotNull { track ->
                    val previousRegisteredCount = deletedTrackObservationCountsByTrack[track.id]
                    if (track.id !in landmarksByTrack && previousRegisteredCount != null) {
                        val registeredCount = track.observations.count { observation ->
                            poseByFrame.getOrNull(observation.frameIndex)?.status?.let { status ->
                                status != CameraPoseStatus.HELD_LAST_VALID &&
                                    status != CameraPoseStatus.UNREGISTERED
                            } == true
                        }
                        if (registeredCount <= previousRegisteredCount) return@mapNotNull null
                        deletedTrackObservationCountsByTrack.remove(track.id)
                    }
                    filteredMapTrack(track)
                }

            fun applyObservationFilter(
                affectedFrameIds: Set<Int>,
                changedTrackIds: Set<Int>,
                intrinsics: PinholeIntrinsics
            ): Int {
                val filtered = ColmapObservationFilter.filter(
                    tracksById = currentMapTracks().associateBy { it.id },
                    landmarksByTrackId = landmarksByTrack,
                    posesByFrame = poseByFrame.filterNotNull().associateBy { it.frameIndex },
                    intrinsics = intrinsics,
                    affectedFrameIds = affectedFrameIds,
                    changedTrackIds = changedTrackIds
                )
                val registeredFrameIds = poseByFrame.filterNotNull().asSequence()
                    .filter { pose ->
                        pose.status != CameraPoseStatus.HELD_LAST_VALID &&
                            pose.status != CameraPoseStatus.UNREGISTERED
                    }
                    .mapTo(linkedSetOf()) { it.frameIndex }
                val rawDeletedTrackRegisteredCounts = trackBuilder
                    .build(trackIds = filtered.deletedTrackIds)
                    .associate { track ->
                        track.id to track.observations.count { it.frameIndex in registeredFrameIds }
                    }
                for (trackId in filtered.deletedTrackIds) {
                    deletedTrackObservationCountsByTrack[trackId] =
                        rawDeletedTrackRegisteredCounts[trackId] ?: 0
                    landmarksByTrack.remove(trackId)
                }
                for ((trackId, observations) in filtered.removedObservationsByTrack) {
                    excludedObservationsByTrack.getOrPut(trackId) { linkedSetOf() }
                        .addAll(observations)
                }
                filtered.landmarksByTrackId.forEach { (trackId, landmark) ->
                    landmarksByTrack[trackId] = landmark
                }
                return filtered.filteredObservationCount
            }

            var filterIntrinsics = PinholeIntrinsics(
                fx = cameraMatrix.get(0, 0)?.firstOrNull() ?: Double.NaN,
                skew = cameraMatrix.get(0, 1)?.firstOrNull() ?: Double.NaN,
                cx = cameraMatrix.get(0, 2)?.firstOrNull() ?: Double.NaN,
                fy = cameraMatrix.get(1, 1)?.firstOrNull() ?: Double.NaN,
                cy = cameraMatrix.get(1, 2)?.firstOrNull() ?: Double.NaN
            )

            fun cameraCenter(pose: CameraPose): DoubleArray {
                val r = pose.rotation
                val t = pose.translation
                return doubleArrayOf(
                    -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
                    -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
                    -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])
                )
            }

            fun adjustLocalBundle(
                referenceFrameId: Int,
                lossMode: ColmapIterativeLocalRefinement.LossMode
            ): ColmapIterativeLocalRefinement.Report? {
                val validPoses = poseByFrame.filterNotNull().filter { pose ->
                    pose.status != CameraPoseStatus.HELD_LAST_VALID &&
                        pose.status != CameraPoseStatus.UNREGISTERED
                }
                val centersByFrame = validPoses.associate { pose ->
                    pose.frameIndex to cameraCenter(pose)
                }
                val mapTracks = currentMapTracks()
                val referenceTracks = mapTracks.filter { track ->
                    track.observations.any { it.frameIndex == referenceFrameId }
                }
                val referencePoints = referenceTracks.associate { track ->
                    val landmark = landmarksByTrack.getValue(track.id)
                    track.id to doubleArrayOf(
                        landmark.position.x, landmark.position.y, landmark.position.z
                    )
                }
                val selection = ColmapLocalBundleSelector.select(
                    referenceFrameId = referenceFrameId,
                    cameraCentersByFrame = centersByFrame,
                    tracks = referenceTracks,
                    pointsByTrackId = referencePoints
                )
                val localFrameIds = selection.frameIds.toSet()

                val localTracks = ColmapLocalBundleTrackSelector.select(mapTracks, localFrameIds)
                val localTrackIds = localTracks.mapTo(linkedSetOf()) { it.id }
                val localObservationCount = localTracks.sumOf { track ->
                    track.observations.count { it.frameIndex in localFrameIds }
                }
                if (!ColmapLocalBundleTrackSelector.meetsBundleAdjustmentMinimum(
                        localFrameIds = localFrameIds,
                        localTracks = localTracks
                    )
                ) return null

                val localCenters = centersByFrame.filterKeys { it in localFrameIds }
                val localRotations = validPoses.asSequence()
                    .filter { it.frameIndex in localFrameIds }
                    .associate { it.frameIndex to it.rotation }
                val localPoints = localTrackIds.associateWith { trackId ->
                    val landmark = landmarksByTrack.getValue(trackId)
                    doubleArrayOf(landmark.position.x, landmark.position.y, landmark.position.z)
                }
                val variablePointTrackIds = mapTracks.asSequence()
                    .filter { it.id in modifiedPointTrackIds && it.length <= 15 }
                    .map { it.id }
                    .toSet()
                val variableBundleTrackIds = variablePointTrackIds.intersect(localTrackIds)
                val candidate = try {
                    JointCameraBundleAdjuster().solve(
                        frameCentersById = localCenters,
                        rotationsByFrame = localRotations,
                        pointsByTrackId = localPoints,
                        tracks = localTracks,
                        cameraMatrix = cameraMatrix,
                        imageWidth = firstFrame.width,
                        imageHeight = firstFrame.height,
                        maxIterations = 8,
                        numThreads = 1,
                        optimizeFocalLength = false,
                        optimizePointTrackIds = variableBundleTrackIds,
                        reprojectionLoss = when (lossMode) {
                            ColmapIterativeLocalRefinement.LossMode.SOFT_L1 ->
                                JointCameraBundleAdjuster.ReprojectionLoss.SOFT_L1
                            ColmapIterativeLocalRefinement.LossMode.TRIVIAL ->
                                JointCameraBundleAdjuster.ReprojectionLoss.TRIVIAL
                        }
                    )
                } catch (_: LinkageError) {
                    null
                } catch (_: RuntimeException) {
                    null
                } ?: return null

                if (!SolverObjective.didNotIncrease(candidate.initialCost, candidate.finalCost)) {
                    return null
                }
                for ((frameId, center) in candidate.frameCentersById) {
                    val oldPose = poseByFrame.getOrNull(frameId) ?: continue
                    val rotation = candidate.rotationsByFrame[frameId] ?: continue
                    val translation = doubleArrayOf(
                        -(rotation[0] * center[0] + rotation[1] * center[1] + rotation[2] * center[2]),
                        -(rotation[3] * center[0] + rotation[4] * center[1] + rotation[5] * center[2]),
                        -(rotation[6] * center[0] + rotation[7] * center[1] + rotation[8] * center[2])
                    )
                    poseByFrame[frameId] = oldPose.copy(
                        rotation = rotation.copyOf(),
                        translation = translation
                    )
                }

                for ((trackId, point) in candidate.pointsByTrackId) {
                    val oldLandmark = landmarksByTrack[trackId] ?: continue
                    landmarksByTrack[trackId] = oldLandmark.copy(
                        position = Point3(point[0], point[1], point[2])
                    )
                }

                val posesAfterAdjustment = validPoses.map { pose ->
                    poseByFrame.getOrNull(pose.frameIndex) ?: pose
                }
                val mergeResult = ColmapTrackMerger.mergeTracks(
                    tracks = mapTracks,
                    landmarks = landmarksByTrack.values.toList(),
                    pairMatches = localMergePairs(),
                    poses = posesAfterAdjustment,
                    intrinsics = filterIntrinsics,
                    candidateTrackIds = variablePointTrackIds
                )
                landmarksByTrack.clear()
                mergeResult.landmarks.forEach { landmark ->
                    landmarksByTrack[landmark.trackId] = landmark
                }
                modifiedPointTrackIds.removeAll(mergeResult.removedTrackIds)
                val remainingMapTrackIds = mergeResult.tracks.mapTo(HashSet<Int>()) { it.id }
                modifiedPointTrackIds.retainAll(remainingMapTrackIds)
                modifiedPointTrackIds.addAll(mergeResult.mergedLandmarks.map { it.trackId })

                val tracksBeforeCompletion = mergeResult.tracks.associateBy { it.id }
                val completion = ColmapTrackCompleter.completeTracks(
                    tracks = mergeResult.tracks,
                    landmarks = mergeResult.landmarks,
                    pairMatches = localCompletionPairs(),
                    activeFrames = posesAfterAdjustment.mapTo(linkedSetOf()) { it.frameIndex },
                    poses = posesAfterAdjustment,
                    intrinsics = filterIntrinsics,
                    featureAt = { frameIndex, featureIndex ->
                        matchedObservationsByNode[featureNode(frameIndex, featureIndex)]
                    },
                    candidateTrackIds = variablePointTrackIds,
                    completionFrameId = referenceFrameId
                )
                val completedTrackIds = completion.tracks.asSequence()
                    .filter { track -> track.length > (tracksBeforeCompletion[track.id]?.length ?: 0) }
                    .mapTo(linkedSetOf()) { it.id }
                modifiedPointTrackIds.addAll(completedTrackIds)
                val replacedTrackIds = mapTracks.mapTo(linkedSetOf()) { it.id }
                trackBuilder.replaceMapTracks(
                    replacedTrackIds = replacedTrackIds,
                    updatedTracks = completion.tracks,
                    removedTrackIds = mergeResult.removedTrackIds,
                    activeFrameIndex = referenceFrameId
                )
                mergeResult.removedTrackIds.forEach(deletedTrackObservationCountsByTrack::remove)

                val changedTrackIds = candidate.pointsByTrackId.keys.toMutableSet().apply {
                    addAll(modifiedPointTrackIds)
                }
                val filteredObservationCount = applyObservationFilter(
                    affectedFrameIds = localFrameIds,
                    changedTrackIds = changedTrackIds,
                    intrinsics = filterIntrinsics
                )
                modifiedPointTrackIds.retainAll(landmarksByTrack.keys)
                return ColmapIterativeLocalRefinement.Report(
                    adjustedObservations = localObservationCount,
                    mergedObservations = mergeResult.mergedObservationCount,
                    completedObservations = completion.addedObservations,
                    filteredObservations = filteredObservationCount
                )
            }

            fun refineLocalBundle(referenceFrameId: Int): Boolean {
                val result = ColmapIterativeLocalRefinement.run { lossMode ->
                    adjustLocalBundle(referenceFrameId, lossMode)
                }
                modifiedPointTrackIds.clear()
                return result.iterations.isNotEmpty()
            }

            var previousDescriptors = secondFrame.descriptors.clone()
            var previousKeypoints = secondFrame.keypoints
            var totalMatches = geometry.matches.size
            var totalCorrespondences = geometry.matches.size
            var totalInliers = geometry.inlierCount
            var registeredFrames = 2
            var acceptedRegistrationsSinceMapUpdate = 0
            var focalLengthHasBeenEstimated = cameraFocalLengthHasPrior
            var estimatedCameraIntrinsics: CameraIntrinsics? = null
            val diagnostics = MutableList(frameCount) { index ->
                FrameTrackingDiagnostic(index, 0, 0, 0, "Not registered")
            }
            diagnostics[selected.firstFrame] = FrameTrackingDiagnostic(
                selected.firstFrame, firstFrame.keypoints.size, 0, 0, "Initialization reference"
            )
            diagnostics[selected.secondFrame] = FrameTrackingDiagnostic(
                selected.secondFrame, secondFrame.keypoints.size,
                geometry.matches.size, geometry.inlierCount, "Initial pair reconstructed"
            )

            val registrationTrials = ColmapRegistrationTrialLedger(frameCount)
            val failedRegistrationFrames = linkedSetOf<Int>()

            fun registerMapFrame(
                frameIndex: Int,
                frameWidth: Int,
                frameHeight: Int,
                featureTracks: List<CameraFeatureTrack>,
                landmarks: Collection<CameraLandmark>,
                cameraMatrix: Mat
            ): CameraRegistrationResult {
                val shouldEstimateFocalLength = !focalLengthHasBeenEstimated
                val registration = IncrementalCameraRegistrar.register(
                    frameIndex = frameIndex,
                    frameWidth = frameWidth,
                    frameHeight = frameHeight,
                    featureTracks = featureTracks,
                    landmarks = landmarks,
                    cameraMatrix = cameraMatrix,
                    estimateFocalLength = shouldEstimateFocalLength
                )
                if (!registration.accepted || !shouldEstimateFocalLength) return registration
                val estimatedIntrinsics = registration.refinedIntrinsics
                    ?: return registration.copy(
                        accepted = false,
                        pose = null,
                        reason = "Accepted P4PF pose did not return its focal estimate"
                    )
                if (cameraMatrix.put(0, 0, *estimatedIntrinsics.toRowMajorMatrix()) != 9) {
                    return registration.copy(
                        accepted = false,
                        pose = null,
                        reason = "Could not commit the accepted P4PF focal estimate"
                    )
                }
                filterIntrinsics = estimatedIntrinsics
                estimatedCameraIntrinsics = CameraIntrinsics(
                    fx = estimatedIntrinsics.fx,
                    skew = estimatedIntrinsics.skew,
                    cx = estimatedIntrinsics.cx,
                    fy = estimatedIntrinsics.fy,
                    cy = estimatedIntrinsics.cy
                )
                focalLengthHasBeenEstimated = true
                return registration
            }

            suspend fun retryFailedFramesAfterMapUpdate(imageWidth: Int, imageHeight: Int) {
                // The streaming Android path only retries frames whose features have
                // already entered the persistent track graph; it does not cache all frames.
                failedRegistrationFrames.removeAll { frameIndex ->
                    poseByFrame[frameIndex] != null ||
                        registrationTrials[frameIndex] >=
                        ColmapNextImageSelector.DEFAULT_MAX_REGISTRATION_TRIALS
                }
                if (failedRegistrationFrames.isEmpty()) return

                val mapTracks = currentMapTracks()
                val visibleObservationsByFrame = HashMap<Int, MutableList<CameraFeatureObservation>>()
                mapTracks.forEach { track ->
                    track.observations.forEach { observation ->
                        if (observation.frameIndex in failedRegistrationFrames) {
                            visibleObservationsByFrame.getOrPut(observation.frameIndex) { mutableListOf() }
                                .add(observation)
                        }
                    }
                }
                val observationCountsByFrame = HashMap<Int, Int>()
                currentFeatureTracks().forEach { track ->
                    track.observations.forEach { observation ->
                        if (observation.frameIndex in failedRegistrationFrames) {
                            observationCountsByFrame[observation.frameIndex] =
                                (observationCountsByFrame[observation.frameIndex] ?: 0) + 1
                        }
                    }
                }
                val candidates = failedRegistrationFrames.map { frameIndex ->
                    ColmapNextImageSelector.Candidate(
                        frameIndex = frameIndex,
                        visibleObservations = visibleObservationsByFrame[frameIndex].orEmpty(),
                        totalObservations = observationCountsByFrame[frameIndex] ?: 0,
                        registrationTrials = registrationTrials[frameIndex]
                    )
                }
                val rankedCandidates = ColmapNextImageSelector.rank(
                    candidates = candidates,
                    imageWidth = imageWidth,
                    imageHeight = imageHeight,
                    options = ColmapNextImageSelector.Options(
                        minimumVisiblePoints = CameraRegistrationPolicy.MINIMUM_CORRESPONDENCES
                    )
                ).take(ColmapNextImageSelector.ANDROID_MAX_CANDIDATES_PER_BATCH)
                val landmarks = landmarksByTrack.values.toList()
                for (candidate in rankedCandidates) {
                    currentCoroutineContext().ensureActive()
                    val candidateFrame = candidate.frameIndex
                    val registrationTrial = registrationTrials.beginTrial(candidateFrame)
                    if (registrationTrial == null) {
                        failedRegistrationFrames.remove(candidateFrame)
                        continue
                    }
                    val registration = registerMapFrame(
                        frameIndex = candidateFrame,
                        frameWidth = imageWidth,
                        frameHeight = imageHeight,
                        featureTracks = mapTracks,
                        landmarks = landmarks,
                        cameraMatrix = cameraMatrix
                    )
                    totalCorrespondences += registration.correspondenceCount
                    val previousDiagnostic = diagnostics[candidateFrame]
                    if (registration.accepted && registration.pose != null) {
                        poseByFrame[candidateFrame] = registration.pose
                        recordModifiedTracksFromFrame(candidateFrame)
                        totalInliers += registration.inlierCount
                        registeredFrames++
                        acceptedRegistrationsSinceMapUpdate++
                        failedRegistrationFrames.remove(candidateFrame)
                        diagnostics[candidateFrame] = previousDiagnostic.copy(
                            inlierFeatures = registration.inlierCount,
                            status = "PnP registered after map update (${registration.inlierCount} inliers, " +
                                "median ${"%.2f".format(registration.medianReprojectionErrorPx)} px)"
                        )
                        return
                    }
                    diagnostics[candidateFrame] = previousDiagnostic.copy(
                        inlierFeatures = registration.inlierCount,
                        status = "Map-based retry $registrationTrial/" +
                            "${ColmapNextImageSelector.DEFAULT_MAX_REGISTRATION_TRIALS}: ${registration.reason}"
                    )
                    if (registrationTrial >=
                        ColmapNextImageSelector.DEFAULT_MAX_REGISTRATION_TRIALS
                    ) failedRegistrationFrames.remove(candidateFrame)
                }
            }

            try {
                for (frameIndex in selected.secondFrame + 1 until frameCount) {
                    currentCoroutineContext().ensureActive()
                    onProgress(
                        CameraTrackingProgress(
                            frameIndex,
                            frameCount,
                            "Registering camera frame",
                            overallFraction = 0.36f + 0.32f *
                                (frameIndex - selected.secondFrame).toFloat() /
                                (frameCount - selected.secondFrame).coerceAtLeast(1),
                            completedFrames = processedFrameCount,
                            totalFrames = frameCount
                        )
                    )
                    val source = frameProvider(frameIndex)
                        ?: throw IllegalStateException("Unable to decode frame ${frameIndex + 1}.")
                    val gray = withSourceReleaseOnFailure(source) { ownedSource ->
                        referenceFrameSize = TrackingFrameDimensions.validate(
                            referenceFrameSize, ownedSource.cols(), ownedSource.rows()
                        )
                        toGray(ownedSource)
                    }
                    val keypointsMat = org.opencv.core.MatOfKeyPoint()
                    val descriptors = Mat()
                    try {
                        val detectionMask = Mat()
                        try {
                            featureExtractor.detectAndCompute(gray, detectionMask, keypointsMat, descriptors)
                        } finally {
                            detectionMask.release()
                        }
                        val currentKeypoints = keypointsMat.toArray()
                        val matches = if (!previousDescriptors.empty() && !descriptors.empty()) {
                            verifiedMatches(
                                previousDescriptors, previousKeypoints,
                                descriptors, currentKeypoints, cameraMatrix, matcher
                            )
                        } else emptyList()

                        rememberVerifiedPair(
                            frameIndex - 1, frameIndex,
                            previousKeypoints, currentKeypoints, matches
                        )

                        totalMatches += matches.size
                        if (matches.isEmpty()) {
                            trackBuilder.breakContinuity()
                        } else {
                            trackBuilder.addInlierMatches(
                                frameIndex, previousKeypoints, currentKeypoints, matches
                            )
                        }

                        checkNotNull(registrationTrials.beginTrial(frameIndex)) {
                            "Forward frame exceeded its registration trial limit before its first attempt"
                        }
                        var registration = registerMapFrame(
                            frameIndex = frameIndex,
                            frameWidth = source.cols(),
                            frameHeight = source.rows(),
                            featureTracks = currentMapTracks(),
                            landmarks = landmarksByTrack.values.toList(),
                            cameraMatrix = cameraMatrix
                        )
                        // If adjacent-frame tracking lost the seed-map features, try
                        // re-localizing against the initialization keyframe before rejecting
                        // this camera. This is a recovery attempt, not a relaxed PnP gate.
                        if (!registration.accepted &&
                            !secondFrame.descriptors.empty() && !descriptors.empty()
                        ) {
                            val anchorMatches = verifiedMatches(
                                secondFrame.descriptors, secondFrame.keypoints,
                                descriptors, currentKeypoints, cameraMatrix, matcher
                            )
                            rememberVerifiedPair(
                                selected.secondFrame, frameIndex,
                                secondFrame.keypoints, currentKeypoints, anchorMatches
                            )
                            if (anchorMatches.isNotEmpty() &&
                                registrationTrials.beginTrial(frameIndex) != null
                            ) {
                                trackBuilder.addInlierMatchesFromFrame(
                                    selected.secondFrame, frameIndex,
                                    secondFrame.keypoints, currentKeypoints, anchorMatches
                                )
                                val recovered = registerMapFrame(
                                    frameIndex = frameIndex,
                                    frameWidth = source.cols(),
                                    frameHeight = source.rows(),
                                    featureTracks = currentMapTracks(),
                                    landmarks = landmarksByTrack.values.toList(),
                                    cameraMatrix = cameraMatrix
                                )
                                if (recovered.accepted ||
                                    recovered.correspondenceCount > registration.correspondenceCount
                                ) registration = recovered
                            }
                        }
                        // Second-keyframe recovery can fail when a frame is much
                        // closer to the first seed. Try that seed only after the
                        // adjacent-frame and second-seed attempts have failed.
                        if (!registration.accepted &&
                            !firstFrame.descriptors.empty() && !descriptors.empty()
                        ) {
                            val firstAnchorMatches = verifiedMatches(
                                firstFrame.descriptors, firstFrame.keypoints,
                                descriptors, currentKeypoints, cameraMatrix, matcher
                            )
                            rememberVerifiedPair(
                                selected.firstFrame, frameIndex,
                                firstFrame.keypoints, currentKeypoints, firstAnchorMatches
                            )
                            if (firstAnchorMatches.isNotEmpty() &&
                                registrationTrials.beginTrial(frameIndex) != null
                            ) {
                                trackBuilder.addInlierMatchesFromFrame(
                                    selected.firstFrame, frameIndex,
                                    firstFrame.keypoints, currentKeypoints, firstAnchorMatches
                                )
                                val recoveredFromFirst = registerMapFrame(
                                    frameIndex = frameIndex,
                                    frameWidth = source.cols(),
                                    frameHeight = source.rows(),
                                    featureTracks = currentMapTracks(),
                                    landmarks = landmarksByTrack.values.toList(),
                                    cameraMatrix = cameraMatrix
                                )
                                if (recoveredFromFirst.accepted ||
                                    recoveredFromFirst.correspondenceCount > registration.correspondenceCount
                                ) registration = recoveredFromFirst
                            }
                        }
                        totalCorrespondences += registration.correspondenceCount
                        if (registration.accepted && registration.pose != null) {
                            poseByFrame[frameIndex] = registration.pose
                            recordModifiedTracksFromFrame(frameIndex)
                            totalInliers += registration.inlierCount
                            registeredFrames++
                            acceptedRegistrationsSinceMapUpdate++
                            val refinementFallbackNote = if (
                                registration.reason.contains("used RANSAC PnP seed")
                            ) " · RANSAC PnP seed used" else ""
                            diagnostics[frameIndex] = FrameTrackingDiagnostic(
                                frameIndex, currentKeypoints.size, matches.size,
                                registration.inlierCount,
                                "PnP registered (${registration.inlierCount} inliers, " +
                                    "median ${"%.2f".format(registration.medianReprojectionErrorPx)} px" +
                                    "$refinementFallbackNote)"
                            )

                            // Bound local map work to one update per two accepted cameras.
                            if (acceptedRegistrationsSinceMapUpdate >= 2) {
                                tracks = currentFeatureTracks()
                                val validPoses = poseByFrame.filterNotNull()
                                val newTracks = tracks.filter { track ->
                                    track.id !in landmarksByTrack &&
                                        track.observations.any { it.frameIndex == frameIndex }
                                }
                                if (newTracks.isNotEmpty()) {
                                    val update = SfMTriangulator.triangulate(
                                        newTracks, validPoses, cameraMatrix
                                    )
                                    update.landmarks.forEach { landmark ->
                                        landmarksByTrack[landmark.trackId] = landmark
                                        modifiedPointTrackIds += landmark.trackId
                                    }
                                }
                                acceptedRegistrationsSinceMapUpdate = 0
                                currentCoroutineContext().ensureActive()
                                refineLocalBundle(frameIndex)
                                retryFailedFramesAfterMapUpdate(firstFrame.width, firstFrame.height)
                            }
                        } else {
                            failedRegistrationFrames += frameIndex
                            diagnostics[frameIndex] = FrameTrackingDiagnostic(
                                frameIndex, currentKeypoints.size, matches.size,
                                registration.inlierCount, registration.reason
                            )
                        }

                        previousDescriptors.release()
                        previousDescriptors = descriptors.clone()
                        previousKeypoints = currentKeypoints
                    } finally {
                        source.release()
                        gray.release()
                        keypointsMat.release()
                        descriptors.release()
                    }
                    markFrameProcessed(frameIndex)
                    onProgress(
                        CameraTrackingProgress(
                            frameIndex + 1,
                            frameCount,
                            "Registered ${registeredFrames} cameras",
                            overallFraction = 0.36f + 0.32f *
                                (frameIndex + 1 - selected.secondFrame).toFloat() /
                                (frameCount - selected.secondFrame).coerceAtLeast(1),
                            completedFrames = processedFrameCount,
                            totalFrames = frameCount
                        )
                    )
                }
            } finally {
                previousDescriptors.release()
            }

            // Run the same map-based registration backwards from the second
            // initialization camera. This fills frames before the selected pair
            // and the interval between the pair without inventing translations.
            trackBuilder.resetToFrame(selected.secondFrame)
            var reverseDescriptors = secondFrame.descriptors.clone()
            var reverseKeypoints = secondFrame.keypoints
            try {
                for (frameIndex in selected.secondFrame - 1 downTo 0) {
                    currentCoroutineContext().ensureActive()
                    onProgress(
                        CameraTrackingProgress(
                            frameIndex + 1,
                            frameCount,
                            "Registering earlier frames",
                            overallFraction = 0.68f + 0.22f *
                                (selected.secondFrame - frameIndex).toFloat() /
                                selected.secondFrame.coerceAtLeast(1),
                            completedFrames = processedFrameCount,
                            totalFrames = frameCount
                        )
                    )
                    val source = frameProvider(frameIndex)
                        ?: throw IllegalStateException("Unable to decode frame ${frameIndex + 1}.")
                    val gray = withSourceReleaseOnFailure(source) { ownedSource ->
                        referenceFrameSize = TrackingFrameDimensions.validate(
                            referenceFrameSize, ownedSource.cols(), ownedSource.rows()
                        )
                        toGray(ownedSource)
                    }
                    val keypointsMat = org.opencv.core.MatOfKeyPoint()
                    val descriptors = Mat()
                    try {
                        val detectionMask = Mat()
                        try {
                            featureExtractor.detectAndCompute(gray, detectionMask, keypointsMat, descriptors)
                        } finally {
                            detectionMask.release()
                        }
                        val currentKeypoints = keypointsMat.toArray()
                        val matches = if (!reverseDescriptors.empty() && !descriptors.empty()) {
                            verifiedMatches(
                                reverseDescriptors, reverseKeypoints,
                                descriptors, currentKeypoints, cameraMatrix, matcher
                            )
                        } else emptyList()
                        rememberVerifiedPair(
                            frameIndex + 1, frameIndex,
                            reverseKeypoints, currentKeypoints, matches
                        )
                        totalMatches += matches.size
                        if (matches.isEmpty()) {
                            trackBuilder.breakContinuity()
                        } else {
                            trackBuilder.addInlierMatchesBackward(
                                frameIndex, reverseKeypoints, currentKeypoints, matches
                            )
                        }
                        
                        if (frameIndex == selected.firstFrame) {
                            diagnostics[frameIndex] = FrameTrackingDiagnostic(
                                frameIndex, currentKeypoints.size, matches.size,
                                matches.size, "Initialization reference"
                            )
                        } else {
                            checkNotNull(registrationTrials.beginTrial(frameIndex)) {
                                "Backward frame exceeded its registration trial limit before its first attempt"
                            }
                            var registration = registerMapFrame(
                                frameIndex = frameIndex,
                                frameWidth = source.cols(),
                                frameHeight = source.rows(),
                                featureTracks = currentMapTracks(),
                                landmarks = landmarksByTrack.values.toList(),
                                cameraMatrix = cameraMatrix
                            )
                            // Backward recovery uses the same initialization keyframe,
                            // so a missed adjacent match does not permanently orphan IDs.
                            if (!registration.accepted &&
                                !secondFrame.descriptors.empty() && !descriptors.empty()
                            ) {
                                val anchorMatches = verifiedMatches(
                                    secondFrame.descriptors, secondFrame.keypoints,
                                    descriptors, currentKeypoints, cameraMatrix, matcher
                                )
                                rememberVerifiedPair(
                                    selected.secondFrame, frameIndex,
                                    secondFrame.keypoints, currentKeypoints, anchorMatches
                                )
                                if (anchorMatches.isNotEmpty() &&
                                    registrationTrials.beginTrial(frameIndex) != null
                                ) {
                                    trackBuilder.addInlierMatchesFromFrame(
                                        selected.secondFrame, frameIndex,
                                        secondFrame.keypoints, currentKeypoints, anchorMatches
                                    )
                                    val recovered = registerMapFrame(
                                        frameIndex = frameIndex,
                                        frameWidth = source.cols(),
                                        frameHeight = source.rows(),
                                        featureTracks = currentMapTracks(),
                                        landmarks = landmarksByTrack.values.toList(),
                                        cameraMatrix = cameraMatrix
                                    )
                                    if (recovered.accepted ||
                                        recovered.correspondenceCount > registration.correspondenceCount
                                    ) registration = recovered
                                }
                            }
                            // Second-keyframe recovery can fail when a frame is much
                            // closer to the first seed. Try that seed only after the
                            // adjacent-frame and second-seed attempts have failed.
                            if (!registration.accepted &&
                                !firstFrame.descriptors.empty() && !descriptors.empty()
                            ) {
                                val firstAnchorMatches = verifiedMatches(
                                    firstFrame.descriptors, firstFrame.keypoints,
                                    descriptors, currentKeypoints, cameraMatrix, matcher
                                )
                                rememberVerifiedPair(
                                    selected.firstFrame, frameIndex,
                                    firstFrame.keypoints, currentKeypoints, firstAnchorMatches
                                )
                                if (firstAnchorMatches.isNotEmpty() &&
                                    registrationTrials.beginTrial(frameIndex) != null
                                ) {
                                    trackBuilder.addInlierMatchesFromFrame(
                                        selected.firstFrame, frameIndex,
                                        firstFrame.keypoints, currentKeypoints, firstAnchorMatches
                                    )
                                    val recoveredFromFirst = registerMapFrame(
                                        frameIndex = frameIndex,
                                        frameWidth = source.cols(),
                                        frameHeight = source.rows(),
                                        featureTracks = currentMapTracks(),
                                        landmarks = landmarksByTrack.values.toList(),
                                        cameraMatrix = cameraMatrix
                                    )
                                    if (recoveredFromFirst.accepted ||
                                        recoveredFromFirst.correspondenceCount > registration.correspondenceCount
                                    ) registration = recoveredFromFirst
                                }
                                }
                                totalCorrespondences += registration.correspondenceCount
                            if (registration.accepted && registration.pose != null) {
                                poseByFrame[frameIndex] = registration.pose
                                recordModifiedTracksFromFrame(frameIndex)
                                totalInliers += registration.inlierCount
                                registeredFrames++
                                acceptedRegistrationsSinceMapUpdate++
                                val refinementFallbackNote = if (
                                    registration.reason.contains("used RANSAC PnP seed")
                                ) " · RANSAC PnP seed used" else ""
                                diagnostics[frameIndex] = FrameTrackingDiagnostic(
                                    frameIndex, currentKeypoints.size, matches.size,
                                    registration.inlierCount,
                                    "PnP registered backwards (${registration.inlierCount} inliers)" +
                                        refinementFallbackNote
                                )
                                if (acceptedRegistrationsSinceMapUpdate >= 2) {
                                    tracks = currentFeatureTracks()
                                    val newTracks = tracks.filter { track ->
                                        track.id !in landmarksByTrack &&
                                            track.observations.any { it.frameIndex == frameIndex }
                                    }
                                    if (newTracks.isNotEmpty()) {
                                        val update = SfMTriangulator.triangulate(
                                            newTracks, poseByFrame.filterNotNull(), cameraMatrix
                                        )
                                    update.landmarks.forEach { landmark ->
                                        val old = landmarksByTrack[landmark.trackId]
                                        if (old == null ||
                                            landmark.meanReprojectionErrorPx < old.meanReprojectionErrorPx
                                        ) landmarksByTrack[landmark.trackId] = landmark
                                        modifiedPointTrackIds += landmark.trackId
                                    }
                                    }
                                    acceptedRegistrationsSinceMapUpdate = 0
                                    currentCoroutineContext().ensureActive()
                                    refineLocalBundle(frameIndex)
                                    retryFailedFramesAfterMapUpdate(firstFrame.width, firstFrame.height)
                                }
                            } else {
                                failedRegistrationFrames += frameIndex
                                diagnostics[frameIndex] = FrameTrackingDiagnostic(
                                    frameIndex, currentKeypoints.size, matches.size,
                                    registration.inlierCount, registration.reason
                                )
                            }
                        }

                        reverseDescriptors.release()
                        reverseDescriptors = descriptors.clone()
                        reverseKeypoints = currentKeypoints
                        markFrameProcessed(frameIndex)
                        onProgress(
                            CameraTrackingProgress(
                                frameIndex + 1,
                                frameCount,
                                "Registered earlier frames",
                                overallFraction = 0.68f + 0.22f *
                                    (selected.secondFrame - frameIndex).toFloat() /
                                    selected.secondFrame.coerceAtLeast(1),
                                completedFrames = processedFrameCount,
                                totalFrames = frameCount
                            )
                        )
                    } finally {
                        source.release()
                        gray.release()
                        keypointsMat.release()
                        descriptors.release()
                    }
                }
            } finally {
                reverseDescriptors.release()
            }

            retryFailedFramesAfterMapUpdate(firstFrame.width, firstFrame.height)
            // Refresh the complete immutable snapshot once for final triangulation and refinement.
            tracks = currentFeatureTracks()
            // Flush newly observed tracks once more after the final registered pose.
            SfMTriangulator.triangulate(
                tracks, poseByFrame.filterNotNull(), cameraMatrix
            ).landmarks.forEach { landmark ->
                val old = landmarksByTrack[landmark.trackId]
                if (old == null || landmark.meanReprojectionErrorPx < old.meanReprojectionErrorPx) {
                    landmarksByTrack[landmark.trackId] = landmark
                }
            }

            // Two bounded block-coordinate BA rounds alternate robust 3D
            // landmark optimization with registered-camera PnP refinement.
            // This reduces map/pose inconsistency without a large sparse solver.
            val refinementFrames = poseByFrame.indices.filter { index ->
                index != selected.firstFrame && poseByFrame[index] != null
            }
            repeat(2) { refinementRound ->
                currentCoroutineContext().ensureActive()
                val adjustedLandmarks = MobileBundleAdjuster.optimizeLandmarks(
                    featureTracks = currentMapTracks(),
                    landmarks = landmarksByTrack.values.toList(),
                    poses = poseByFrame.filterNotNull(),
                    cameraMatrix = cameraMatrix
                )
                adjustedLandmarks.forEach { landmark ->
                    val old = landmarksByTrack[landmark.trackId]
                    if (old == null || landmark.meanReprojectionErrorPx <= old.meanReprojectionErrorPx) {
                        landmarksByTrack[landmark.trackId] = landmark
                    }
                }
                for ((position, frameIndex) in refinementFrames.withIndex()) {
                    currentCoroutineContext().ensureActive()
                    val registration = registerMapFrame(
                        frameIndex = frameIndex,
                        frameWidth = firstFrame.width,
                        frameHeight = firstFrame.height,
                        featureTracks = currentMapTracks(),
                        landmarks = landmarksByTrack.values.toList(),
                        cameraMatrix = cameraMatrix
                    )
                    if (registration.accepted && registration.pose != null) {
                        poseByFrame[frameIndex] = registration.pose
                        diagnostics[frameIndex] = FrameTrackingDiagnostic(
                            frameIndex,
                            diagnostics[frameIndex].detectedFeatures,
                            diagnostics[frameIndex].matchedFeatures,
                            registration.inlierCount,
                            "Refined PnP (${registration.inlierCount} inliers, " +
                                "median ${"%.2f".format(registration.medianReprojectionErrorPx)} px)"
                        )
                    }
                    if (position % 8 == 0 || position == refinementFrames.lastIndex) {
                        onProgress(
                            CameraTrackingProgress(
                                frameIndex + 1,
                                frameCount,
                                "Bundle adjustment · refining cameras",
                                overallFraction = 0.90f + 0.09f *
                                    (refinementRound +
                                        (position + 1).toFloat() /
                                            refinementFrames.size.coerceAtLeast(1)) / 2f,
                                completedFrames = frameCount,
                                totalFrames = frameCount
                            )
                        )
                    }
                }

                val postPoseLandmarks = MobileBundleAdjuster.optimizeLandmarks(
                    featureTracks = currentMapTracks(),
                    landmarks = landmarksByTrack.values.toList(),
                    poses = poseByFrame.filterNotNull(),
                    cameraMatrix = cameraMatrix
                )
                postPoseLandmarks.forEach { landmark ->
                    val old = landmarksByTrack[landmark.trackId]
                    if (old == null || landmark.meanReprojectionErrorPx <= old.meanReprojectionErrorPx) {
                        landmarksByTrack[landmark.trackId] = landmark
                    }
                }
                val refinedMap = SfMTriangulator.triangulate(
                    tracks, poseByFrame.filterNotNull(), cameraMatrix
                )
                refinedMap.landmarks.forEach { landmark ->
                    val old = landmarksByTrack[landmark.trackId]
                    if (old == null ||
                        landmark.meanReprojectionErrorPx < old.meanReprojectionErrorPx
                    ) {
                        landmarksByTrack[landmark.trackId] = landmark
                    }
                }
            }

            // Refine camera centers and sparse points with the pinned GLOMAP
            // ONLY_POINTS formulation. Use globally averaged rotations wherever
            // the bounded view graph covers a registered frame; retain the PnP
            // map for uncovered frames and as the rollback state.
            var globalPositioningApplied = false
            var globalPositioningInitialCost = Double.NaN
            var globalPositioningFinalCost = Double.NaN
            var globalPositioningIterations = 0
            var globalPositioningStatus = "Skipped: insufficient registered tracks"

            val fx = cameraMatrix.get(0, 0)[0]
            val skew = cameraMatrix.get(0, 1)[0]
            val cx = cameraMatrix.get(0, 2)[0]
            val fy = cameraMatrix.get(1, 1)[0]
            val cy = cameraMatrix.get(1, 2)[0]
            val validPoseByFrame = poseByFrame.filterNotNull().filter { pose ->
                pose.status != CameraPoseStatus.HELD_LAST_VALID &&
                    pose.status != CameraPoseStatus.UNREGISTERED
            }.associateBy { it.frameIndex }

            // Rotation averaging has an arbitrary world gauge. Align it to one
            // camera from the selected PnP seed pair so its rotations share the
            // existing centers/points coordinate system. Only graph-covered,
            // registered frames receive an override.
            val alignedGlobalRotationsByFrame: Map<Int, DoubleArray> =
                viewGraphRotation?.let { result ->
                    val anchorFrameId = listOf(selected.firstFrame, selected.secondFrame)
                        .firstOrNull { it in result.rotationsByFrame && it in validPoseByFrame }
                    val anchorPose = anchorFrameId?.let { validPoseByFrame[it] }
                    if (anchorFrameId == null || anchorPose == null) {
                        emptyMap()
                    } else {
                        GlobalViewGraphSolver.alignRotationsToPoseGauge(
                            rotationsByFrame = result.rotationsByFrame,
                            anchorFrameId = anchorFrameId,
                            targetAnchorRotation = anchorPose.rotation
                        ).orEmpty()
                    }
                }.orEmpty()
            val positioningRotationsByFrame = validPoseByFrame.mapValues { (frameId, pose) ->
                alignedGlobalRotationsByFrame[frameId] ?: pose.rotation
            }

            val initialCameraCenters = validPoseByFrame.mapValues { (_, pose) ->
                cameraCenter(pose)
            }
            val initialPointsByTrack = landmarksByTrack.mapValues { (_, landmark) ->
                doubleArrayOf(landmark.position.x, landmark.position.y, landmark.position.z)
            }

            val globalTracks = tracks.mapNotNull { track ->
                val point = initialPointsByTrack[track.id] ?: return@mapNotNull null
                val observations = track.observations.mapNotNull { observation ->
                    val pose = validPoseByFrame[observation.frameIndex] ?: return@mapNotNull null
                    if (!fx.isFinite() || !fy.isFinite() ||
                        kotlin.math.abs(fx) < 1e-9 || kotlin.math.abs(fy) < 1e-9 ||
                        !observation.x.isFinite() || !observation.y.isFinite()
                    ) return@mapNotNull null

                    val normalizedY = (observation.y - cy) / fy
                    val normalizedX = (observation.x - cx - skew * normalizedY) / fx
                    GlomapGlobalPositioningProblemBuilder.CameraObservation(
                        frameIndex = observation.frameIndex,
                        rotation = positioningRotationsByFrame.getValue(observation.frameIndex),
                        cameraCenter = initialCameraCenters.getValue(observation.frameIndex),
                        undistortedFeatureDirection = doubleArrayOf(normalizedX, normalizedY, 1.0),
                        isCalibrated = cameraFocalLengthHasPrior
                    )
                }
                if (observations.size < GlomapGlobalPositioningProblemBuilder.DEFAULT_MINIMUM_TRACK_VIEWS) {
                    null
                } else {
                    GlomapGlobalPositioningProblemBuilder.Track(
                        trackId = track.id,
                        observations = observations,
                        initializedPoint = point
                    )
                }
            }

            if (globalTracks.size >= 10) {
                val globalProblem = GlomapGlobalPositioningProblemBuilder.build(globalTracks)
                if (globalProblem.constraints.size >= 30) {
                    onProgress(
                        CameraTrackingProgress(
                            frameCount, frameCount, "Global positioning · Ceres",
                            overallFraction = 0.985f,
                            completedFrames = frameCount,
                            totalFrames = frameCount
                        )
                    )
                    try {
                        val solution = GlomapNativeSolver().solve(
                            problem = globalProblem,
                            frameCentersById = initialCameraCenters,
                            pointsByTrackId = initialPointsByTrack,
                            maxIterations = 100,
                            numThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4)
                        )
                        if (solution == null) {
                            globalPositioningStatus = "Ceres did not return a usable solution"
                        } else {
                            globalPositioningInitialCost = solution.initialCost
                            globalPositioningFinalCost = solution.finalCost
                            globalPositioningIterations = solution.iterations
                            if (solution.initialCost.isFinite() && solution.finalCost.isFinite() &&
                                solution.finalCost < solution.initialCost
                            ) {
                                for ((frameId, center) in solution.frameCentersById) {
                                    val oldPose = poseByFrame.getOrNull(frameId) ?: continue
                                    val rotation = positioningRotationsByFrame[frameId]
                                        ?: oldPose.rotation
                                    val translation = doubleArrayOf(
                                        -(rotation[0] * center[0] + rotation[1] * center[1] + rotation[2] * center[2]),
                                        -(rotation[3] * center[0] + rotation[4] * center[1] + rotation[5] * center[2]),
                                        -(rotation[6] * center[0] + rotation[7] * center[1] + rotation[8] * center[2])
                                    )
                                    poseByFrame[frameId] = oldPose.copy(translation = translation)
                                }
                                for ((trackId, point) in solution.pointsByTrackId) {
                                    val old = landmarksByTrack[trackId] ?: continue
                                    landmarksByTrack[trackId] = old.copy(
                                        position = Point3(point[0], point[1], point[2])
                                    )
                                }
                                globalPositioningApplied = true
                                globalPositioningStatus = "Applied"
                            } else {
                                globalPositioningStatus = "No cost improvement; kept incremental solution"
                            }
                        }
                    } catch (failure: LinkageError) {
                        globalPositioningStatus = "Native solver unavailable: ${failure.message ?: "library load failed"}"
                    } catch (failure: RuntimeException) {
                        globalPositioningStatus = "Ceres solve skipped: ${failure.message ?: "runtime error"}"
                    }
                } else {
                    globalPositioningStatus = "Skipped: fewer than 30 point-to-camera constraints"
                }
            } else {
                globalPositioningStatus = "Skipped: fewer than 10 tracks with 3 registered views"
            }

            var jointBundleAdjustmentApplied = false
            var jointBundleAdjustmentInitialCost = Double.NaN
            var jointBundleAdjustmentFinalCost = Double.NaN
            var jointBundleAdjustmentIterations = 0
            var jointBundleAdjustmentStatus = "Skipped: insufficient registered observations"
            var jointBundleAdjustmentPositiveDepthRatio = Double.NaN
            var jointBundleAdjustmentPrincipalPointRefinementEnabled = false
            var jointRefinedTrackIds: Set<Int> = emptySet()
            var refinedCameraIntrinsics: CameraIntrinsics? = null

            val jointTracks = currentMapTracks()
            val jointPoseByFrame = poseByFrame.filterNotNull().filter { pose ->
                pose.status != CameraPoseStatus.HELD_LAST_VALID &&
                    pose.status != CameraPoseStatus.UNREGISTERED
            }.associateBy { it.frameIndex }
            val jointCenters = jointPoseByFrame.mapValues { (_, pose) -> cameraCenter(pose) }
            val jointRotations = jointPoseByFrame.mapValues { (_, pose) -> pose.rotation }
            val jointPoints = landmarksByTrack.mapValues { (_, landmark) ->
                doubleArrayOf(landmark.position.x, landmark.position.y, landmark.position.z)
            }

            // The Android path runs a bounded local window after each two accepted cameras,
            // then keeps this closing pass after the map-wide positioning/refinement stages.
            val localBundleSelection = if (
                jointPoseByFrame.size > ColmapLocalBundleSelector.DEFAULT_MAX_NUM_IMAGES
            ) {
                // The highest registered frame is the stable temporal anchor here.
                jointPoseByFrame.keys.maxOrNull()?.let { referenceFrameId ->
                    ColmapLocalBundleSelector.select(
                        referenceFrameId = referenceFrameId,
                        cameraCentersByFrame = jointCenters,
                        tracks = jointTracks,
                        pointsByTrackId = jointPoints
                    )
                }
            } else {
                null
            }
            val localFrameIds = localBundleSelection?.frameIds?.toSet().orEmpty()
            val localBundleTracks = if (localBundleSelection != null) {
                jointTracks.filter { track ->
                    track.observations.count { it.frameIndex in localFrameIds } >= 3
                }
            } else {
                emptyList()
            }
            val localTrackIds = localBundleTracks.map { it.id }.toSet()
            val localVariablePointTrackIds = localBundleTracks.asSequence()
                .filter { it.length <= 15 }
                .map { it.id }
                .toSet()
            val localObservationCount = localBundleTracks.sumOf { track ->
                track.observations.count { it.frameIndex in localFrameIds }
            }
            val useLocalBundle = localBundleSelection != null &&
                localBundleSelection.frameIds.size >= 3 &&
                localBundleTracks.size >= 20 &&
                localObservationCount >= 60
            val bundleFrameCenters = if (useLocalBundle) {
                jointCenters.filterKeys { it in localFrameIds }
            } else {
                jointCenters
            }
            val bundleRotations = if (useLocalBundle) {
                jointRotations.filterKeys { it in localFrameIds }
            } else {
                jointRotations
            }
            val bundlePoints = if (useLocalBundle) {
                jointPoints.filterKeys { it in localTrackIds }
            } else {
                jointPoints
            }
            val bundleTracks = if (useLocalBundle) localBundleTracks else jointTracks

            when {
                jointPoseByFrame.size < 3 || jointTracks.count { track ->
                    track.observations.count { it.frameIndex in jointPoseByFrame } >= 3
                } < 20 -> {
                    jointBundleAdjustmentStatus = "Skipped: fewer than 20 tracks with three registered views"
                }
                else -> {
                    onProgress(
                        CameraTrackingProgress(
                            frameCount, frameCount, "Joint camera/point refinement · Ceres",
                            overallFraction = 0.99f,
                            completedFrames = frameCount,
                            totalFrames = frameCount
                        )
                    )
                    try {
                        val adjuster = JointCameraBundleAdjuster()
                        fun solveJointBundle(
                            centers: Map<Int, DoubleArray>,
                            rotations: Map<Int, DoubleArray>,
                            points: Map<Int, DoubleArray>,
                            tracks: List<CameraFeatureTrack>,
                            optimizeFocalLength: Boolean,
                            optimizePointTrackIds: Set<Int>? = null
                        ) = adjuster.solve(
                            frameCentersById = centers,
                            rotationsByFrame = rotations,
                            pointsByTrackId = points,
                            tracks = tracks,
                            cameraMatrix = cameraMatrix,
                            imageWidth = firstFrame.width,
                            imageHeight = firstFrame.height,
                            maxIterations = 45,
                            numThreads = Runtime.getRuntime().availableProcessors().coerceIn(1, 4),
                            optimizeFocalLength = optimizeFocalLength,
                            optimizePointTrackIds = optimizePointTrackIds
                        )

                        val localCandidate = if (useLocalBundle) {
                            try {
                                solveJointBundle(
                                    bundleFrameCenters,
                                    bundleRotations,
                                    bundlePoints,
                                    bundleTracks,
                                    optimizeFocalLength = false,
                                    optimizePointTrackIds = localVariablePointTrackIds
                                )
                            } catch (_: RuntimeException) {
                                null
                            }
                        } else {
                            null
                        }
                        val candidate = localCandidate ?: solveJointBundle(
                            jointCenters,
                            jointRotations,
                            jointPoints,
                            jointTracks,
                            optimizeFocalLength = true
                        )
                        if (candidate == null) {
                            jointBundleAdjustmentStatus = "Skipped: insufficient connected observations"
                        } else {
                            jointBundleAdjustmentInitialCost = candidate.initialCost
                            jointBundleAdjustmentFinalCost = candidate.finalCost
                            jointBundleAdjustmentIterations = candidate.iterations
                            jointBundleAdjustmentPositiveDepthRatio = candidate.positiveDepthRatio
                            jointBundleAdjustmentPrincipalPointRefinementEnabled =
                                candidate.principalPointRefinementEnabled

                            if (SolverObjective.didNotIncrease(
                                    candidate.initialCost,
                                    candidate.finalCost
                                )
                            ) {
                                for ((frameId, center) in candidate.frameCentersById) {
                                    val oldPose = poseByFrame.getOrNull(frameId) ?: continue
                                    val rotation = candidate.rotationsByFrame[frameId] ?: continue
                                    val translation = doubleArrayOf(
                                        -(rotation[0] * center[0] + rotation[1] * center[1] + rotation[2] * center[2]),
                                        -(rotation[3] * center[0] + rotation[4] * center[1] + rotation[5] * center[2]),
                                        -(rotation[6] * center[0] + rotation[7] * center[1] + rotation[8] * center[2])
                                    )
                                    poseByFrame[frameId] = oldPose.copy(
                                        rotation = rotation.copyOf(),
                                        translation = translation
                                    )
                                }
                                for ((trackId, point) in candidate.pointsByTrackId) {
                                    val old = landmarksByTrack[trackId] ?: continue
                                    landmarksByTrack[trackId] = old.copy(
                                        position = Point3(point[0], point[1], point[2])
                                    )
                                }

                                val intrinsics = PinholeIntrinsics(
                                    fx = candidate.intrinsics.fx,
                                    fy = candidate.intrinsics.fy,
                                    skew = candidate.intrinsics.skew,
                                    cx = candidate.intrinsics.cx,
                                    cy = candidate.intrinsics.cy
                                )
                                applyObservationFilter(
                                    affectedFrameIds = candidate.frameCentersById.keys,
                                    changedTrackIds = candidate.pointsByTrackId.keys,
                                    intrinsics = intrinsics
                                )
                                for (track in currentMapTracks()) {
                                    val landmark = landmarksByTrack[track.id] ?: continue
                                    val point = doubleArrayOf(
                                        landmark.position.x,
                                        landmark.position.y,
                                        landmark.position.z
                                    )
                                    val errors = track.observations.mapNotNull { observation ->
                                        val pose = poseByFrame.getOrNull(observation.frameIndex)
                                            ?: return@mapNotNull null
                                        val r = pose.rotation
                                        val t = pose.translation
                                        val cameraPoint = doubleArrayOf(
                                            r[0] * point[0] + r[1] * point[1] + r[2] * point[2] + t[0],
                                            r[3] * point[0] + r[4] * point[1] + r[5] * point[2] + t[1],
                                            r[6] * point[0] + r[7] * point[1] + r[8] * point[2] + t[2]
                                        )
                                        if (cameraPoint.any { !it.isFinite() } || cameraPoint[2] <= 1e-5) {
                                            return@mapNotNull null
                                        }
                                        val pixel = CameraProjectionMath.project(cameraPoint, intrinsics)
                                        hypot(pixel[0] - observation.x, pixel[1] - observation.y)
                                            .takeIf {
                                                it.isFinite() &&
                                                    it <= MobileBundleAdjuster.MAX_OBSERVATION_ERROR_PX
                                            }
                                    }
                                    if (errors.size >= 2) {
                                        landmarksByTrack[track.id] = landmark.copy(
                                            observationCount = errors.size,
                                            meanReprojectionErrorPx = errors.average()
                                        )
                                    }
                                }
                                refinedCameraIntrinsics = candidate.intrinsics
                                jointBundleAdjustmentApplied = true
                                jointBundleAdjustmentStatus = "Applied"
                                jointRefinedTrackIds = candidate.pointsByTrackId.keys
                                    .filterTo(linkedSetOf()) { it in landmarksByTrack }
                            } else {
                                jointBundleAdjustmentStatus =
                                    "No cost improvement; kept incremental solution"
                            }
                        }
                    } catch (failure: LinkageError) {
                        jointBundleAdjustmentStatus =
                            "Native solver unavailable: ${failure.message ?: "library load failed"}"
                    } catch (failure: RuntimeException) {
                        jointBundleAdjustmentStatus =
                            "Ceres solve skipped: ${failure.message ?: "runtime error"}"
                    }
                }
            }

            onProgress(
                CameraTrackingProgress(
                    frameCount, frameCount, "Joint refinement complete",
                    overallFraction = 0.99f,
                    completedFrames = frameCount,
                    totalFrames = frameCount
                )
            )

            val denseTimeline = ArrayList<CameraPose>(frameCount)
            var lastValid: CameraPose? = null
            for (index in 0 until frameCount) {
                val actual = poseByFrame[index]
                if (actual != null) {
                    denseTimeline += actual
                    lastValid = actual
                } else {
                    val beforeInitialization = index < selected.firstFrame
                    val held = lastValid ?: referencePose
                    denseTimeline += CameraPose(
                        index,
                        if (beforeInitialization) identity3() else held.rotation.copyOf(),
                        if (beforeInitialization) doubleArrayOf(0.0, 0.0, 0.0)
                        else held.translation.copyOf(),
                        if (beforeInitialization) CameraPoseStatus.UNREGISTERED
                        else CameraPoseStatus.HELD_LAST_VALID
                    )
                }
            }

            val allCorrespondences = max(1, totalCorrespondences)
            val finalCameraMatrix = refinedCameraIntrinsics?.let { intrinsics ->
                Mat(3, 3, CvType.CV_64F).also { matrix ->
                    matrix.put(0, 0, *intrinsics.toRowMajorMatrix())
                }
            } ?: cameraMatrix
            val finalTriangulation = try {
                SfMTriangulator.triangulate(
                    currentFeatureTracks(), poseByFrame.filterNotNull(), finalCameraMatrix
                )
            } finally {
                if (finalCameraMatrix !== cameraMatrix) finalCameraMatrix.release()
            }
            val finalLandmarks = finalTriangulation.landmarks.map { triangulated ->
                val refined = if (jointBundleAdjustmentApplied && triangulated.trackId in jointRefinedTrackIds) {
                    landmarksByTrack[triangulated.trackId]
                } else null
                refined?.copy(
                    triangulationAngleDegrees = triangulated.triangulationAngleDegrees
                ) ?: triangulated
            }
            val finalStats = finalTriangulation.stats.withFinalLandmarks(finalLandmarks)
            val finalLandmarkTrackIds = finalLandmarks.mapTo(linkedSetOf<Int>()) { it.trackId }
            onProgress(CameraTrackingProgress(
                frameCount, frameCount, "Camera solve complete",
                overallFraction = 1f,
                completedFrames = frameCount,
                totalFrames = frameCount
            ))
            CameraTrackingResult(
                poses = denseTimeline,
                diagnostics = diagnostics,
                processedFrames = frameCount,
                solvedFramePairs = registeredFrames - 1,
                totalMatches = totalMatches,
                totalInliers = totalInliers,
                inlierRatio = (totalInliers.toDouble() / allCorrespondences).coerceIn(0.0, 1.0),
                isCameraMotionEstimated = registeredFrames >= 2,
                featureTracks = finalTriangulation.inlierTracks.filter {
                    it.id in finalLandmarkTrackIds
                },
                landmarks = finalLandmarks,
                triangulationStats = finalStats,
                initialPairFrames = selected.firstFrame to selected.secondFrame,
                registeredCameraCount = registeredFrames,
                globalRotationsByFrame = viewGraphRotation?.rotationsByFrame.orEmpty(),
                viewGraphEdgeCount = viewGraphRotation?.inputEdgeCount ?: 0,
                viewGraphAcceptedEdgeCount = viewGraphRotation?.acceptedEdgeCount ?: 0,
                globalRotationIterations = viewGraphRotation?.iterations ?: 0,
                globalPositioningApplied = globalPositioningApplied,
                globalPositioningInitialCost = globalPositioningInitialCost,
                globalPositioningFinalCost = globalPositioningFinalCost,
                globalPositioningIterations = globalPositioningIterations,
                globalPositioningStatus = globalPositioningStatus,
                jointBundleAdjustmentApplied = jointBundleAdjustmentApplied,
                jointBundleAdjustmentInitialCost = jointBundleAdjustmentInitialCost,
                jointBundleAdjustmentFinalCost = jointBundleAdjustmentFinalCost,
                jointBundleAdjustmentIterations = jointBundleAdjustmentIterations,
                jointBundleAdjustmentStatus = jointBundleAdjustmentStatus,
                jointBundleAdjustmentPositiveDepthRatio = jointBundleAdjustmentPositiveDepthRatio,
                jointBundleAdjustmentPrincipalPointRefinementEnabled =
                    jointBundleAdjustmentPrincipalPointRefinementEnabled,
                refinedCameraIntrinsics = refinedCameraIntrinsics ?: estimatedCameraIntrinsics
            )
        } finally {
            coarseDescriptorCache.values.forEach { it.release() }
            cache.values.forEach { it.descriptors.release() }
            matcher.clear()
            featureExtractor.clear()
            try {
                originalCameraMatrix.copyTo(cameraMatrix)
            } finally {
                originalCameraMatrix.release()
            }
        }
    }

    private fun coarseDescriptorSubset(frame: CachedFrame, maximumDescriptors: Int): Mat {
        require(maximumDescriptors > 0)
        val descriptors = frame.descriptors
        if (descriptors.empty() || frame.keypoints.isEmpty()) return Mat()
        require(descriptors.type() == CvType.CV_8U) {
            "Coarse SIFT matching requires quantized single-channel descriptors."
        }
        val availableRows = minOf(descriptors.rows(), frame.keypoints.size)
        if (availableRows <= 0) return Mat()
        val selectedRows = (0 until availableRows)
            .sortedWith(
                compareByDescending<Int> { row ->
                    frame.keypoints[row].response.takeIf { it.isFinite() } ?: Float.NEGATIVE_INFINITY
                }.thenBy { it }
            )
            .take(maximumDescriptors)
        val subset = Mat(selectedRows.size, descriptors.cols(), descriptors.type())
        val rowBuffer = ByteArray(descriptors.cols() * descriptors.channels())
        try {
            selectedRows.forEachIndexed { subsetRow, sourceRow ->
                val copied = descriptors.get(sourceRow, 0, rowBuffer)
                check(copied == rowBuffer.size) { "Unable to read a coarse SIFT descriptor row." }
                subset.put(subsetRow, 0, rowBuffer)
            }
            return subset
        } catch (failure: Throwable) {
            subset.release()
            throw failure
        }
    }

    private fun estimatePairGeometry(
        first: CachedFrame,
        second: CachedFrame,
        cameraMatrix: Mat,
        matcher: SiftDescriptorMatcher
    ): PairGeometry? {
        val raw = ratioMatches(
            first.descriptors, second.descriptors,
            first.keypoints, second.keypoints, matcher
        )
        if (raw.size < 8) return null
        return verifyGeometry(raw, first.keypoints, second.keypoints, cameraMatrix, true)
    }

    private fun verifiedMatches(
        firstDescriptors: Mat,
        firstKeypoints: Array<KeyPoint>,
        secondDescriptors: Mat,
        secondKeypoints: Array<KeyPoint>,
        cameraMatrix: Mat,
        matcher: SiftDescriptorMatcher
    ): List<Pair<Int, Int>> {
        val raw = ratioMatches(
            firstDescriptors, secondDescriptors, firstKeypoints, secondKeypoints, matcher
        )
        if (raw.size < 8) return emptyList()
        return verifyTrackGeometry(raw, firstKeypoints, secondKeypoints)
    }

    /**
     * Verifies 2D feature continuity without requiring recoverPose/cheirality.
     *
     * recoverPose is appropriate for initializing a 3D camera pair, but it is
     * too restrictive for ordinary frame-to-frame tracking: small-baseline,
     * near-rotation and mostly-planar motion can still have valid 2D tracks.
     * Use robust epipolar and planar models here; PnP remains the gate for
     * accepting an actual 3D camera pose.
     */
    private fun verifyTrackGeometry(
        matches: List<Pair<Int, Int>>,
        firstKeypoints: Array<KeyPoint>,
        secondKeypoints: Array<KeyPoint>
    ): List<Pair<Int, Int>> {
        if (matches.size < 8) return emptyList()
        val points1 = MatOfPoint2f()
        val points2 = MatOfPoint2f()
        val fundamentalMask = Mat()
        val homographyMask = Mat()
        var fundamental: Mat? = null
        var homography: Mat? = null
        try {
            points1.fromArray(*matches.map { firstKeypoints[it.first].pt }.toTypedArray())
            points2.fromArray(*matches.map { secondKeypoints[it.second].pt }.toTypedArray())

            fundamental = Calib3d.findFundamentalMat(
                points1, points2, Calib3d.FM_RANSAC, 1.5, 0.995, fundamentalMask
            )
            fun maskIsInlier(mask: Mat, index: Int): Boolean {
                if (mask.empty() || mask.channels() != 1) return false
                val coordinate = OpenCvMaskLayout.coordinates(
                    mask.rows(), mask.cols(), index, matches.size
                ) ?: return false
                return mask.get(coordinate.first, coordinate.second)
                    ?.firstOrNull()?.let { it != 0.0 } == true
            }
            fun maskInlierCount(mask: Mat): Int {
                return (0 until matches.size).count { index ->
                    maskIsInlier(mask, index)
                }
            }

            val fundamentalInliers = maskInlierCount(fundamentalMask)
            // Fundamental RANSAC is the normal fast path. Only pay for a second
            // homography RANSAC when epipolar verification is weak; this retains
            // planar/near-planar recovery without doubling the cost on most frames.
            val strongFundamentalModel = fundamentalInliers >=
                max(16, (matches.size * 0.45f).toInt())
            if (!strongFundamentalModel) {
                homography = Calib3d.findHomography(
                    points1, points2, Calib3d.RANSAC, 3.0, homographyMask
                )
            }
            val homographyInliers = maskInlierCount(homographyMask)
            val selectedMask = if (homographyInliers > fundamentalInliers) {
                homographyMask
            } else {
                fundamentalMask
            }
            if (max(fundamentalInliers, homographyInliers) < 8) return emptyList()

            return matches.filterIndexed { index, _ -> maskIsInlier(selectedMask, index) }
        } finally {
            points1.release()
            points2.release()
            fundamentalMask.release()
            homographyMask.release()
            fundamental?.release()
            homography?.release()
        }
    }

    private fun ratioMatches(
        firstDescriptors: Mat,
        secondDescriptors: Mat,
        firstKeypoints: Array<KeyPoint>,
        secondKeypoints: Array<KeyPoint>,
        matcher: SiftDescriptorMatcher
    ): List<Pair<Int, Int>> {
        if (firstDescriptors.empty() || secondDescriptors.empty() ||
            firstDescriptors.rows() == 0 || secondDescriptors.rows() == 0
        ) return emptyList()
        return matcher.match(firstDescriptors, secondDescriptors).filter { (query, train) ->
            query in firstKeypoints.indices && train in secondKeypoints.indices
        }
    }

    private fun verifyGeometry(
        matches: List<Pair<Int, Int>>,
        firstKeypoints: Array<KeyPoint>,
        secondKeypoints: Array<KeyPoint>,
        cameraMatrix: Mat,
        includePose: Boolean
    ): PairGeometry? {
        val points1 = MatOfPoint2f()
        val points2 = MatOfPoint2f()
        val mask = Mat()
        val rotation = Mat()
        val translation = Mat()
        try {
            val matchedPoints1 = matches.map { firstKeypoints[it.first].pt }.toTypedArray()
            val matchedPoints2 = matches.map { secondKeypoints[it.second].pt }.toTypedArray()
            points1.fromArray(*matchedPoints1)
            points2.fromArray(*matchedPoints2)
            val essential = Calib3d.findEssentialMat(
                points1, points2, cameraMatrix, Calib3d.RANSAC,
                0.999, 1.25, 1200, mask
            )
            try {
                if (essential.empty() || essential.cols() != 3 ||
                    essential.rows() < 3 || essential.rows() % 3 != 0
                ) return null

                var bestCount = 0
                var bestMask: Mat? = null
                var bestR: Mat? = null
                var bestT: Mat? = null
                var bestEssential: Mat? = null
                try {
                    for (index in 0 until essential.rows() / 3) {
                        val candidate = essential.rowRange(index * 3, index * 3 + 3).clone()
                        val candidateMask = mask.clone()
                        val r = Mat()
                        val t = Mat()
                        try {
                            val count = OpenCvTwoViewPoseRecovery.recover(
                                candidate, points1, points2, cameraMatrix, r, t, candidateMask
                            )
                            if (count > bestCount) {
                                bestCount = count
                                bestMask?.release()
                                bestR?.release()
                                bestT?.release()
                                bestMask = candidateMask.clone()
                                bestR = r.clone()
                                bestT = t.clone()
                                bestEssential?.release()
                                bestEssential = candidate.clone()
                            }
                        } finally {
                            candidate.release()
                            candidateMask.release()
                            r.release()
                            t.release()
                        }
                    }

                    val selectedEssential = bestEssential
                    if (selectedEssential != null) {
                        val initialEssential = DoubleArray(9)
                        if (selectedEssential.get(0, 0, initialEssential) ==
                            initialEssential.size * java.lang.Double.BYTES
                        ) {
                            val polished = OpenCvTwoViewLocalOptimizer.refine(
                                matchedPoints1,
                                matchedPoints2,
                                cameraMatrix,
                                initialEssential,
                                bestMask?.let {
                                    OpenCvTwoViewLocalOptimizer.inliersFromMask(it, matches.size)
                                }.orEmpty(),
                                1.25,
                                0.999,
                                1200
                            )
                            if (polished != null && polished.inlierIndices.size >= bestCount) {
                                val polishedMask = OpenCvTwoViewLocalOptimizer.maskFromIndices(
                                    polished.inlierIndices, matches.size
                                )
                                val polishedRotation = Mat(3, 3, CvType.CV_64F)
                                val polishedTranslation = Mat(3, 1, CvType.CV_64F)
                                var retained = false
                                try {
                                    if (polishedRotation.put(0, 0, *polished.rotation) == polished.rotation.size &&
                                        polishedTranslation.put(0, 0, *polished.translation) == polished.translation.size
                                    ) {
                                        bestMask?.release()
                                        bestR?.release()
                                        bestT?.release()
                                        bestMask = polishedMask
                                        bestR = polishedRotation
                                        bestT = polishedTranslation
                                        bestCount = polished.inlierIndices.size
                                        retained = true
                                    }
                                } finally {
                                    if (!retained) {
                                        polishedMask.release()
                                        polishedRotation.release()
                                        polishedTranslation.release()
                                    }
                                }
                            }
                        }
                    }

                    val chosenMask = bestMask ?: return null
                    val inlierPairs = matches.filterIndexed { index, _ ->
                        if (chosenMask.empty() || chosenMask.channels() != 1) return@filterIndexed false
                        val coordinate = OpenCvMaskLayout.coordinates(
                            chosenMask.rows(), chosenMask.cols(), index, matches.size
                        ) ?: return@filterIndexed false
                        chosenMask.get(coordinate.first, coordinate.second)
                            ?.firstOrNull()?.let { it != 0.0 } == true
                    }
                    if (inlierPairs.size != bestCount || inlierPairs.size < 8) return null
                    if (!includePose) {
                        return PairGeometry(inlierPairs, identity3(), doubleArrayOf(0.0, 0.0, 0.0), bestCount)
                    }
                    val chosenR = bestR ?: return null
                    val chosenT = bestT ?: return null
                    val rValues = OpenCvMatrixReadback.readCv64F(chosenR, 3, 3) ?: return null
                    val tValues = OpenCvMatrixReadback.readCv64F(chosenT, 3, 1) ?: return null
                    return PairGeometry(inlierPairs, rValues, tValues, bestCount)
                } finally {
                    bestMask?.release()
                    bestR?.release()
                    bestT?.release()
                    bestEssential?.release()
                }
            } finally {
                essential.release()
            }
        } finally {
            points1.release()
            points2.release()
            mask.release()
            rotation.release()
            translation.release()
        }
    }

    internal fun toGray(source: Mat): Mat {
        val gray = Mat()
        when (source.channels()) {
            1 -> source.copyTo(gray)
            3 -> Imgproc.cvtColor(source, gray, Imgproc.COLOR_BGR2GRAY)
            4 -> Imgproc.cvtColor(source, gray, Imgproc.COLOR_RGBA2GRAY)
            else -> {
                gray.release()
                throw IllegalArgumentException("Unsupported image channel count: ${source.channels()}")
            }
        }
        return gray
    }

    private fun identity3() = doubleArrayOf(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
}
