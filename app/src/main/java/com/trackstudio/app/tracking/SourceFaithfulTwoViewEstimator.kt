package com.trackstudio.app.tracking

/** Bounded Android bridge for source-faithful initial-pair E/F/H estimation and pose recovery. */
internal object SourceFaithfulTwoViewEstimator {
    data class ModelEstimate(
        val success: Boolean,
        val modelRowMajor: DoubleArray?,
        val inlierMask: BooleanArray,
        val inlierCount: Int,
        val numTrials: Int,
        val residualSum: Double,
        val localOptimizationTrials: Int,
        val acceptedLocalRefits: Int,
        val bestModelWasLocal: Boolean
    )

    data class EpipolarEstimate(
        val essential: ModelEstimate,
        val fundamental: ModelEstimate
    )

    data class PoseEstimate(
        val rotationRowMajor: DoubleArray,
        val translation: DoubleArray,
        val selectedCandidate: Int,
        val cheiralityMask: BooleanArray,
        val inlierIndices: List<Int>
    )

    enum class Configuration { DEGENERATE, CALIBRATED, UNCALIBRATED, PLANAR_OR_PANORAMIC }
    enum class InlierModel { NONE, ESSENTIAL, FUNDAMENTAL, HOMOGRAPHY }
    enum class PoseSource { NONE, ESSENTIAL, FUNDAMENTAL, HOMOGRAPHY }

    data class IntegratedModelEstimate(
        val success: Boolean,
        val inlierCount: Int,
        val inlierMask: BooleanArray,
        val numTrials: Int,
        val localOptimizationTrials: Int,
        val acceptedLocalRefits: Int,
        val bestModelWasLocal: Boolean,
        val residualSum: Double
    )

    data class InitialPairEstimate(
        val configuration: Configuration,
        val selectedModel: InlierModel,
        val poseSource: PoseSource,
        val numInliers: Int,
        val essential: IntegratedModelEstimate,
        val fundamental: IntegratedModelEstimate,
        val homography: IntegratedModelEstimate,
        val selectedInlierMask: BooleanArray,
        val pose: PoseEstimate?,
        val medianTriangulationAngleRadians: Double,
        val passesInitialPairGates: Boolean
    )

    private val isAvailable: Boolean = try {
        System.loadLibrary("trackstudio_camera")
        true
    } catch (_: UnsatisfiedLinkError) {
        false
    }

    fun estimate(
        firstPixelsXY: DoubleArray,
        secondPixelsXY: DoubleArray,
        cameraMatrixRowMajor: DoubleArray,
        maxErrorPixels: Double
    ): EpipolarEstimate? {
        if (!isAvailable || firstPixelsXY.size != secondPixelsXY.size ||
            firstPixelsXY.size % 2 != 0 ||
            firstPixelsXY.size !in MINIMUM_MATCHES * 2..MAX_INPUT_MATCHES * 2 ||
            cameraMatrixRowMajor.size != 9 ||
            firstPixelsXY.any { !it.isFinite() } ||
            secondPixelsXY.any { !it.isFinite() } ||
            cameraMatrixRowMajor.any { !it.isFinite() } ||
            cameraMatrixRowMajor[0] <= 0.0 || cameraMatrixRowMajor[4] <= 0.0 ||
            !maxErrorPixels.isFinite() || maxErrorPixels <= 0.0
        ) return null

        val matchCount = firstPixelsXY.size / 2
        val packed = nativeEstimate(
            firstPixelsXY, secondPixelsXY, cameraMatrixRowMajor, maxErrorPixels
        ) ?: return null
        val blockSize = MODEL_HEADER_SIZE + matchCount
        if (packed.size != 2 * blockSize || packed[0] != RESULT_VERSION ||
            packed[blockSize] != RESULT_VERSION
        ) return null

        val essential = parseModel(packed, 0, matchCount, MINIMUM_ESSENTIAL_MATCHES)
            ?: return null
        val fundamental = parseModel(
            packed, blockSize, matchCount, MINIMUM_FUNDAMENTAL_MATCHES
        ) ?: return null
        return EpipolarEstimate(essential, fundamental)
    }

    fun recoverPose(
        essentialRowMajor: DoubleArray,
        firstPixelsXY: DoubleArray,
        secondPixelsXY: DoubleArray,
        cameraMatrixRowMajor: DoubleArray,
        epipolarInlierMask: BooleanArray
    ): PoseEstimate? {
        if (!isAvailable || essentialRowMajor.size != 9 ||
            firstPixelsXY.size != secondPixelsXY.size || firstPixelsXY.size % 2 != 0 ||
            firstPixelsXY.size !in MINIMUM_MATCHES * 2..MAX_INPUT_MATCHES * 2 ||
            epipolarInlierMask.size != firstPixelsXY.size / 2 ||
            epipolarInlierMask.count { it } < MINIMUM_ESSENTIAL_MATCHES ||
            cameraMatrixRowMajor.size != 9 ||
            essentialRowMajor.any { !it.isFinite() } ||
            firstPixelsXY.any { !it.isFinite() } || secondPixelsXY.any { !it.isFinite() } ||
            cameraMatrixRowMajor.any { !it.isFinite() } ||
            cameraMatrixRowMajor[0] <= 0.0 || cameraMatrixRowMajor[4] <= 0.0
        ) return null

        val packed = nativeRecoverPose(
            essentialRowMajor,
            firstPixelsXY,
            secondPixelsXY,
            cameraMatrixRowMajor,
            IntArray(epipolarInlierMask.size) { if (epipolarInlierMask[it]) 1 else 0 }
        ) ?: return null
        val matchCount = epipolarInlierMask.size
        if (packed.size != POSE_HEADER_SIZE + matchCount ||
            packed[0] != RESULT_VERSION || packed.take(POSE_HEADER_SIZE).any { !it.isFinite() }
        ) return null

        val cheiralCount = packed[1].toInt()
        val selectedCandidate = packed[2].toInt()
        if (cheiralCount <= 0 || cheiralCount > epipolarInlierMask.count { it } ||
            selectedCandidate !in 0..3
        ) return null
        val cheiralityMask = BooleanArray(matchCount) {
            packed[POSE_HEADER_SIZE + it] == 1.0
        }
        if ((0 until matchCount).any {
                packed[POSE_HEADER_SIZE + it] != 0.0 && packed[POSE_HEADER_SIZE + it] != 1.0
            } || cheiralityMask.count { it } != cheiralCount ||
            (0 until matchCount).any { cheiralityMask[it] && !epipolarInlierMask[it] }
        ) return null

        val rotation = packed.copyOfRange(3, 12)
        val translation = packed.copyOfRange(12, 15)
        if (rotation.any { !it.isFinite() } || translation.any { !it.isFinite() }) return null
        return PoseEstimate(
            rotationRowMajor = rotation,
            translation = translation,
            selectedCandidate = selectedCandidate,
            cheiralityMask = cheiralityMask,
            inlierIndices = (0 until matchCount).filter { cheiralityMask[it] }
        )
    }

    fun estimateInitialPair(
        firstPixelsXY: DoubleArray,
        secondPixelsXY: DoubleArray,
        cameraMatrixRowMajor: DoubleArray,
        bothCamerasHaveFocalPriors: Boolean,
        maxErrorPixels: Double,
        minimumInliers: Int = 15,
        minimumEssentialToFundamentalRatio: Double = 0.95,
        maximumHomographyRatio: Double = 0.8
    ): InitialPairEstimate? {
        if (!isAvailable || firstPixelsXY.size != secondPixelsXY.size ||
            firstPixelsXY.size % 2 != 0 ||
            firstPixelsXY.size !in MINIMUM_MATCHES * 2..MAX_INPUT_MATCHES * 2 ||
            cameraMatrixRowMajor.size != 9 || firstPixelsXY.any { !it.isFinite() } ||
            secondPixelsXY.any { !it.isFinite() } ||
            cameraMatrixRowMajor.any { !it.isFinite() } ||
            cameraMatrixRowMajor[0] <= 0.0 || cameraMatrixRowMajor[4] <= 0.0 ||
            !maxErrorPixels.isFinite() || maxErrorPixels <= 0.0 || minimumInliers <= 0 ||
            !minimumEssentialToFundamentalRatio.isFinite() ||
            minimumEssentialToFundamentalRatio !in 0.0..1.0 ||
            !maximumHomographyRatio.isFinite() || maximumHomographyRatio !in 0.0..1.0
        ) return null

        val matchCount = firstPixelsXY.size / 2
        val packed = nativeEstimateInitialPair(
            firstPixelsXY,
            secondPixelsXY,
            cameraMatrixRowMajor,
            bothCamerasHaveFocalPriors,
            maxErrorPixels,
            minimumInliers,
            minimumEssentialToFundamentalRatio,
            maximumHomographyRatio
        ) ?: return null
        if (packed.size != INITIAL_PAIR_HEADER_SIZE + 5 * matchCount ||
            packed[0] != RESULT_VERSION ||
            packed.sliceArray(0 until INITIAL_PAIR_HEADER_SIZE).any { !it.isFinite() }
        ) return null

        val configuration = Configuration.values().getOrNull(packed[1].toInt()) ?: return null
        val selectedModel = InlierModel.values().getOrNull(packed[2].toInt()) ?: return null
        val poseSource = PoseSource.values().getOrNull(packed[3].toInt()) ?: return null
        val numInliers = packed[4].toInt()
        if (numInliers !in 0..matchCount ||
            packed[5] !in 0.0..1.0 || packed[9] !in 0.0..1.0
        ) return null

        val essential = parseIntegratedModel(packed, 10, INITIAL_PAIR_HEADER_SIZE, matchCount)
            ?: return null
        val fundamental = parseIntegratedModel(
            packed, 16, INITIAL_PAIR_HEADER_SIZE + matchCount, matchCount
        ) ?: return null
        val homography = parseIntegratedModel(
            packed, 22, INITIAL_PAIR_HEADER_SIZE + 2 * matchCount, matchCount,
            residualOffset = 28
        ) ?: return null
        val selectedMask = parseMask(packed, INITIAL_PAIR_HEADER_SIZE + 3 * matchCount, matchCount)
            ?: return null
        if (selectedMask.count { it } != numInliers) return null
        val selectedModelMask = when (selectedModel) {
            InlierModel.NONE -> BooleanArray(matchCount)
            InlierModel.ESSENTIAL -> essential.inlierMask
            InlierModel.FUNDAMENTAL -> fundamental.inlierMask
            InlierModel.HOMOGRAPHY -> homography.inlierMask
        }
        if (!selectedMask.contentEquals(selectedModelMask)) return null

        val poseSuccess = packed[5] == 1.0
        val positiveDepthCount = packed[7].toInt()
        val selectedCandidate = packed[6].toInt()
        val medianAngle = packed[8]
        if (positiveDepthCount !in 0..matchCount || selectedCandidate !in 0..3 ||
            !medianAngle.isFinite()
        ) return null
        val cheiralityMask = parseMask(
            packed, INITIAL_PAIR_HEADER_SIZE + 4 * matchCount, matchCount
        ) ?: return null
        if (cheiralityMask.count { it } != positiveDepthCount ||
            (0 until matchCount).any { cheiralityMask[it] && !selectedMask[it] }
        ) return null
        val pose = if (poseSuccess) {
            val rotation = packed.copyOfRange(29, 38)
            val translation = packed.copyOfRange(38, 41)
            if (rotation.any { !it.isFinite() } || translation.any { !it.isFinite() }) {
                return null
            }
            PoseEstimate(
                rotationRowMajor = rotation,
                translation = translation,
                selectedCandidate = selectedCandidate,
                cheiralityMask = cheiralityMask,
                inlierIndices = cheiralityMask.indices.filter { cheiralityMask[it] }
            )
        } else {
            if (positiveDepthCount != 0 || cheiralityMask.any { it }) return null
            null
        }
        return InitialPairEstimate(
            configuration = configuration,
            selectedModel = selectedModel,
            poseSource = poseSource,
            numInliers = numInliers,
            essential = essential,
            fundamental = fundamental,
            homography = homography,
            selectedInlierMask = selectedMask,
            pose = pose,
            medianTriangulationAngleRadians = medianAngle,
            passesInitialPairGates = packed[9] == 1.0
        )
    }

    private fun parseIntegratedModel(
        packed: DoubleArray,
        headerStart: Int,
        maskStart: Int,
        matchCount: Int,
        residualOffset: Int? = null
    ): IntegratedModelEstimate? {
        if (packed.sliceArray(headerStart until headerStart + 6).any { !it.isFinite() } ||
            packed[headerStart] !in 0.0..1.0 || packed[headerStart + 5] !in 0.0..1.0
        ) return null
        val success = packed[headerStart] == 1.0
        val inlierCount = packed[headerStart + 1].toInt()
        val trials = packed[headerStart + 2].toInt()
        val localTrials = packed[headerStart + 3].toInt()
        val acceptedRefits = packed[headerStart + 4].toInt()
        if (inlierCount !in 0..matchCount || trials !in 0..MAX_RANSAC_TRIALS ||
            localTrials !in 0..(MAX_RANSAC_TRIALS * MAX_LOCAL_OPTIMIZATION_TRIALS) ||
            acceptedRefits !in 0..localTrials
        ) return null
        val mask = parseMask(packed, maskStart, matchCount) ?: return null
        if (mask.count { it } != inlierCount || (!success && inlierCount != 0)) return null
        val residualSum = residualOffset?.let { packed[it] } ?: 0.0
        if (!residualSum.isFinite()) return null
        return IntegratedModelEstimate(
            success = success,
            inlierCount = inlierCount,
            inlierMask = mask,
            numTrials = trials,
            localOptimizationTrials = localTrials,
            acceptedLocalRefits = acceptedRefits,
            bestModelWasLocal = packed[headerStart + 5] == 1.0,
            residualSum = residualSum
        )
    }

    private fun parseMask(packed: DoubleArray, start: Int, count: Int): BooleanArray? {
        val mask = BooleanArray(count)
        for (i in 0 until count) {
            val value = packed[start + i]
            if (value != 0.0 && value != 1.0) return null
            mask[i] = value == 1.0
        }
        return mask
    }

    private fun parseModel(
        packed: DoubleArray,
        blockStart: Int,
        matchCount: Int,
        minimumSupport: Int
    ): ModelEstimate? {
        val headerEnd = blockStart + MODEL_HEADER_SIZE
        if (packed.sliceArray(blockStart until headerEnd).any { !it.isFinite() } ||
            packed[blockStart + 1] !in 0.0..1.0
        ) return null
        val success = packed[blockStart + 1] == 1.0
        val inlierCount = packed[blockStart + 2].toInt()
        val trials = packed[blockStart + 3].toInt()
        val localTrials = packed[blockStart + 5].toInt()
        val acceptedRefits = packed[blockStart + 6].toInt()
        if (inlierCount !in 0..matchCount || trials !in 0..MAX_RANSAC_TRIALS ||
            localTrials !in 0..(MAX_RANSAC_TRIALS * MAX_LOCAL_OPTIMIZATION_TRIALS) ||
            acceptedRefits !in 0..localTrials ||
            packed[blockStart + 7] !in 0.0..1.0
        ) return null
        val maskStart = headerEnd
        val mask = BooleanArray(matchCount)
        for (i in 0 until matchCount) {
            val value = packed[maskStart + i]
            if (value != 0.0 && value != 1.0) return null
            mask[i] = value == 1.0
        }
        if (mask.count { it } != inlierCount) return null

        val modelValues = packed.copyOfRange(blockStart + 8, blockStart + 17)
        val model = if (success) modelValues else null
        if (success && (inlierCount < minimumSupport || model!!.any { !it.isFinite() })) {
            return null
        }
        if (!success && inlierCount != 0) return null
        return ModelEstimate(
            success = success,
            modelRowMajor = model,
            inlierMask = mask,
            inlierCount = inlierCount,
            numTrials = trials,
            residualSum = packed[blockStart + 4],
            localOptimizationTrials = localTrials,
            acceptedLocalRefits = acceptedRefits,
            bestModelWasLocal = packed[blockStart + 7] == 1.0
        )
    }

    private external fun nativeEstimate(
        firstPixelsXY: DoubleArray,
        secondPixelsXY: DoubleArray,
        cameraMatrixRowMajor: DoubleArray,
        maxErrorPixels: Double
    ): DoubleArray?

    private external fun nativeRecoverPose(
        essentialRowMajor: DoubleArray,
        firstPixelsXY: DoubleArray,
        secondPixelsXY: DoubleArray,
        cameraMatrixRowMajor: DoubleArray,
        epipolarInlierMask: IntArray
    ): DoubleArray?

    private external fun nativeEstimateInitialPair(
        firstPixelsXY: DoubleArray,
        secondPixelsXY: DoubleArray,
        cameraMatrixRowMajor: DoubleArray,
        bothCamerasHaveFocalPriors: Boolean,
        maxErrorPixels: Double,
        minimumInliers: Int,
        minimumEssentialToFundamentalRatio: Double,
        maximumHomographyRatio: Double
    ): DoubleArray?

    private const val RESULT_VERSION = 1.0
    private const val MODEL_HEADER_SIZE = 17
    private const val POSE_HEADER_SIZE = 15
    private const val INITIAL_PAIR_HEADER_SIZE = 41
    const val MINIMUM_ESSENTIAL_MATCHES = 5
    const val MINIMUM_FUNDAMENTAL_MATCHES = 7
    const val MINIMUM_MATCHES = 7
    const val MAX_INPUT_MATCHES = 2500
    const val MAX_RANSAC_TRIALS = 10000
    const val MAX_LOCAL_OPTIMIZATION_TRIALS = 10
}
