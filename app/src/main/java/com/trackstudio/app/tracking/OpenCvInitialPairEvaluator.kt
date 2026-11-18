package com.trackstudio.app.tracking

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.KeyPoint
import kotlin.math.sqrt

/**
 * Two-view geometry evaluator for initialization candidates.
 *
 * The native path evaluates E/F/H models, dispatches the selected pose source,
 * and performs source-faithful pose recovery. The pure Kotlin selector applies
 * the app's remaining seed ranking policy.
 *
 * Inputs remain caller-owned. All OpenCV objects created here are released
 * before returning, so candidate evaluation does not retain native memory.
 */
internal object OpenCvInitialPairEvaluator {

    fun evaluate(
        firstFrame: Int,
        secondFrame: Int,
        firstKeypoints: Array<KeyPoint>,
        firstDescriptors: Mat,
        secondKeypoints: Array<KeyPoint>,
        secondDescriptors: Mat,
        cameraMatrix: Mat,
        matcher: SiftDescriptorMatcher,
        frameWidth: Int,
        frameHeight: Int,
        policy: InitialPairPolicy = InitialPairPolicy(),
        cameraFocalLengthHasPrior: Boolean = false,
        maximumMatches: Int = 2500,
        ransacThresholdPx: Double = 4.0
    ): InitialPairEvaluation {
        require(firstFrame >= 0 && secondFrame > firstFrame)
        require(frameWidth > 0 && frameHeight > 0)
        require(!cameraMatrix.empty() && cameraMatrix.rows() == 3 && cameraMatrix.cols() == 3)
        if (firstDescriptors.empty() || secondDescriptors.empty() ||
            firstKeypoints.isEmpty() || secondKeypoints.isEmpty()
        ) {
            return rejected(firstFrame, secondFrame, "Missing descriptors or keypoints")
        }
        val matched = matcher.match(
            firstDescriptors,
            secondDescriptors,
            maximumMatches.coerceIn(1, SourceFaithfulTwoViewEstimator.MAX_INPUT_MATCHES)
        ).filter { (query, train) ->
            query in firstKeypoints.indices && train in secondKeypoints.indices
        }
        val matchedFirst = matched.map { firstKeypoints[it.first].pt }
        val matchedSecond = matched.map { secondKeypoints[it.second].pt }
        val matchCount = matchedFirst.size
        if (matchCount < maxOf(policy.minimumInliers, SourceFaithfulTwoViewEstimator.MINIMUM_MATCHES)) {
            return rejected(firstFrame, secondFrame, "Insufficient descriptor matches", matchCount)
        }
        val cameraValues = readCameraMatrix(cameraMatrix)
            ?: return rejected(firstFrame, secondFrame, "Invalid camera matrix readback", matchCount)
        val firstPixelsXY = DoubleArray(matchCount * 2)
        val secondPixelsXY = DoubleArray(matchCount * 2)
        for (index in 0 until matchCount) {
            firstPixelsXY[2 * index] = matchedFirst[index].x
            firstPixelsXY[2 * index + 1] = matchedFirst[index].y
            secondPixelsXY[2 * index] = matchedSecond[index].x
            secondPixelsXY[2 * index + 1] = matchedSecond[index].y
        }
        val geometry = SourceFaithfulTwoViewEstimator.estimateInitialPair(
            firstPixelsXY = firstPixelsXY,
            secondPixelsXY = secondPixelsXY,
            cameraMatrixRowMajor = cameraValues,
            bothCamerasHaveFocalPriors = cameraFocalLengthHasPrior,
            maxErrorPixels = ransacThresholdPx,
            minimumEssentialToFundamentalRatio =
                policy.minimumEssentialToFundamentalInlierRatio,
            maximumHomographyRatio = policy.planarDominanceRatio
        ) ?: return rejected(
            firstFrame, secondFrame, "Native source-faithful E/F/H estimation failed", matchCount
        )
        if (geometry.configuration == SourceFaithfulTwoViewEstimator.Configuration.DEGENERATE) {
            return rejected(firstFrame, secondFrame, "E/F/H arbitration classified the pair as DEGENERATE", matchCount)
        }
        val pose = geometry.pose
            ?: return rejected(firstFrame, secondFrame, "Selected model pose recovery failed", matchCount)
        val homographyPoseRecovered =
            geometry.poseSource == SourceFaithfulTwoViewEstimator.PoseSource.HOMOGRAPHY
        if (homographyPoseRecovered && !geometry.passesInitialPairGates) {
            return rejected(
                firstFrame,
                secondFrame,
                "Homography pose failed the source-derived initial-pair gates",
                matchCount
            )
        }
        if (geometry.selectedInlierMask.count { it } != geometry.numInliers) {
            return rejected(firstFrame, secondFrame, "Invalid selected inlier mask", matchCount)
        }
        val inlierIndices = pose.inlierIndices
        val inlierCount = inlierIndices.size
        if (inlierCount != pose.cheiralityMask.count { it } || inlierCount == 0 ||
            inlierIndices.any { it !in matched.indices || !geometry.selectedInlierMask[it] }
        ) {
            return rejected(firstFrame, secondFrame, "Invalid cheirality inlier mask", matchCount)
        }
        val t = pose.translation
        val relativeRotation = pose.rotationRowMajor
        val essentialToFundamentalRatio = if (geometry.fundamental.inlierCount == 0) {
            Double.MAX_VALUE
        } else {
            geometry.essential.inlierCount.toDouble() / geometry.fundamental.inlierCount
        }
        val homographyReferenceInliers = if (cameraFocalLengthHasPrior) {
            geometry.essential.inlierCount
        } else {
            geometry.fundamental.inlierCount
        }
        val homographyRatio = geometry.homography.inlierCount.toDouble() /
            homographyReferenceInliers.coerceAtLeast(1)
        val coverage = occupiedGridCells(
            inlierIndices.map { matchedFirst[it] }, frameWidth, frameHeight
        )
        val fx = cameraValues[0]
        val skew = cameraValues[1]
        val cx = cameraValues[2]
        val fy = cameraValues[4]
        val cy = cameraValues[5]
        val intrinsics = PinholeIntrinsics(fx, fy, skew, cx, cy)
        val firstPose = CameraPose(
            frameIndex = firstFrame,
            rotation = doubleArrayOf(
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ),
            translation = doubleArrayOf(0.0, 0.0, 0.0),
            status = CameraPoseStatus.REFERENCE
        )
        val secondPose = CameraPose(
            frameIndex = secondFrame,
            rotation = relativeRotation,
            translation = t,
            status = CameraPoseStatus.RELATIVE_ESTIMATE
        )
        val parallax = inlierIndices.mapNotNull { index ->
            MultiViewTriangulationMath.triangulationAngleDegrees(
                first = TriangulationView(
                    matchedFirst[index].x, matchedFirst[index].y, firstPose
                ),
                second = TriangulationView(
                    matchedSecond[index].x, matchedSecond[index].y, secondPose
                ),
                intrinsics = intrinsics
            )
        }.filter { it.isFinite() }.sorted()
        if (parallax.isEmpty()) {
            return rejected(
                firstFrame, secondFrame, "No valid triangulation-angle measurements", matchCount
            )
        }
        val medianParallax = if (homographyPoseRecovered) {
            Math.toDegrees(geometry.medianTriangulationAngleRadians)
        } else {
            percentile(parallax, 0.50)
        }
        val upperQuartileParallax = percentile(parallax, 0.75)
        val translationNorm = sqrt(t.sumOf { it * it })
        if (!translationNorm.isFinite() || translationNorm <= 1e-12 || t.any { !it.isFinite() }) {
            return rejected(firstFrame, secondFrame, "Invalid relative translation", matchCount)
        }
        val forwardMotionRatio = (kotlin.math.abs(t[2]) / translationNorm).coerceIn(0.0, 1.0)
        val relativeTranslationDirection = DoubleArray(3) { t[it] / translationNorm }
        val candidate = InitialPairCandidate(
            firstFrame = firstFrame,
            secondFrame = secondFrame,
            matchedCount = matchCount,
            inlierCount = inlierCount,
            coveredGridCells = coverage,
            medianParallaxDegrees = medianParallax,
            upperQuartileParallaxDegrees = upperQuartileParallax,
            cheiralityRatio = inlierCount.toDouble() / geometry.numInliers.coerceAtLeast(1),
            homographyInlierRatio = homographyRatio,
            forwardMotionRatio = forwardMotionRatio,
            relativeRotation = relativeRotation,
            relativeTranslationDirection = relativeTranslationDirection,
            verifiedMatches = inlierIndices.map { matched[it] },
            essentialToFundamentalInlierRatio = essentialToFundamentalRatio,
            fundamentalPoseRecovered =
                geometry.poseSource == SourceFaithfulTwoViewEstimator.PoseSource.FUNDAMENTAL,
            homographyPoseRecovered = homographyPoseRecovered,
            selectedInlierModel = geometry.selectedModel,
            poseSource = geometry.poseSource
        )
        return InitialPairSelector.evaluate(candidate, policy)
    }

    /**
     * Refines H consensus with at most two deterministic all-inlier fits. The
     * original OpenCV RANSAC count remains a floor because its one-sided
     * reprojection test is not the symmetric transfer metric used by the
     * bounded local optimizer; replacing it could weaken planar rejection.
     */
    internal fun conservativeHomographySupportCount(
        firstPoints: Array<org.opencv.core.Point>,
        secondPoints: Array<org.opencv.core.Point>,
        initialHomography: DoubleArray,
        initialInlierIndices: List<Int>,
        rawOpenCvInlierCount: Int,
        thresholdPx: Double
    ): Int {
        val rawSupport = rawOpenCvInlierCount.coerceIn(0, firstPoints.size)
        if (firstPoints.size != secondPoints.size ||
            firstPoints.size < OpenCvHomographyLocalOptimizer.MINIMUM_FIT_POINTS
        ) return rawSupport

        val refined = OpenCvHomographyLocalOptimizer.refine(
            firstPoints = firstPoints,
            secondPoints = secondPoints,
            initialHomography = initialHomography,
            initialRansacInliers = initialInlierIndices,
            thresholdPx = thresholdPx,
            maximumRefitRounds = OpenCvHomographyLocalOptimizer.MAXIMUM_REFIT_ROUNDS
        )
        return maxOf(rawSupport, refined?.inlierIndices?.size ?: 0)
    }

    private fun rejected(
        first: Int,
        second: Int,
        reason: String,
        matches: Int = 0
    ): InitialPairEvaluation {
        val candidate = InitialPairCandidate(
            firstFrame = first,
            secondFrame = second,
            matchedCount = matches,
            inlierCount = 0,
            coveredGridCells = 0,
            medianParallaxDegrees = 0.0,
            upperQuartileParallaxDegrees = 0.0,
            cheiralityRatio = 0.0,
            homographyInlierRatio = 0.0,
            forwardMotionRatio = 1.0
        )
        return InitialPairEvaluation(candidate, false, 0.0, reason)
    }

    private fun occupiedGridCells(
        points: List<org.opencv.core.Point>,
        width: Int,
        height: Int
    ): Int {
        if (width <= 0 || height <= 0) return 0
        val cells = BooleanArray(16)
        for (point in points) {
            val x = (point.x * 4.0 / width).toInt().coerceIn(0, 3)
            val y = (point.y * 4.0 / height).toInt().coerceIn(0, 3)
            cells[y * 4 + x] = true
        }
        return cells.count { it }
    }

    private fun percentile(sorted: List<Double>, fraction: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val index = (fraction * (sorted.size - 1)).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private const val HOMOGRAPHY_THRESHOLD_PX = 4.0

    private fun readCameraMatrix(cameraMatrix: Mat): DoubleArray? {
        if (cameraMatrix.type() == CvType.CV_64F && cameraMatrix.channels() == 1) {
            return OpenCvMatrixReadback.readCv64F(cameraMatrix, 3, 3)
        }
        val converted = Mat()
        return try {
            cameraMatrix.convertTo(converted, CvType.CV_64F)
            OpenCvMatrixReadback.readCv64F(converted, 3, 3)
        } finally {
            converted.release()
        }
    }

    private fun countMaskInliers(mask: Mat, expectedSize: Int): Int {
        if (mask.empty() || mask.channels() != 1 || mask.total().toInt() != expectedSize) return 0
        return (0 until expectedSize).count { maskValue(mask, it, expectedSize) }
    }

    private fun maskValue(mask: Mat, index: Int, expectedSize: Int): Boolean {
        if (mask.empty() || mask.channels() != 1) return false
        val coordinate = OpenCvMaskLayout.coordinates(
            mask.rows(), mask.cols(), index, expectedSize
        ) ?: return false
        return mask.get(coordinate.first, coordinate.second)
            ?.firstOrNull()?.let { it != 0.0 } == true
    }
}
