package com.trackstudio.app.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvException
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.KeyPoint
import kotlin.math.sqrt

/**
 * Two-view geometry evaluator for initialization candidates.
 *
 * The native E/F path uses the pinned source-faithful minimal solvers,
 * normalized-ray/pixel Sampson residuals, LO-RANSAC, and four-way essential
 * pose cheirality selection. OpenCV remains responsible only for the competing
 * homography model. The pure Kotlin selector decides seed suitability.
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
        val epipolarEstimate = SourceFaithfulTwoViewEstimator.estimate(
            firstPixelsXY, secondPixelsXY, cameraValues, ransacThresholdPx
        ) ?: return rejected(firstFrame, secondFrame, "Native source-faithful E/F estimation failed", matchCount)

        val points1 = MatOfPoint2f()
        val points2 = MatOfPoint2f()
        val homographyMask = Mat()
        try {
            points1.fromList(matchedFirst)
            points2.fromList(matchedSecond)

            val essentialSucceeded = epipolarEstimate.essential.success
            val essentialInlierCount = epipolarEstimate.essential.inlierCount
            val fundamentalSucceeded = epipolarEstimate.fundamental.success
            val fundamentalInlierCount = epipolarEstimate.fundamental.inlierCount
            var homographySucceeded = false
            var homographyInlierCount = 0
            try {
                val estimatedHomography = Calib3d.findHomography(
                    points1, points2, Calib3d.RANSAC, HOMOGRAPHY_THRESHOLD_PX,
                    homographyMask, 10000, 0.999
                )
                try {
                    homographySucceeded = estimatedHomography.rows() == 3 &&
                        estimatedHomography.cols() == 3
                    if (homographySucceeded) {
                        val rawSupport = countMaskInliers(homographyMask, matchCount)
                        val homographyValues = OpenCvMatrixReadback.readCv64F(
                            estimatedHomography, 3, 3
                        )
                        homographyInlierCount = if (homographyValues == null) {
                            rawSupport
                        } else {
                            conservativeHomographySupportCount(
                                matchedFirst.toTypedArray(),
                                matchedSecond.toTypedArray(),
                                homographyValues,
                                OpenCvTwoViewLocalOptimizer.inliersFromMask(
                                    homographyMask, matchCount
                                ),
                                rawSupport,
                                HOMOGRAPHY_THRESHOLD_PX
                            )
                        }
                    }
                } finally {
                    estimatedHomography.release()
                }
            } catch (_: CvException) {
                homographySucceeded = false
                homographyInlierCount = 0
            }

            val modelDecision = ColmapTwoViewModelArbitrator.arbitrate(
                essentialSucceeded = essentialSucceeded,
                essentialInliers = essentialInlierCount,
                fundamentalSucceeded = fundamentalSucceeded,
                fundamentalInliers = fundamentalInlierCount,
                homographySucceeded = homographySucceeded,
                homographyInliers = homographyInlierCount,
                minimumEssentialToFundamentalRatio =
                    policy.minimumEssentialToFundamentalInlierRatio,
                maximumHomographyRatio = policy.planarDominanceRatio
            )
            if (modelDecision.configuration !=
                ColmapTwoViewModelArbitrator.Configuration.CALIBRATED &&
                modelDecision.configuration !=
                ColmapTwoViewModelArbitrator.Configuration.UNCALIBRATED
            ) {
                return rejected(
                    firstFrame,
                    secondFrame,
                    "E/F/H arbitration classified the pair as ${modelDecision.configuration}",
                    matchCount
                )
            }
            val selectedModelMask = when (modelDecision.inlierModel) {
                ColmapTwoViewModelArbitrator.InlierModel.ESSENTIAL ->
                    epipolarEstimate.essential.inlierMask
                ColmapTwoViewModelArbitrator.InlierModel.FUNDAMENTAL ->
                    epipolarEstimate.fundamental.inlierMask
                ColmapTwoViewModelArbitrator.InlierModel.HOMOGRAPHY,
                ColmapTwoViewModelArbitrator.InlierModel.NONE ->
                    return rejected(firstFrame, secondFrame, "No calibrated epipolar inlier mask", matchCount)
            }
            if (selectedModelMask.count { it } != modelDecision.inlierCount) {
                return rejected(firstFrame, secondFrame, "Invalid selected inlier mask", matchCount)
            }

            val poseEssential = if (modelDecision.configuration ==
                ColmapTwoViewModelArbitrator.Configuration.UNCALIBRATED
            ) {
                val fundamental = epipolarEstimate.fundamental.modelRowMajor
                    ?: return rejected(firstFrame, secondFrame, "Missing fundamental model", matchCount)
                EpipolarGeometryMath.essentialFromFundamental(
                    fundamental, cameraValues, cameraValues
                ) ?: return rejected(firstFrame, secondFrame, "Could not convert F to E", matchCount)
            } else {
                epipolarEstimate.essential.modelRowMajor
                    ?: return rejected(firstFrame, secondFrame, "Missing essential model", matchCount)
            }
            val pose = SourceFaithfulTwoViewEstimator.recoverPose(
                poseEssential,
                firstPixelsXY,
                secondPixelsXY,
                cameraValues,
                selectedModelMask
            ) ?: return rejected(firstFrame, secondFrame, "Cheirality check failed", matchCount)

            val inlierIndices = pose.inlierIndices
            val inlierCount = inlierIndices.size
            if (inlierCount != pose.cheiralityMask.count { it } || inlierCount == 0) {
                return rejected(firstFrame, secondFrame, "Invalid cheirality inlier mask", matchCount)
            }
            val t = pose.translation
            val relativeRotation = pose.rotationRowMajor
            val essentialToFundamentalRatio = if (fundamentalInlierCount == 0) {
                Double.MAX_VALUE
            } else {
                essentialInlierCount.toDouble() / fundamentalInlierCount
            }
            val homographyReferenceInliers = if (modelDecision.configuration ==
                ColmapTwoViewModelArbitrator.Configuration.UNCALIBRATED
            ) {
                modelDecision.inlierCount
            } else {
                essentialInlierCount
            }
            val homographyRatio =
                homographyInlierCount.toDouble() / homographyReferenceInliers.coerceAtLeast(1)
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
            val medianParallax = percentile(parallax, 0.50)
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
                cheiralityRatio = inlierCount.toDouble() / modelDecision.inlierCount.coerceAtLeast(1),
                homographyInlierRatio = homographyRatio,
                forwardMotionRatio = forwardMotionRatio,
                relativeRotation = relativeRotation,
                relativeTranslationDirection = relativeTranslationDirection,
                verifiedMatches = inlierIndices.map { matched[it] },
                essentialToFundamentalInlierRatio = essentialToFundamentalRatio,
                fundamentalPoseRecovered = modelDecision.configuration ==
                    ColmapTwoViewModelArbitrator.Configuration.UNCALIBRATED
            )
            return InitialPairSelector.evaluate(candidate, policy)
        } finally {
            points1.release()
            points2.release()
            homographyMask.release()
        }
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
