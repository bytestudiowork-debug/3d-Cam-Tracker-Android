package com.trackstudio.app.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.KeyPoint
import org.opencv.core.CvType
import org.opencv.core.CvException
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * OpenCV-backed two-view geometry evaluator for initialization candidates.
 *
 * Descriptor matching proposes correspondences; Essential-matrix RANSAC
 * verifies calibrated epipolar geometry; recoverPose applies cheirality.
 * A competing homography estimates how much of the pair can be explained
 * by a single plane / pure-rotation model. The pure Kotlin selector then
 * decides whether the measurements are suitable for 3D initialization.
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

        val matched = matcher.match(firstDescriptors, secondDescriptors, maximumMatches.coerceAtLeast(1))
            .filter { (query, train) ->
                query in firstKeypoints.indices && train in secondKeypoints.indices
        }
        val matchedFirst = matched.map { firstKeypoints[it.first].pt }
        val matchedSecond = matched.map { secondKeypoints[it.second].pt }

        val matchCount = matchedFirst.size
        if (matchCount < policy.minimumInliers) {
            return rejected(firstFrame, secondFrame, "Insufficient descriptor matches", matchCount)
        }

        val points1 = MatOfPoint2f()
        val points2 = MatOfPoint2f()
        val essentialMask = Mat()
        val fundamentalMask = Mat()
        val cheiralityMask = Mat()
        val homographyMask = Mat()
        val rotation = Mat()
        val translation = Mat()
        var fundamentalForPose: Mat? = null

        try {
            points1.fromList(matchedFirst)
            points2.fromList(matchedSecond)

            val essentialEstimate = Calib3d.findEssentialMat(
                points1,
                points2,
                cameraMatrix,
                Calib3d.RANSAC,
                0.999,
                ransacThresholdPx,
                10000,
                essentialMask
            )
            try {
                val essentialSucceeded = !essentialEstimate.empty() && essentialEstimate.cols() == 3 &&
                    essentialEstimate.rows() >= 3 && essentialEstimate.rows() % 3 == 0
                val essentialInlierCount = if (essentialSucceeded) {
                    countMaskInliers(essentialMask, matchCount)
                } else {
                    0
                }

                val estimatedFundamental = Calib3d.findFundamentalMat(
                    points1, points2, Calib3d.FM_RANSAC,
                    ransacThresholdPx, 0.999, 10000, fundamentalMask
                )
                val fundamentalSucceeded = try {
                    val succeeded = !estimatedFundamental.empty() && estimatedFundamental.cols() == 3 &&
                        estimatedFundamental.rows() >= 3 && estimatedFundamental.rows() % 3 == 0
                    if (succeeded) {
                        fundamentalForPose = estimatedFundamental.rowRange(0, 3).clone()
                    }
                    succeeded
                } finally {
                    estimatedFundamental.release()
                }
                val fundamentalInlierCount = if (fundamentalSucceeded) {
                    countMaskInliers(fundamentalMask, matchCount)
                } else {
                    0
                }
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
                            homographyInlierCount = countMaskInliers(homographyMask, matchCount)
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
                        "COLMAP E/F/H arbitration classified the pair as ${modelDecision.configuration}",
                        matchCount
                    )
                }
                val selectedModelMask = when (modelDecision.inlierModel) {
                    ColmapTwoViewModelArbitrator.InlierModel.ESSENTIAL -> essentialMask
                    ColmapTwoViewModelArbitrator.InlierModel.FUNDAMENTAL -> fundamentalMask
                    ColmapTwoViewModelArbitrator.InlierModel.HOMOGRAPHY,
                    ColmapTwoViewModelArbitrator.InlierModel.NONE ->
                        return rejected(firstFrame, secondFrame, "No calibrated epipolar inlier mask", matchCount)
                }
                if (countMaskInliers(selectedModelMask, matchCount) != modelDecision.inlierCount) {
                    return rejected(firstFrame, secondFrame, "Invalid COLMAP-selected inlier mask", matchCount)
                }

                val fundamentalPoseEssential = if (modelDecision.configuration ==
                    ColmapTwoViewModelArbitrator.Configuration.UNCALIBRATED
                ) {
                    val fundamental = fundamentalForPose
                        ?: return rejected(firstFrame, secondFrame, "Missing COLMAP fundamental model", matchCount)
                    val fundamentalValues = OpenCvMatrixReadback.readCv64F(fundamental, 3, 3)
                        ?: return rejected(firstFrame, secondFrame, "Invalid fundamental matrix readback", matchCount)
                    val cameraValues = OpenCvMatrixReadback.readCv64F(cameraMatrix, 3, 3)
                        ?: return rejected(firstFrame, secondFrame, "Invalid camera matrix readback", matchCount)
                    val essentialValues = EpipolarGeometryMath.essentialFromFundamental(
                        fundamentalValues, cameraValues, cameraValues
                    ) ?: return rejected(firstFrame, secondFrame, "Could not convert F to E", matchCount)
                    Mat(3, 3, CvType.CV_64F).apply {
                        if (put(0, 0, *essentialValues) != essentialValues.size) {
                            release()
                            return rejected(firstFrame, secondFrame, "Could not store F-derived E", matchCount)
                        }
                    }
                } else {
                    null
                }

                var bestPoseInliers = 0
                var bestPoseMask: Mat? = null
                var bestRotation: Mat? = null
                var bestTranslation: Mat? = null
                var bestEssential: Mat? = null
                try {
                    val poseCandidateCount = if (fundamentalPoseEssential != null) {
                        1
                    } else {
                        essentialEstimate.rows() / 3
                    }
                    for (candidateIndex in 0 until poseCandidateCount) {
                        val candidate = fundamentalPoseEssential?.clone() ?: essentialEstimate
                            .rowRange(candidateIndex * 3, candidateIndex * 3 + 3)
                            .clone()
                        val mask = selectedModelMask.clone()
                        val candidateRotation = Mat()
                        val candidateTranslation = Mat()
                        try {
                            val inliers = Calib3d.recoverPose(
                                candidate, points1, points2, cameraMatrix,
                                candidateRotation, candidateTranslation, mask
                            )
                            if (inliers > bestPoseInliers) {
                                bestPoseMask?.release()
                                bestRotation?.release()
                                bestTranslation?.release()
                                bestPoseInliers = inliers
                                bestPoseMask = mask.clone()
                                bestRotation = candidateRotation.clone()
                                bestTranslation = candidateTranslation.clone()
                                bestEssential?.release()
                                bestEssential = candidate.clone()
                            }
                        } finally {
                            candidate.release()
                            mask.release()
                            candidateRotation.release()
                            candidateTranslation.release()
                        }
                    }

                    val selectedEssential = bestEssential
                    if (selectedEssential != null) {
                        val initialEssential = DoubleArray(9)
                        if (selectedEssential.get(0, 0, initialEssential) ==
                            initialEssential.size * java.lang.Double.BYTES
                        ) {
                            val polished = OpenCvTwoViewLocalOptimizer.refine(
                                matchedFirst.toTypedArray(),
                                matchedSecond.toTypedArray(),
                                cameraMatrix,
                                initialEssential,
                                bestPoseMask?.let {
                                    OpenCvTwoViewLocalOptimizer.inliersFromMask(it, matchCount)
                                }.orEmpty(),
                                ransacThresholdPx,
                                0.999,
                                10000
                            )
                            if (polished != null && polished.inlierIndices.size >= bestPoseInliers) {
                                val polishedMask = OpenCvTwoViewLocalOptimizer.maskFromIndices(
                                    polished.inlierIndices, matchCount
                                )
                                val polishedRotation = Mat(3, 3, CvType.CV_64F)
                                val polishedTranslation = Mat(3, 1, CvType.CV_64F)
                                var retained = false
                                try {
                                    if (polishedRotation.put(0, 0, *polished.rotation) == polished.rotation.size &&
                                        polishedTranslation.put(0, 0, *polished.translation) == polished.translation.size
                                    ) {
                                        bestPoseMask?.release()
                                        bestRotation?.release()
                                        bestTranslation?.release()
                                        bestPoseMask = polishedMask
                                        bestRotation = polishedRotation
                                        bestTranslation = polishedTranslation
                                        bestPoseInliers = polished.inlierIndices.size
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

                    if (bestPoseInliers == 0 || bestPoseMask == null ||
                        bestRotation == null || bestTranslation == null
                    ) {
                        return rejected(firstFrame, secondFrame, "Cheirality check failed", matchCount)
                    }
                    bestPoseMask.copyTo(cheiralityMask)
                    bestRotation.copyTo(rotation)
                    bestTranslation.copyTo(translation)
                } finally {
                    bestPoseMask?.release()
                    bestRotation?.release()
                    bestTranslation?.release()
                    bestEssential?.release()
                    fundamentalPoseEssential?.release()
                }

                val inlierIndices = (0 until matchCount).filter { index ->
                    maskValue(cheiralityMask, index, matchCount)
                }
                if (inlierIndices.size != bestPoseInliers) {
                    return rejected(firstFrame, secondFrame, "Invalid cheirality inlier mask", matchCount)
                }
                val inlierCount = inlierIndices.size
                val t = OpenCvMatrixReadback.readCv64F(translation, 3, 1)
                    ?: return rejected(
                        firstFrame, secondFrame, "Invalid relative translation readback", matchCount
                    )
                val relativeRotation = OpenCvMatrixReadback.readCv64F(rotation, 3, 3)
                    ?: return rejected(
                        firstFrame, secondFrame, "Invalid relative rotation readback", matchCount
                    )
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
                val parallax = inlierIndices.map { index ->
                    rayAngleDegrees(
                        matchedFirst[index], matchedSecond[index],
                        relativeRotation, cameraMatrix
                    )
                }.filter { it.isFinite() }.sorted()

                if (parallax.isEmpty()) {
                    return rejected(firstFrame, secondFrame, "No valid parallax measurements", matchCount)
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
                essentialEstimate.release()
                fundamentalForPose?.release()
            }
        } finally {
            points1.release()
            points2.release()
            essentialMask.release()
            fundamentalMask.release()
            cheiralityMask.release()
            homographyMask.release()
            rotation.release()
            translation.release()
        }
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

    private fun rayAngleDegrees(
        first: org.opencv.core.Point,
        second: org.opencv.core.Point,
        rotationValues: DoubleArray,
        cameraMatrix: Mat
    ): Double {
        val fx = cameraMatrix.get(0, 0)[0]
        val fy = cameraMatrix.get(1, 1)[0]
        val skew = cameraMatrix.get(0, 1)[0]
        val cx = cameraMatrix.get(0, 2)[0]
        val cy = cameraMatrix.get(1, 2)[0]
        if (kotlin.math.abs(fx) < 1e-12 || kotlin.math.abs(fy) < 1e-12) return 0.0
        val intrinsics = PinholeIntrinsics(fx, fy, skew, cx, cy)

        fun normalize(point: org.opencv.core.Point): DoubleArray {
            val ray = CameraProjectionMath.pixelToNormalized(point.x, point.y, intrinsics)
            val n = sqrt(ray[0] * ray[0] + ray[1] * ray[1] + 1.0)
            return doubleArrayOf(ray[0] / n, ray[1] / n, 1.0 / n)
        }

        val a = normalize(first)
        val bCamera = normalize(second)
        // recoverPose returns X2 = R * X1 + t; rotate the second ray
        // back into camera-1 coordinates before comparing viewing rays.
        val b = doubleArrayOf(
            rotationValues[0] * bCamera[0] +
                rotationValues[3] * bCamera[1] +
                rotationValues[6] * bCamera[2],
            rotationValues[1] * bCamera[0] +
                rotationValues[4] * bCamera[1] +
                rotationValues[7] * bCamera[2],
            rotationValues[2] * bCamera[0] +
                rotationValues[5] * bCamera[1] +
                rotationValues[8] * bCamera[2]
        )
        val dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
            .coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(dot))
    }

    private fun percentile(sorted: List<Double>, fraction: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val index = (fraction * (sorted.size - 1)).toInt().coerceIn(0, sorted.lastIndex)
        return sorted[index]
    }

    private const val HOMOGRAPHY_THRESHOLD_PX = 4.0

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
