package com.trackstudio.app.tracking

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.core.Mat

/**
 * Public, UI-independent entry point for TrackStudio camera solving.
 *
 * The solver selects a well-conditioned initialization pair, builds a sparse
 * map, then registers subsequent cameras with PnP in that shared map. The
 * arbitrary unit baseline is a reconstruction gauge, not a metric distance.
 */
class CameraTrackingEngine(
    private val maxFeatures: Int = 1800,
    private val ratioThreshold: Float = 0.8f,
    private val cameraFocalLengthHasPrior: Boolean = false
) : ReconstructionEngine {
    init {
        require(maxFeatures in 400..5000)
        require(ratioThreshold.isFinite() && ratioThreshold > 0f && ratioThreshold < 1f)
    }

    override suspend fun solve(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit
    ): CameraTrackingResult = solveWithFocalPrior(
        frameCount, cameraMatrix, frameProvider, onProgress, cameraFocalLengthHasPrior
    )

    override suspend fun solveWithKnownCameraIntrinsics(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit
    ): CameraTrackingResult = solveWithFocalPrior(
        frameCount, cameraMatrix, frameProvider, onProgress, focalLengthHasPrior = true
    )

    private suspend fun solveWithFocalPrior(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit,
        focalLengthHasPrior: Boolean
    ): CameraTrackingResult = withContext(Dispatchers.Default) {
        IncrementalSfMSolver(maxFeatures, ratioThreshold, focalLengthHasPrior).solve(
            frameCount = frameCount,
            cameraMatrix = cameraMatrix,
            frameProvider = frameProvider,
            onProgress = onProgress
        )
    }
}

data class CameraTrackingProgress(
    /** Number of source frames processed in the current pass. */
    val frameIndex: Int,
    val frameCount: Int,
    val phase: String,
    val overallFraction: Float? = null,
    val completedFrames: Int = frameIndex,
    val totalFrames: Int = frameCount
) {
    val frameLabel: String
        get() = "${completedFrames.coerceIn(0, totalFrames)}/${totalFrames.coerceAtLeast(0)} frames"

    val fraction: Float
        get() = overallFraction?.coerceIn(0f, 1f)
            ?: if (frameCount <= 0) 0f
            else (frameIndex.toFloat() / frameCount).coerceIn(0f, 1f)
}

enum class CameraPoseStatus {
    REFERENCE,
    RELATIVE_ESTIMATE,
    PNP_REGISTERED,
    HELD_LAST_VALID,
    UNREGISTERED
}

/** World-to-camera transform: X_camera = R * X_world + t. */
data class CameraPose(
    val frameIndex: Int,
    val rotation: DoubleArray,
    val translation: DoubleArray,
    val status: CameraPoseStatus = CameraPoseStatus.RELATIVE_ESTIMATE
)

data class CameraIntrinsics(
    val fx: Double,
    val skew: Double,
    val cx: Double,
    val fy: Double,
    val cy: Double
) {
    fun toRowMajorMatrix(): DoubleArray = doubleArrayOf(
        fx, skew, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0
    )
}

data class FrameTrackingDiagnostic(
    val frameIndex: Int,
    val detectedFeatures: Int,
    val matchedFeatures: Int,
    val inlierFeatures: Int,
    val status: String
)

data class CameraTrackingResult(
    val poses: List<CameraPose>,
    val diagnostics: List<FrameTrackingDiagnostic>,
    val processedFrames: Int,
    val solvedFramePairs: Int,
    val totalMatches: Int,
    val totalInliers: Int,
    val inlierRatio: Double,
    val isCameraMotionEstimated: Boolean,
    val featureTracks: List<CameraFeatureTrack> = emptyList(),
    val landmarks: List<CameraLandmark> = emptyList(),
    val triangulationStats: TriangulationStats = TriangulationStats(),
    val initialPairFrames: Pair<Int, Int>? = null,
    val registeredCameraCount: Int = 0,
    /** Global-path rotations after view-graph averaging and accepted bundle-adjustment passes. */
    val globalRotationsByFrame: Map<Int, DoubleArray> = emptyMap(),
    val viewGraphEdgeCount: Int = 0,
    val viewGraphAcceptedEdgeCount: Int = 0,
    val globalViewGraphFocalCalibrationApplied: Boolean = false,
    val globalViewGraphInitialFocalLength: Double = Double.NaN,
    val globalViewGraphFocalLength: Double = Double.NaN,
    val globalViewGraphRejectedPairCount: Int = 0,
    val globalViewGraphCalibrationStatus: String = "Not run",
    val globalRotationIterations: Int = 0,
    val globalPositioningApplied: Boolean = false,
    val globalPositioningInitialCost: Double = Double.NaN,
    val globalPositioningFinalCost: Double = Double.NaN,
    val globalPositioningIterations: Int = 0,
    val globalPositioningStatus: String = "Not run",
    val globalBundleAdjustmentApplied: Boolean = false,
    val globalBundleAdjustmentInitialCost: Double = Double.NaN,
    val globalBundleAdjustmentFinalCost: Double = Double.NaN,
    val globalBundleAdjustmentIterations: Int = 0,
    val globalBundleAdjustmentStatus: String = "Not run",
    val globalBundleAdjustmentPositiveDepthRatio: Double = Double.NaN,
    val globalBundleAdjustmentPrincipalPointRefinementEnabled: Boolean = false,
    val jointBundleAdjustmentApplied: Boolean = false,
    val jointBundleAdjustmentInitialCost: Double = Double.NaN,
    val jointBundleAdjustmentFinalCost: Double = Double.NaN,
    val jointBundleAdjustmentIterations: Int = 0,
    val jointBundleAdjustmentStatus: String = "Not run",
    val jointBundleAdjustmentPositiveDepthRatio: Double = Double.NaN,
    val jointBundleAdjustmentPrincipalPointRefinementEnabled: Boolean = false,
    val inputCameraCalibration: CameraCalibration? = null,
    val observationCoordinateModel: CameraObservationCoordinateModel =
        CameraObservationCoordinateModel.PINHOLE_PIXELS_ASSUMED,
    val refinedCameraIntrinsics: CameraIntrinsics? = null,
    val observationCameraIntrinsics: CameraIntrinsics? = null,
    val postRetriangulationBundleAdjustmentApplied: Boolean = false,
    val postRetriangulationBundleAdjustmentInitialCost: Double = Double.NaN,
    val postRetriangulationBundleAdjustmentFinalCost: Double = Double.NaN,
    val postRetriangulationBundleAdjustmentIterations: Int = 0,
    val postRetriangulationBundleAdjustmentStatus: String = "Not run",
    val globalCovisibilityPruningApplied: Boolean = false,
    val globalCovisibilityClusterIdByFrame: Map<Int, Int> = emptyMap(),
    val globalCovisibilityClusterCount: Int = 0,
    val globalCovisibilityStrongThreshold: Double = Double.NaN,
    val globalCovisibilityEligiblePairCount: Int = 0,
    /** Frame-pair visibility edges invalidated by final weak-connectivity pruning. */
    val globalCovisibilityInvalidatedPairWeights: Map<Pair<Int, Int>, Long> = emptyMap(),
    val globalCovisibilityBridgeIterations: Int = 0
) {
    val effectiveCameraIntrinsics: CameraIntrinsics?
        get() = refinedCameraIntrinsics ?: observationCameraIntrinsics ?: inputCameraCalibration?.let { calibration ->
            CameraIntrinsics(
                fx = calibration.fx,
                skew = calibration.skew,
                cx = calibration.cx,
                fy = calibration.fy,
                cy = calibration.cy
            )
        }
}
