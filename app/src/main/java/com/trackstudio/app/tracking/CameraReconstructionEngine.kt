package com.trackstudio.app.tracking

import org.opencv.core.Mat

class CameraReconstructionOptions(
    val customMaxFeatures: Int = 1800,
    val globalMaxFeatures: Int = 1400,
    val ratioThreshold: Float = 0.8f,
    val maximumPairCount: Int = 420,
    val enableWeakConnectivityPruning: Boolean = false,
    val globalMinTracksPerView: Int = -1,
    val globalMaxTracks: Int = 10_000_000
) {
    init {
        require(customMaxFeatures in 400..5000)
        require(globalMaxFeatures in 500..4000)
        require(ratioThreshold.isFinite() && ratioThreshold > 0f && ratioThreshold < 1f)
        require(maximumPairCount > 0)
        require(globalMaxTracks >= 0)
    }
}

class CameraReconstructionEngine(
    mode: ReconstructionMode,
    options: CameraReconstructionOptions = CameraReconstructionOptions()
) : ReconstructionEngine {
    private val delegate: ReconstructionEngine = when (mode) {
        ReconstructionMode.CUSTOM -> CameraTrackingEngine(
            maxFeatures = options.customMaxFeatures,
            ratioThreshold = options.ratioThreshold
        )
        ReconstructionMode.GLOBAL -> GlobalReconstructionEngine(
            maxFeatures = options.globalMaxFeatures,
            ratioThreshold = options.ratioThreshold,
            maximumPairCount = options.maximumPairCount,
            enableWeakConnectivityPruning = options.enableWeakConnectivityPruning,
            minTracksPerView = options.globalMinTracksPerView,
            maximumTracks = options.globalMaxTracks
        )
    }

    override suspend fun solve(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit
    ): CameraTrackingResult = delegate.solve(frameCount, cameraMatrix, frameProvider, onProgress)

    override suspend fun solveWithKnownCameraIntrinsics(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit
    ): CameraTrackingResult = delegate.solveWithKnownCameraIntrinsics(
        frameCount, cameraMatrix, frameProvider, onProgress
    )
}
