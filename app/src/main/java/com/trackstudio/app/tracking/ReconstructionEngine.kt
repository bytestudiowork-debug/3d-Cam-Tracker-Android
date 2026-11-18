package com.trackstudio.app.tracking

import org.opencv.core.Mat

/**
 * UI-independent reconstruction contract shared by the custom and GLOMAP engines.
 *
 * The frame provider owns decoding; an engine must release every Mat it obtains
 * from the provider after processing it. The camera matrix is borrowed and must
 * not be released or mutated by an implementation.
 */
enum class ReconstructionMode(val label: String) {
    CUSTOM("Custom"),
    GLOBAL("Global")
}

enum class CameraObservationCoordinateModel {
    PINHOLE_PIXELS_ASSUMED,
    CALIBRATED_PINHOLE_PIXELS
}

interface ReconstructionEngine {
    suspend fun solve(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit = {}
    ): CameraTrackingResult

    /**
     * Solves with caller-supplied camera intrinsics treated as a focal-length prior.
     * Engines that do not estimate focal length may use the ordinary solve path.
     */
    suspend fun solveWithKnownCameraIntrinsics(
        frameCount: Int,
        cameraMatrix: Mat,
        frameProvider: suspend (Int) -> Mat?,
        onProgress: suspend (CameraTrackingProgress) -> Unit = {}
    ): CameraTrackingResult = solve(frameCount, cameraMatrix, frameProvider, onProgress)
}
