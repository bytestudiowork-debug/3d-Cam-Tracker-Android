package com.trackstudio.app.tracking

internal object NativeL1Solver {
    val isAvailable: Boolean = try {
        System.loadLibrary("trackstudio_camera")
        true
    } catch (_: UnsatisfiedLinkError) {
        false
    }

    @Volatile
    var successfulInvocations: Int = 0
        private set

    @Synchronized
    fun solve(
        frameCount: Int,
        firstFrames: IntArray,
        secondFrames: IntArray,
        residuals: DoubleArray,
        anchorResidual: DoubleArray,
        maxIterations: Int
    ): DoubleArray? {
        if (!isAvailable || frameCount < 2 || firstFrames.isEmpty() ||
            firstFrames.size != secondFrames.size ||
            residuals.size.toLong() != firstFrames.size.toLong() * 3L ||
            anchorResidual.size != 3 || anchorResidual.any { !it.isFinite() } ||
            residuals.any { !it.isFinite() } || maxIterations <= 0
        ) return null
        val result = nativeSolve(
            frameCount, firstFrames, secondFrames, residuals, anchorResidual, maxIterations
        )
        if (result != null) successfulInvocations++
        return result
    }

    @Synchronized
    fun resetSuccessfulInvocationsForTest() {
        successfulInvocations = 0
    }

    private external fun nativeSolve(
        frameCount: Int,
        firstFrames: IntArray,
        secondFrames: IntArray,
        residuals: DoubleArray,
        anchorResidual: DoubleArray,
        maxIterations: Int
    ): DoubleArray?
}
