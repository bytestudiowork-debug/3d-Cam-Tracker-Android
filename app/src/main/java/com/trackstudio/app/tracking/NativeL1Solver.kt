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
        maxIterations: Int
    ): DoubleArray? {
        if (!isAvailable) return null
        val result = nativeSolve(
            frameCount, firstFrames, secondFrames, residuals, maxIterations
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
        maxIterations: Int
    ): DoubleArray?
}
