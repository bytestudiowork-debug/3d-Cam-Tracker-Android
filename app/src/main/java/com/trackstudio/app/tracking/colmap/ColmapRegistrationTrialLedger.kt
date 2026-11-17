package com.trackstudio.app.tracking.colmap

internal class ColmapRegistrationTrialLedger(
    frameCount: Int,
    private val maximumTrials: Int = ColmapNextImageSelector.DEFAULT_MAX_REGISTRATION_TRIALS
) {
    private val trialCounts = IntArray(frameCount)

    init {
        require(frameCount > 0)
        require(maximumTrials > 0)
    }

    fun beginTrial(frameIndex: Int): Int? {
        require(frameIndex in trialCounts.indices)
        if (trialCounts[frameIndex] >= maximumTrials) return null
        trialCounts[frameIndex] += 1
        return trialCounts[frameIndex]
    }

    operator fun get(frameIndex: Int): Int {
        require(frameIndex in trialCounts.indices)
        return trialCounts[frameIndex]
    }
}
