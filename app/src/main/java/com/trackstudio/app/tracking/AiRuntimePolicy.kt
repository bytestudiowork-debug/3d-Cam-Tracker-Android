package com.trackstudio.app.tracking

import android.app.ActivityManager
import android.content.Context
import kotlin.math.min

/**
 * Selects conservative inference limits from Android's reported memory class,
 * low-RAM classification, and available CPU cores. Geometry code is unaffected.
 */
internal data class AiRuntimeSettings(
    val maximumFeatures: Int,
    val maximumInputDimension: Int,
    val threadCount: Int
) {
    init {
        require(maximumFeatures > 0)
        require(maximumInputDimension >= 32 && maximumInputDimension % 32 == 0)
        require(threadCount > 0)
    }
}

internal object AiRuntimePolicy {
    fun forDevice(context: Context): AiRuntimeSettings {
        val appContext = context.applicationContext
        val activityManager = appContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return select(
            processorCount = Runtime.getRuntime().availableProcessors(),
            memoryClassMb = activityManager.memoryClass,
            isLowRamDevice = activityManager.isLowRamDevice
        )
    }

    internal fun select(
        processorCount: Int,
        memoryClassMb: Int,
        isLowRamDevice: Boolean
    ): AiRuntimeSettings {
        require(processorCount > 0)
        require(memoryClassMb > 0)

        val lowMemory = isLowRamDevice || memoryClassMb < 192
        val constrainedMemory = memoryClassMb < 384
        val maximumFeatures = when {
            lowMemory -> 512
            constrainedMemory -> 800
            else -> 1400
        }
        val maximumInputDimension = when {
            lowMemory -> 480
            constrainedMemory -> 512
            else -> 640
        }
        val threadCount = when {
            lowMemory || processorCount <= 2 -> 1
            processorCount <= 6 -> 2
            else -> min(4, processorCount / 2)
        }

        return AiRuntimeSettings(
            maximumFeatures = maximumFeatures,
            maximumInputDimension = maximumInputDimension,
            threadCount = threadCount
        )
    }
}
