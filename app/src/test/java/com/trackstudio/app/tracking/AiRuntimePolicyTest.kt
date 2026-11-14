package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Test

class AiRuntimePolicyTest {
    @Test
    fun lowRamDevicesUseConservativeInferenceLimits() {
        val settings = AiRuntimePolicy.select(
            processorCount = 8,
            memoryClassMb = 512,
            isLowRamDevice = true
        )

        assertEquals(512, settings.maximumFeatures)
        assertEquals(480, settings.maximumInputDimension)
        assertEquals(1, settings.threadCount)
    }

    @Test
    fun constrainedMemoryDevicesUseModerateInferenceLimits() {
        val settings = AiRuntimePolicy.select(
            processorCount = 4,
            memoryClassMb = 256,
            isLowRamDevice = false
        )

        assertEquals(800, settings.maximumFeatures)
        assertEquals(512, settings.maximumInputDimension)
        assertEquals(2, settings.threadCount)
    }

    @Test
    fun capableDevicesUseHigherLimitsWithoutOversubscribingCpus() {
        val settings = AiRuntimePolicy.select(
            processorCount = 8,
            memoryClassMb = 512,
            isLowRamDevice = false
        )

        assertEquals(1400, settings.maximumFeatures)
        assertEquals(640, settings.maximumInputDimension)
        assertEquals(4, settings.threadCount)
    }

    @Test
    fun smallCpuCountsRemainSingleThreaded() {
        val settings = AiRuntimePolicy.select(
            processorCount = 2,
            memoryClassMb = 512,
            isLowRamDevice = false
        )

        assertEquals(1, settings.threadCount)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsInvalidProcessorCount() {
        AiRuntimePolicy.select(
            processorCount = 0,
            memoryClassMb = 256,
            isLowRamDevice = false
        )
    }
}
