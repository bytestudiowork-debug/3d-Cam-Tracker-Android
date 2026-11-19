package com.trackstudio.app.tracking

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test

class IncrementalMapUpdateSequenceTest {
    @Test
    fun retryObservesTheCompletedMapFlush() = runBlocking {
        var mapGeneration = 0
        var generationSeenByRetry = -1

        IncrementalMapUpdateSequence.flushThenRetry(
            flush = { mapGeneration++ },
            retry = { generationSeenByRetry = mapGeneration }
        )

        assertEquals(1, generationSeenByRetry)
    }
}
