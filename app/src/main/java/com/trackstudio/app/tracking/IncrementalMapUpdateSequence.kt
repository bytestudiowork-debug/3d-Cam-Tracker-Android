package com.trackstudio.app.tracking

internal object IncrementalMapUpdateSequence {
    suspend fun flushThenRetry(
        flush: () -> Unit,
        retry: suspend () -> Unit
    ) {
        flush()
        retry()
    }
}
