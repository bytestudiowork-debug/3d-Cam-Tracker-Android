package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.GlobalRotationAverager
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GlomapRotationGraphReducerTest {
    @Test
    fun keepsOnlyTheLargestPostFilterComponentAndOmitsIsolatedFrames() {
        val result = GlomapRotationGraphReducer.keepLargestConnectedComponent(
            registeredFrameIds = listOf(1, 2, 3, 4, 10, 11, 12, 99),
            validEdges = listOf(
                edge(1, 2), edge(2, 3), edge(3, 4),
                edge(10, 11), edge(11, 12)
            )
        )

        assertEquals(listOf(1, 2, 3, 4), result?.frameIds)
        assertEquals(
            listOf(1 to 2, 2 to 3, 3 to 4),
            result?.edges?.map { it.firstFrame to it.secondFrame }
        )
    }

    @Test
    fun equalSizedComponentsUseLowestFrameIdRatherThanInputOrder() {
        val result = GlomapRotationGraphReducer.keepLargestConnectedComponent(
            registeredFrameIds = listOf(1, 2, 10, 11),
            validEdges = listOf(edge(11, 10), edge(2, 1))
        )

        assertEquals(listOf(1, 2), result?.frameIds)
        assertEquals(listOf(2 to 1), result?.edges?.map { it.firstFrame to it.secondFrame })
    }

    @Test
    fun ignoresEdgesOutsideTheSolvedFrameSet() {
        val result = GlomapRotationGraphReducer.keepLargestConnectedComponent(
            registeredFrameIds = listOf(1, 2),
            validEdges = listOf(edge(1, 2), edge(2, 50), edge(50, 51), edge(1, 1))
        )

        assertEquals(listOf(1, 2), result?.frameIds)
        assertEquals(listOf(1 to 2), result?.edges?.map { it.firstFrame to it.secondFrame })
    }

    @Test
    fun returnsNullWhenNoValidPairCanFormAComponent() {
        assertNull(
            GlomapRotationGraphReducer.keepLargestConnectedComponent(
                registeredFrameIds = listOf(1, 2),
                validEdges = listOf(edge(1, 1), edge(2, 50))
            )
        )
    }

    private fun edge(first: Int, second: Int) = GlobalRotationAverager.Edge(
        firstFrame = first,
        secondFrame = second,
        relativeRotation = doubleArrayOf(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ),
        inlierCount = 20
    )
}
