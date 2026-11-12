package com.trackstudio.ai

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.ln

class AssignmentDecoderTest {
    @Test
    fun returnsMutualAssignmentsAboveConfidenceThreshold() {
        val assignment = FloatArray(3 * 3) { ln(0.001f) }
        assignment[0 * 3 + 1] = ln(0.92f)
        assignment[1 * 3 + 0] = ln(0.81f)

        val matches = AssignmentDecoder.decode(
            assignment = assignment,
            firstCount = 2,
            secondCount = 2,
            minimumConfidence = 0.1f
        )

        assertEquals(2, matches.size)
        assertEquals(0, matches[0].firstFeatureIndex)
        assertEquals(1, matches[0].secondFeatureIndex)
        assertEquals(0.92f, matches[0].confidence, 1e-5f)
        assertEquals(1, matches[1].firstFeatureIndex)
        assertEquals(0, matches[1].secondFeatureIndex)
    }

    @Test
    fun rejectsNonMutualAndLowConfidenceAssignments() {
        val assignment = FloatArray(3 * 3) { ln(0.001f) }
        assignment[0 * 3 + 0] = ln(0.4f)
        assignment[0 * 3 + 1] = ln(0.7f)
        assignment[1 * 3 + 1] = ln(0.6f)

        val matches = AssignmentDecoder.decode(
            assignment = assignment,
            firstCount = 2,
            secondCount = 2,
            minimumConfidence = 0.65f
        )

        assertEquals(1, matches.size)
        assertEquals(0, matches[0].firstFeatureIndex)
        assertEquals(1, matches[0].secondFeatureIndex)
    }
}
