package com.trackstudio.app.tracking

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class GlobalRotationAveragerTest {
    @Test fun recoversConsistentRotationsFromRedundantViewGraph() {
        val expected = mapOf(0 to rot(0.0,0.0,0.0), 1 to rot(0.0,0.12,0.0), 2 to rot(0.08,0.25,0.02), 3 to rot(0.12,0.38,-0.03))
        val edges = listOf(edge(0,1,expected),edge(1,2,expected),edge(2,3,expected),edge(0,2,expected),edge(1,3,expected))
        val solved = GlobalRotationAverager.solve(expected.keys.toList(), edges)!!
        expected.forEach { (frame, truth) ->
            val error = relativeAngle(solved.rotationsByFrame.getValue(frame), truth)
            assertTrue("frame=$frame error=$error", error < 0.01)
        }
        assertEquals(5, solved.acceptedEdges.size)
    }

    @Test fun robustlyRejectsBadRelativeRotationEdge() {
        val expected = mapOf(0 to rot(0.0,0.0,0.0), 1 to rot(0.0,0.1,0.0), 2 to rot(0.0,0.2,0.0), 3 to rot(0.0,0.3,0.0))
        val edges = mutableListOf(edge(0,1,expected),edge(1,2,expected),edge(2,3,expected),edge(0,2,expected),edge(0,3,expected))
        edges += GlobalRotationAverager.Edge(0,3,rot(0.0,1.1,0.0),120)
        val solved = GlobalRotationAverager.solve(expected.keys.toList(), edges)!!
        assertTrue(solved.rejectedEdges.isNotEmpty())
        assertTrue(relativeAngle(solved.rotationsByFrame.getValue(3), expected.getValue(3)) < 0.08)
    }

    @Test fun ignoresDisconnectedFramesAndUsesLargestComponent() {
        val expected = mapOf(0 to rot(0.0,0.0,0.0), 1 to rot(0.0,0.1,0.0), 2 to rot(0.0,0.2,0.0), 8 to rot(0.0,0.0,0.0))
        val solved = GlobalRotationAverager.solve(expected.keys.toList(), listOf(edge(0,1,expected),edge(1,2,expected)))!!
        assertEquals(setOf(0,1,2), solved.rotationsByFrame.keys)
    }

    @Test fun rejectsMalformedRelativeRotationMatrices() {
        val invalid = GlobalRotationAverager.Edge(
            0, 1,
            doubleArrayOf(2.0,0.0,0.0, 0.0,2.0,0.0, 0.0,0.0,2.0),
            100
        )
        assertNull(GlobalRotationAverager.solve(listOf(0, 1), listOf(invalid)))
    }

    @Test fun nearHalfTurnResidualIsNotCollapsedToZero() {
        val halfTurn = rot(PI, 0.0, 0.0)
        val identity = rot(0.0, 0.0, 0.0)
        val edges = listOf(
            GlobalRotationAverager.Edge(0, 1, halfTurn, 100),
            GlobalRotationAverager.Edge(1, 2, identity, 100),
            GlobalRotationAverager.Edge(0, 2, identity, 1)
        )

        val solved = GlobalRotationAverager.solve(
            frameIndices = listOf(0, 1, 2),
            edges = edges,
            l1Iterations = 0,
            irlsIterations = 1,
            rotationOutlierDegrees = 5.0
        )!!

        assertTrue(solved.rejectedEdges.any { it.firstFrame == 0 && it.secondFrame == 2 })
        assertTrue(relativeAngle(solved.rotationsByFrame.getValue(1), halfTurn) < 1e-5)
        assertTrue(relativeAngle(solved.rotationsByFrame.getValue(2), halfTurn) < 1e-5)
    }

    @Test fun reversingEdgeDirectionAndPermutingInputDoesNotChangeSolution() {
        val truth = mapOf(
            0 to rot(0.0, 0.0, 0.0),
            1 to rot(0.15, -0.2, 0.1),
            2 to rot(-0.1, 0.35, 0.05),
            3 to rot(0.25, 0.1, -0.2)
        )
        val edges = listOf(
            edge(0, 1, truth), edge(1, 2, truth), edge(2, 3, truth),
            edge(0, 2, truth), edge(1, 3, truth), edge(0, 3, truth)
        )
        val reversed = edges.reversed().map {
            GlobalRotationAverager.Edge(
                it.secondFrame, it.firstFrame, transpose(it.relativeRotation), it.inlierCount
            )
        }

        val first = GlobalRotationAverager.solve(truth.keys.toList(), edges)!!
        val second = GlobalRotationAverager.solve(truth.keys.toList(), reversed)!!
        truth.keys.forEach { frame ->
            assertTrue(
                "frame=$frame",
                relativeAngle(first.rotationsByFrame.getValue(frame), second.rotationsByFrame.getValue(frame)) < 1e-8
            )
        }
        assertEquals(first.acceptedEdges.size, second.acceptedEdges.size)
    }

    @Test fun rejectsInvalidRobustSolverParameters() {
        val edges = listOf(edge(0, 1, mapOf(0 to rot(0.0, 0.0, 0.0), 1 to rot(0.0, 0.1, 0.0))))
        assertNull(GlobalRotationAverager.solve(listOf(0, 1), edges, sigmaDegrees = 0.0))
        assertNull(GlobalRotationAverager.solve(listOf(0, 1), edges, rotationOutlierDegrees = 181.0))
    }

    @Test fun solvesLargerSparseViewGraphWithPreconditionedLinearSolver() {
        val frameIds = (0 until 40).toList()
        val expected = frameIds.associateWith { frame ->
            rot(0.0015 * frame, 0.006 * frame, -0.002 * frame)
        }
        val edges = mutableListOf<GlobalRotationAverager.Edge>()
        for (frame in 0 until 39) edges += edge(frame, frame + 1, expected)
        for (frame in 0 until 36 step 3) edges += edge(frame, frame + 4, expected)

        val solved = GlobalRotationAverager.solve(frameIds, edges)!!
        assertEquals(40, solved.rotationsByFrame.size)
        frameIds.forEach { frame ->
            val error = relativeAngle(
                solved.rotationsByFrame.getValue(frame),
                expected.getValue(frame)
            )
            assertTrue("frame=$frame error=$error", error < 0.02)
        }
    }

    @Test fun gemanMcClureWeightsFollowTheConfiguredAngularScale() {
        val sigma = Math.toRadians(5.0)
        val squaredError = 0.04
        val denominator = squaredError + sigma * sigma
        val expected = sigma * sigma / (denominator * denominator)

        assertEquals(expected, GlobalRotationAverager.gemanMcClureWeight(squaredError, sigma), 1e-15)
        assertEquals(1.0 / (sigma * sigma), GlobalRotationAverager.gemanMcClureWeight(0.0, sigma), 1e-12)
        assertTrue(GlobalRotationAverager.gemanMcClureWeight(1.0, sigma) < expected)
    }

    @Test fun averageUpdateUsesMeanEuclideanNormAcrossAllFramesIncludingGauge() {
        val deltas = arrayOf(
            doubleArrayOf(0.003, 0.004, 0.0),
            doubleArrayOf(0.003, 0.0, 0.0),
            doubleArrayOf(0.0, 0.006, 0.0),
            doubleArrayOf(0.0, 0.0, 0.009)
        )

        assertEquals(0.00575, GlobalRotationAverager.averageUpdateMagnitude(deltas), 1e-12)
    }

    @Test fun rootGaugeIsDeterministicAndNormalizesAllExpectedRotations() {
        val truth = mapOf(
            2 to rot(0.12, -0.08, 0.03),
            5 to rot(-0.10, 0.25, 0.04),
            7 to rot(0.16, 0.12, -0.06)
        )
        val edges = listOf(edge(7, 5, truth), edge(5, 2, truth), edge(7, 2, truth))
        val solved = GlobalRotationAverager.solve(listOf(7, 5, 2), edges.reversed())!!
        val rootTranspose = transpose(truth.getValue(2))

        assertEquals(2, solved.rootFrame)
        truth.forEach { (frame, rotation) ->
            val normalized = multiply(rotation, rootTranspose)
            assertTrue(relativeAngle(solved.rotationsByFrame.getValue(frame), normalized) < 1e-7)
        }
    }

    @Test fun maximumInlierTreeUsesKruskalPathAndReversedEdgeDirection() {
        val truth = mapOf(
            2 to rot(0.12, -0.08, 0.03),
            5 to rot(-0.10, 0.25, 0.04),
            7 to rot(0.16, 0.12, -0.06)
        )
        val edges = listOf(
            GlobalRotationAverager.Edge(2, 5, multiply(truth.getValue(5), transpose(truth.getValue(2))), 30),
            GlobalRotationAverager.Edge(7, 5, multiply(truth.getValue(5), transpose(truth.getValue(7))), 20),
            GlobalRotationAverager.Edge(2, 7, multiply(truth.getValue(7), transpose(truth.getValue(2))), 10)
        )

        val solved = GlobalRotationAverager.solve(
            frameIndices = listOf(7, 2, 5),
            edges = edges,
            rotationOutlierDegrees = 180.0,
            l1Iterations = 0,
            irlsIterations = 0
        )!!
        val rootTranspose = transpose(truth.getValue(2))

        assertEquals(2, solved.rootFrame)
        truth.forEach { (frame, rotation) ->
            assertTrue(
                "frame=$frame",
                relativeAngle(solved.rotationsByFrame.getValue(frame), multiply(rotation, rootTranspose)) < 1e-7
            )
        }
    }

    @Test fun equalWeightKruskalTiesAreStableAcrossPermutationAndReversal() {
        val truth = mapOf(
            2 to rot(0.0, 0.0, 0.0),
            5 to rot(0.0, 0.1, 0.0),
            7 to rot(0.0, 0.2, 0.0)
        )
        val consistentFirst = edge(2, 5, truth)
        val consistentSecond = edge(2, 7, truth)
        val conflicting = GlobalRotationAverager.Edge(7, 5, rot(0.0, -0.4, 0.0), 100)
        val edges = listOf(conflicting, consistentSecond, consistentFirst)
        val reversed = edges.reversed().map {
            GlobalRotationAverager.Edge(
                it.secondFrame, it.firstFrame, transpose(it.relativeRotation), it.inlierCount
            )
        }
        val first = GlobalRotationAverager.solve(
            truth.keys.toList(), edges, rotationOutlierDegrees = 180.0,
            l1Iterations = 0, irlsIterations = 0
        )!!
        val second = GlobalRotationAverager.solve(
            truth.keys.toList(), reversed, rotationOutlierDegrees = 180.0,
            l1Iterations = 0, irlsIterations = 0
        )!!

        truth.keys.forEach { frame ->
            assertTrue(
                "frame=$frame",
                relativeAngle(first.rotationsByFrame.getValue(frame), second.rotationsByFrame.getValue(frame)) < 1e-8
            )
            assertTrue(relativeAngle(first.rotationsByFrame.getValue(frame), truth.getValue(frame)) < 1e-7)
        }
    }

    @Test fun l1AndIrlsSystemsUseTheExplicitThreeAxisAnchorRow() {
        val index = mapOf(2 to 0, 5 to 1, 7 to 2)
        val identity = rot(0.0, 0.0, 0.0)
        val edges = listOf(
            GlobalRotationAverager.Edge(2, 5, identity, 100),
            GlobalRotationAverager.Edge(5, 7, identity, 100),
            GlobalRotationAverager.Edge(2, 7, identity, 100)
        )
        val residuals = listOf(
            doubleArrayOf(0.1, -0.2, 0.3),
            doubleArrayOf(-0.5, 0.7, -0.9),
            doubleArrayOf(-0.4, 0.5, -0.6)
        )
        val anchor = doubleArrayOf(0.02, -0.03, 0.04)
        val expected = arrayOf(
            doubleArrayOf(0.02, -0.03, 0.04),
            doubleArrayOf(0.12, -0.23, 0.34),
            doubleArrayOf(-0.38, 0.47, -0.56)
        )

        val l1 = GlobalRotationAverager.solveL1AdmmKotlinFallback(
            3, index, edges, residuals, anchor, 1000
        )!!
        val irls = GlobalRotationAverager.solveTangentUpdates(
            3, index, edges, residuals, doubleArrayOf(1.0, 0.7, 1.4), anchor
        )!!
        for (frame in 0..2) for (axis in 0..2) {
            assertEquals("L1 frame=$frame axis=$axis", expected[frame][axis], l1[frame][axis], 1e-8)
            assertEquals("IRLS frame=$frame axis=$axis", expected[frame][axis], irls[frame][axis], 1e-8)
        }
    }

    @Test fun edgeAndAnchorResidualJacobiansMatchIncidenceSigns() {
        val first = rot(0.12, -0.05, 0.08)
        val second = rot(-0.08, 0.18, -0.04)
        val edge = GlobalRotationAverager.Edge(
            4, 9, multiply(second, transpose(first)), 100
        )
        val epsilon = 1e-7
        for (axis in 0..2) {
            val tangent = DoubleArray(3).also { it[axis] = epsilon }
            val increment = rot(tangent[0], tangent[1], tangent[2])
            val firstResidual = GlobalRotationAverager.relativeResidual(
                edge, mapOf(4 to multiply(first, increment), 9 to second)
            )
            val secondResidual = GlobalRotationAverager.relativeResidual(
                edge, mapOf(4 to first, 9 to multiply(second, increment))
            )
            val gaugeResidual = GlobalRotationAverager.anchorResidual(
                first, multiply(first, increment)
            )
            for (component in 0..2) {
                val expectedEdgeFirst = if (component == axis) -epsilon else 0.0
                val expectedEdgeSecond = if (component == axis) epsilon else 0.0
                val expectedGauge = if (component == axis) epsilon else 0.0
                assertEquals(expectedEdgeFirst, firstResidual[component], 2e-8)
                assertEquals(expectedEdgeSecond, secondResidual[component], 2e-8)
                assertEquals(expectedGauge, gaugeResidual[component], 2e-8)
            }
        }
    }

    private fun edge(a:Int,b:Int,truth:Map<Int,DoubleArray>) =
        GlobalRotationAverager.Edge(a,b,multiply(truth.getValue(b),transpose(truth.getValue(a))),100)

    private fun rot(x:Double,y:Double,z:Double):DoubleArray {
        fun axis(v:DoubleArray):DoubleArray {
            val t=sqrt(v.sumOf{it*it}); if(t<1e-12)return doubleArrayOf(1.0,0.0,0.0,0.0)
            val s=sin(t/2)/t
            return doubleArrayOf(cos(t/2),v[0]*s,v[1]*s,v[2]*s)
        }
        val q=axis(doubleArrayOf(x,y,z)); val w=q[0]; val a=q[1]; val b=q[2]; val c=q[3]
        return doubleArrayOf(1-2*(b*b+c*c),2*(a*b-c*w),2*(a*c+b*w),2*(a*b+c*w),1-2*(a*a+c*c),2*(b*c-a*w),2*(a*c-b*w),2*(b*c+a*w),1-2*(a*a+b*b))
    }
    private fun transpose(a:DoubleArray)=doubleArrayOf(a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8])
    private fun multiply(a:DoubleArray,b:DoubleArray):DoubleArray {
        val o=DoubleArray(9); for(r in 0..2)for(c in 0..2)for(k in 0..2)o[r*3+c]+=a[r*3+k]*b[k*3+c]; return o
    }
    private fun relativeAngle(a:DoubleArray,b:DoubleArray):Double {
        val d=multiply(a,transpose(b)); return acos(((d[0]+d[4]+d[8]-1)*0.5).coerceIn(-1.0,1.0))
    }
}
