package com.trackstudio.app.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs
import kotlin.math.hypot

class LocalConsensusOptimizerTest {
    private data class Sample(val id: Int, val x: Double, val y: Double)
    private data class Line(val slope: Double, val intercept: Double)
    private data class TranslationPose(val x: Double)
    private data class TranslationObservation(val worldX: Double, val depth: Double, val imageX: Double)

    @Test
    fun refitImprovesSupportAndKeepsGrossOutlierOutsideConsensus() {
        val samples = lineSamples()
        val result = optimizeLine(samples)

        assertTrue(result.refitRounds > 0)
        assertEquals(11, result.inlierIndices.size)
        assertFalse(result.inlierIndices.any { samples[it].id == 99 })
        assertEquals(1.5, result.model.slope, 0.01)
        assertEquals(-0.4, result.model.intercept, 0.02)
    }

    @Test
    fun localPoseRefitRecoversSyntheticTranslationAndRejectsGrossOutlier() {
        val focal = 800.0
        val principalX = 640.0
        val truth = 0.18
        val observations = (0 until 12).map { index ->
            val worldX = -1.1 + 0.2 * index
            val depth = 20.0 + 0.7 * index
            val noise = if (index % 2 == 0) 0.04 else -0.04
            val outlier = if (index == 11) 18.0 else 0.0
            TranslationObservation(
                worldX,
                depth,
                focal * (worldX - truth) / depth + principalX + noise + outlier
            )
        }
        val result = LocalConsensusOptimizer.refine(
            initialModel = TranslationPose(0.16),
            observationCount = observations.size,
            initialInliers = (0 until 11).toList(),
            minimumInliers = 8,
            threshold = 1.0,
            maximumRefitRounds = 2,
            fitCandidates = { _, indices ->
                val estimates = indices.map { index ->
                    val observation = observations[index]
                    observation.worldX -
                        (observation.imageX - principalX) * observation.depth / focal
                }
                listOf(TranslationPose(estimates.average()))
            },
            residual = { pose, index ->
                val observation = observations[index]
                abs(
                    focal * (observation.worldX - pose.x) / observation.depth +
                        principalX - observation.imageX
                )
            }
        )!!

        assertEquals(1, result.refitRounds)
        assertEquals(11, result.inlierIndices.size)
        assertFalse(11 in result.inlierIndices)
        assertEquals(truth, result.model.x, 0.01)
    }

    @Test
    fun polishingIsStableUnderObservationPermutation() {
        val original = lineSamples()
        val shuffled = original.reversed()
        val first = optimizeLine(original)
        val second = optimizeLine(shuffled)

        assertEquals(first.model.slope, second.model.slope, 1e-12)
        assertEquals(first.model.intercept, second.model.intercept, 1e-12)
        assertEquals(
            first.inlierIndices.map { original[it].id }.toSet(),
            second.inlierIndices.map { shuffled[it].id }.toSet()
        )
    }

    @Test
    fun rejectsInsufficientOrMalformedConsensusInputs() {
        val model = Line(2.0, 1.0)
        assertEquals(null, LocalConsensusOptimizer.refine(
            initialModel = model,
            observationCount = 2,
            initialInliers = listOf(0, 1),
            minimumInliers = 3,
            threshold = 0.1,
            maximumRefitRounds = 1,
            fitCandidates = { _, _ -> listOf(model) },
            residual = { _, _ -> 0.0 }
        ))
        assertEquals(null, LocalConsensusOptimizer.refine(
            initialModel = model,
            observationCount = 2,
            initialInliers = listOf(0, 1),
            minimumInliers = 2,
            threshold = Double.NaN,
            maximumRefitRounds = 1,
            fitCandidates = { _, _ -> listOf(model) },
            residual = { _, _ -> 0.0 }
        ))
    }

    @Test
    fun symmetricEpipolarErrorUsesPixelPointToLineDistances() {
        val fundamental = doubleArrayOf(
            0.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0
        )

        assertEquals(0.0, EpipolarGeometryMath.symmetricEpipolarError(
            fundamental, 2.0, 3.0, 5.0, 3.0
        ), 1e-12)
        assertEquals(2.0, EpipolarGeometryMath.symmetricEpipolarError(
            fundamental, 2.0, 3.0, 5.0, 5.0
        ), 1e-12)
    }

    @Test
    fun essentialToFundamentalConversionHonorsSkewedCameraMatrix() {
        val intrinsics = doubleArrayOf(
            700.0, 3.0, 640.0,
            0.0, 710.0, 360.0,
            0.0, 0.0, 1.0
        )
        val essential = doubleArrayOf(
            0.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 1.0, 0.0
        )
        val fundamental = EpipolarGeometryMath.fundamentalFromEssential(essential, intrinsics)!!
        val first = doubleArrayOf(0.3, -0.2, 1.0)
        val second = doubleArrayOf(-0.1, -0.2, 1.0)
        val firstPixel = project(first, intrinsics)
        val secondPixel = project(second, intrinsics)

        assertEquals(0.0, EpipolarGeometryMath.symmetricEpipolarError(
            fundamental, firstPixel[0], firstPixel[1], secondPixel[0], secondPixel[1]
        ), 1e-8)
        assertEquals(null, EpipolarGeometryMath.fundamentalFromEssential(
            essential, DoubleArray(9)
        ))
    }

    private fun optimizeLine(samples: List<Sample>): LocalConsensusOptimizer.Result<Line> {
        val initial = Line(1.25, -0.4)
        val initialInliers = samples.indices.filter { index ->
            perpendicularError(initial, samples[index]) <= 0.65
        }
        return LocalConsensusOptimizer.refine(
            initialModel = initial,
            observationCount = samples.size,
            initialInliers = initialInliers,
            minimumInliers = 5,
            threshold = 0.65,
            maximumRefitRounds = 3,
            fitCandidates = { _, indices ->
                val selected = indices.map(samples::get)
                val meanX = selected.map { it.x }.average()
                val meanY = selected.map { it.y }.average()
                val denominator = selected.sumOf { (it.x - meanX) * (it.x - meanX) }
                val slope = selected.sumOf { (it.x - meanX) * (it.y - meanY) } / denominator
                listOf(Line(slope, meanY - slope * meanX))
            },
            residual = { model, index -> perpendicularError(model, samples[index]) }
        )!!
    }

    private fun lineSamples(): List<Sample> = buildList {
        for (index in -5..5) {
            val noise = if (index % 2 == 0) 0.008 else -0.008
            add(Sample(index, index.toDouble(), 1.5 * index - 0.4 + noise))
        }
        add(Sample(99, 0.5, 18.0))
    }

    private fun perpendicularError(line: Line, sample: Sample): Double =
        abs(line.slope * sample.x - sample.y + line.intercept) / hypot(line.slope, 1.0)

    private fun project(point: DoubleArray, k: DoubleArray): DoubleArray = doubleArrayOf(
        k[0] * point[0] + k[1] * point[1] + k[2],
        k[4] * point[1] + k[5]
    )
}
