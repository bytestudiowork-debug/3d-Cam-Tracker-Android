package com.trackstudio.app.tracking

import kotlin.math.acos
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt

internal data class PrincipalPointRefinementEvidence(
    val activeFrameCount: Int,
    val validTrackCount: Int,
    val observationCount: Int,
    val occupiedGridCells: Int,
    val broadlyCoveredFrameCount: Int,
    val horizontalSpanFraction: Double,
    val verticalSpanFraction: Double,
    val baselineToMedianDepthRatio: Double,
    val maximumRotationFromReferenceRadians: Double,
    val rotationAxesWithSpread: Int
)

internal object PrincipalPointRefinementPolicy {
    private const val GRID_SIDE = 4
    private const val MIN_ACTIVE_FRAMES = 8
    private const val MIN_TRACKS = 120
    private const val MIN_OBSERVATIONS = 600
    private const val MIN_OCCUPIED_CELLS = 12
    private const val MIN_BROAD_FRAMES = 6
    private const val MIN_AXIS_SPAN = 0.65
    private const val MIN_BASELINE_TO_DEPTH = 0.03
    private const val MIN_ROTATION_SPREAD = 0.06
    private const val MIN_MAX_ROTATION = 0.12

    fun isObservable(evidence: PrincipalPointRefinementEvidence): Boolean =
        evidence.activeFrameCount >= MIN_ACTIVE_FRAMES &&
            evidence.validTrackCount >= MIN_TRACKS &&
            evidence.observationCount >= MIN_OBSERVATIONS &&
            evidence.occupiedGridCells >= MIN_OCCUPIED_CELLS &&
            evidence.broadlyCoveredFrameCount >= MIN_BROAD_FRAMES &&
            evidence.horizontalSpanFraction >= MIN_AXIS_SPAN &&
            evidence.verticalSpanFraction >= MIN_AXIS_SPAN &&
            evidence.baselineToMedianDepthRatio >= MIN_BASELINE_TO_DEPTH &&
            evidence.maximumRotationFromReferenceRadians >= MIN_MAX_ROTATION &&
            evidence.rotationAxesWithSpread >= 2 &&
            evidence.horizontalSpanFraction.isFinite() &&
            evidence.verticalSpanFraction.isFinite() &&
            evidence.baselineToMedianDepthRatio.isFinite() &&
            evidence.maximumRotationFromReferenceRadians.isFinite()

    fun assess(
        tracks: List<CameraFeatureTrack>,
        frameCentersById: Map<Int, DoubleArray>,
        rotationsByFrame: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        imageWidth: Int,
        imageHeight: Int
    ): PrincipalPointRefinementEvidence {
        if (imageWidth <= 0 || imageHeight <= 0) return emptyEvidence()

        val cellsByFrame = HashMap<Int, MutableSet<Int>>()
        val validFrames = HashSet<Int>()
        val depths = ArrayList<Double>()
        var validTrackCount = 0
        var observationCount = 0
        var minimumX = Double.POSITIVE_INFINITY
        var maximumX = Double.NEGATIVE_INFINITY
        var minimumY = Double.POSITIVE_INFINITY
        var maximumY = Double.NEGATIVE_INFINITY

        tracks.forEach { track ->
            val point = pointsByTrackId[track.id]
            if (point == null || point.size != 3 || point.any { !it.isFinite() }) return@forEach
            val trackFrames = HashSet<Int>()
            var validObservations = 0
            track.observations.forEach observationLoop@ { observation ->
                if (!observation.x.isFinite() || !observation.y.isFinite()) return@observationLoop
                val center = frameCentersById[observation.frameIndex] ?: return@observationLoop
                val rotation = rotationsByFrame[observation.frameIndex] ?: return@observationLoop
                if (center.size != 3 || center.any { !it.isFinite() } ||
                    rotation.size != 9 || rotation.any { !it.isFinite() }
                ) return@observationLoop
                val dx = point[0] - center[0]
                val dy = point[1] - center[1]
                val dz = point[2] - center[2]
                val cameraDepth = rotation[6] * dx + rotation[7] * dy + rotation[8] * dz
                if (!cameraDepth.isFinite() || cameraDepth <= 1e-6) return@observationLoop

                validObservations++
                trackFrames += observation.frameIndex
                validFrames += observation.frameIndex
                observationCount++
                depths += cameraDepth
                minimumX = minOf(minimumX, observation.x)
                maximumX = maxOf(maximumX, observation.x)
                minimumY = minOf(minimumY, observation.y)
                maximumY = maxOf(maximumY, observation.y)

                val column = floor(observation.x / imageWidth * GRID_SIDE)
                    .toInt().coerceIn(0, GRID_SIDE - 1)
                val row = floor(observation.y / imageHeight * GRID_SIDE)
                    .toInt().coerceIn(0, GRID_SIDE - 1)
                cellsByFrame.getOrPut(observation.frameIndex) { HashSet() }
                    .add(row * GRID_SIDE + column)
            }
            if (validObservations >= 3 && trackFrames.size >= 3) validTrackCount++
        }

        var maximumBaseline = 0.0
        val referenceFrame = validFrames.minOrNull()
        val referenceCenter = referenceFrame?.let(frameCentersById::get)
        if (referenceCenter != null && referenceCenter.size == 3 && referenceCenter.all { it.isFinite() }) {
            validFrames.forEach { frame ->
                val center = frameCentersById[frame] ?: return@forEach
                if (center.size != 3 || center.any { !it.isFinite() }) return@forEach
                val dx = center[0] - referenceCenter[0]
                val dy = center[1] - referenceCenter[1]
                val dz = center[2] - referenceCenter[2]
                maximumBaseline = maxOf(maximumBaseline, sqrt(dx * dx + dy * dy + dz * dz))
            }
        }
        val medianDepth = median(depths)
        val baselineToMedianDepth = if (medianDepth > 0.0) maximumBaseline / medianDepth else 0.0
        val horizontalSpan = if (minimumX.isFinite() && maximumX.isFinite()) {
            (maximumX - minimumX) / imageWidth
        } else 0.0
        val verticalSpan = if (minimumY.isFinite() && maximumY.isFinite()) {
            (maximumY - minimumY) / imageHeight
        } else 0.0
        val relativeRotations = if (referenceFrame == null) emptyList() else {
            val referenceRotation = rotationsByFrame[referenceFrame]
            if (referenceRotation == null || referenceRotation.size != 9) emptyList()
            else validFrames.mapNotNull { frame ->
                val rotation = rotationsByFrame[frame] ?: return@mapNotNull null
                if (rotation.size != 9 || rotation.any { !it.isFinite() }) null
                else rotationVectorRelative(rotation, referenceRotation)
            }
        }
        val maximumRotation = relativeRotations.maxOfOrNull { vector ->
            sqrt(vector.sumOf { it * it })
        } ?: 0.0
        val rotationAxesWithSpread = (0..2).count { axis ->
            val values = relativeRotations.map { it[axis] }
            values.isNotEmpty() && values.maxOrNull()!! - values.minOrNull()!! >= MIN_ROTATION_SPREAD
        }

        return PrincipalPointRefinementEvidence(
            activeFrameCount = validFrames.size,
            validTrackCount = validTrackCount,
            observationCount = observationCount,
            occupiedGridCells = cellsByFrame.values.flatten().toSet().size,
            broadlyCoveredFrameCount = cellsByFrame.values.count { it.size >= GRID_SIDE * GRID_SIDE / 2 },
            horizontalSpanFraction = horizontalSpan,
            verticalSpanFraction = verticalSpan,
            baselineToMedianDepthRatio = baselineToMedianDepth,
            maximumRotationFromReferenceRadians = maximumRotation,
            rotationAxesWithSpread = rotationAxesWithSpread
        )
    }

    private fun rotationVectorRelative(rotation: DoubleArray, reference: DoubleArray): DoubleArray {
        val relative = DoubleArray(9)
        for (row in 0..2) for (column in 0..2) {
            for (index in 0..2) {
                relative[row * 3 + column] += rotation[row * 3 + index] * reference[column * 3 + index]
            }
        }
        val cosine = ((relative[0] + relative[4] + relative[8] - 1.0) * 0.5).coerceIn(-1.0, 1.0)
        val angle = acos(cosine)
        if (angle < 1e-8) {
            return doubleArrayOf(
                0.5 * (relative[7] - relative[5]),
                0.5 * (relative[2] - relative[6]),
                0.5 * (relative[3] - relative[1])
            )
        }
        val sine = sin(angle)
        if (abs(sine) < 1e-8) return doubleArrayOf(0.0, 0.0, 0.0)
        val scale = angle / (2.0 * sine)
        return doubleArrayOf(
            scale * (relative[7] - relative[5]),
            scale * (relative[2] - relative[6]),
            scale * (relative[3] - relative[1])
        )
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) * 0.5
        } else sorted[middle]
    }

    private fun emptyEvidence() = PrincipalPointRefinementEvidence(
        activeFrameCount = 0,
        validTrackCount = 0,
        observationCount = 0,
        occupiedGridCells = 0,
        broadlyCoveredFrameCount = 0,
        horizontalSpanFraction = 0.0,
        verticalSpanFraction = 0.0,
        baselineToMedianDepthRatio = 0.0,
        maximumRotationFromReferenceRadians = 0.0,
        rotationAxesWithSpread = 0
    )
}
