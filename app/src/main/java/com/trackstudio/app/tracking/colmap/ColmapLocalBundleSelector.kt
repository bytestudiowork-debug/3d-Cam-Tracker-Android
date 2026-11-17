package com.trackstudio.app.tracking.colmap

import com.trackstudio.app.tracking.CameraFeatureTrack
import kotlin.math.PI
import kotlin.math.acos
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Android adaptation of COLMAP's adaptive local-bundle image selection.
 * It returns a reference frame plus its highest-connectivity registered neighbors.
 */
internal object ColmapLocalBundleSelector {
    const val DEFAULT_MAX_NUM_IMAGES = 6
    const val DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES = 6.0

    data class Selection(
        val referenceFrameId: Int,
        val neighborFrameIds: List<Int>,
        val sharedObservationsByFrame: Map<Int, Int>,
        val triangulationAngle75RadiansByFrame: Map<Int, Double>,
        val queryPointCount: Int
    ) {
        val frameIds: List<Int>
            get() = listOf(referenceFrameId) + neighborFrameIds
    }

    private data class Neighbor(
        val frameId: Int,
        val sharedTrackIds: List<Int>
    ) {
        val sharedCount: Int
            get() = sharedTrackIds.size
    }

    private data class SelectionThreshold(
        val minimumAngleRadians: Double,
        val minimumSharedObservations: Double
    )

    /**
     * Selects neighboring frames using the pinned COLMAP overlap/triangulation policy.
     * `cameraCentersByFrame` must contain only registered, valid world-space centers.
     */
    fun select(
        referenceFrameId: Int,
        cameraCentersByFrame: Map<Int, DoubleArray>,
        tracks: List<CameraFeatureTrack>,
        pointsByTrackId: Map<Int, DoubleArray>,
        maxNumImages: Int = DEFAULT_MAX_NUM_IMAGES,
        minTriangulationAngleDegrees: Double = DEFAULT_MIN_TRIANGULATION_ANGLE_DEGREES
    ): Selection {
        require(maxNumImages >= 2)
        require(minTriangulationAngleDegrees.isFinite() && minTriangulationAngleDegrees >= 0.0)

        val validCenters = cameraCentersByFrame.filterValues(::isVector3)
        val referenceCenter = validCenters[referenceFrameId]
            ?: return Selection(referenceFrameId, emptyList(), emptyMap(), emptyMap(), 0)
        val sharedTrackIdsByFrame = linkedMapOf<Int, MutableList<Int>>()
        val seenTrackIds = HashSet<Int>()
        var queryPointCount = 0

        for (track in tracks) {
            if (!seenTrackIds.add(track.id)) continue
            if (pointsByTrackId[track.id]?.let(::isVector3) != true) continue
            val observedFrameIds = track.observations.asSequence()
                .map { it.frameIndex }
                .filter { it in validCenters }
                .distinct()
                .toList()
            if (referenceFrameId !in observedFrameIds) continue

            queryPointCount++
            for (frameId in observedFrameIds) {
                if (frameId != referenceFrameId) {
                    sharedTrackIdsByFrame.getOrPut(frameId) { mutableListOf() }.add(track.id)
                }
            }
        }

        // COLMAP's std::sort leaves equal-overlap ordering unspecified. Use frame ID as a
        // deterministic Android tie-break while preserving descending shared-track priority.
        val overlapping = sharedTrackIdsByFrame.map { (frameId, trackIds) ->
            Neighbor(frameId, trackIds.toList())
        }.sortedWith(compareByDescending<Neighbor> { it.sharedCount }.thenBy { it.frameId })
        val neighborBudget = maxNumImages - 1
        val effectiveNeighborCount = min(neighborBudget, overlapping.size)
        val selected = mutableListOf<Int>()
        val angleByFrame = linkedMapOf<Int, Double>()

        // Pinned COLMAP fast path: if every overlapping image fits, skip angle gating.
        if (overlapping.size == effectiveNeighborCount) {
            selected += overlapping.map { it.frameId }
            return result(referenceFrameId, selected, overlapping, angleByFrame, queryPointCount)
        }

        val minimumAngle = Math.toRadians(minTriangulationAngleDegrees)
        val thresholds = listOf(
            SelectionThreshold(minimumAngle / 1.0, 0.6 * queryPointCount),
            SelectionThreshold(minimumAngle / 1.5, 0.6 * queryPointCount),
            SelectionThreshold(minimumAngle / 2.0, 0.5 * queryPointCount),
            SelectionThreshold(minimumAngle / 2.5, 0.4 * queryPointCount),
            SelectionThreshold(minimumAngle / 3.0, 0.3 * queryPointCount),
            SelectionThreshold(minimumAngle / 4.0, 0.2 * queryPointCount),
            SelectionThreshold(minimumAngle / 5.0, 0.1 * queryPointCount),
            SelectionThreshold(minimumAngle / 6.0, 0.1 * queryPointCount)
        )

        for (threshold in thresholds) {
            for (neighbor in overlapping) {
                // Overlap is descending, so the remainder cannot satisfy this gate.
                if (neighbor.sharedCount < threshold.minimumSharedObservations) break
                if (neighbor.frameId in selected) continue

                val angle75 = angleByFrame.getOrPut(neighbor.frameId) {
                    triangulationAnglePercentile75(
                        referenceCenter = referenceCenter,
                        neighborCenter = validCenters.getValue(neighbor.frameId),
                        trackIds = neighbor.sharedTrackIds,
                        pointsByTrackId = pointsByTrackId
                    )
                }
                if (angle75 >= threshold.minimumAngleRadians) {
                    selected += neighbor.frameId
                    if (selected.size >= effectiveNeighborCount) break
                }
            }
            if (selected.size >= effectiveNeighborCount) break
        }

        if (selected.size < effectiveNeighborCount) {
            for (neighbor in overlapping) {
                if (neighbor.frameId !in selected) selected += neighbor.frameId
                if (selected.size >= effectiveNeighborCount) break
            }
        }

        return result(referenceFrameId, selected, overlapping, angleByFrame, queryPointCount)
    }

    private fun result(
        referenceFrameId: Int,
        selectedFrameIds: List<Int>,
        overlapping: List<Neighbor>,
        angleByFrame: Map<Int, Double>,
        queryPointCount: Int
    ) = Selection(
        referenceFrameId = referenceFrameId,
        neighborFrameIds = selectedFrameIds.toList(),
        sharedObservationsByFrame = overlapping.associateTo(linkedMapOf()) {
            it.frameId to it.sharedCount
        },
        triangulationAngle75RadiansByFrame = angleByFrame.toMap(),
        queryPointCount = queryPointCount
    )

    private fun triangulationAnglePercentile75(
        referenceCenter: DoubleArray,
        neighborCenter: DoubleArray,
        trackIds: List<Int>,
        pointsByTrackId: Map<Int, DoubleArray>
    ): Double {
        val angles = trackIds.mapNotNull { trackId ->
            val point = pointsByTrackId[trackId] ?: return@mapNotNull null
            triangulationAngle(referenceCenter, neighborCenter, point)
                .takeIf(Double::isFinite)
        }
        if (angles.isEmpty()) return Double.NaN
        val ordered = angles.sorted()
        val index = 0.75 * (ordered.size - 1)
        val leftIndex = floor(index).toInt()
        val rightIndex = ceil(index).toInt()
        if (leftIndex == rightIndex) return ordered[rightIndex]
        val left = ordered[leftIndex]
        val right = ordered[rightIndex]
        return (rightIndex - index) * left + (index - leftIndex) * right
    }

    /** Law-of-cosines angle folded to COLMAP's stable smaller intersection angle. */
    private fun triangulationAngle(
        center1: DoubleArray,
        center2: DoubleArray,
        point: DoubleArray
    ): Double {
        val dx = center1[0] - center2[0]
        val dy = center1[1] - center2[1]
        val dz = center1[2] - center2[2]
        val baselineSquared = dx * dx + dy * dy + dz * dz
        val x1 = point[0] - center1[0]
        val y1 = point[1] - center1[1]
        val z1 = point[2] - center1[2]
        val ray1Squared = x1 * x1 + y1 * y1 + z1 * z1
        val x2 = point[0] - center2[0]
        val y2 = point[1] - center2[1]
        val z2 = point[2] - center2[2]
        val ray2Squared = x2 * x2 + y2 * y2 + z2 * z2
        val denominator = 2.0 * sqrt(ray1Squared * ray2Squared)
        if (denominator == 0.0) return 0.0
        if (!denominator.isFinite()) return Double.NaN
        val cosine = (ray1Squared + ray2Squared - baselineSquared) / denominator
        if (!cosine.isFinite()) return Double.NaN
        val angle = acos(cosine.coerceIn(-1.0, 1.0))
        return min(angle, PI - angle)
    }

    private fun isVector3(values: DoubleArray): Boolean =
        values.size == 3 && values.all(Double::isFinite)
}
