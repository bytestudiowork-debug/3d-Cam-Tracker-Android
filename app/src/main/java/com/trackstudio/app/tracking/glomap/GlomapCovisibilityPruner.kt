/*
 * Copyright (c) 2024, ETH Zurich. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the name of ETH Zurich nor the names of its contributors may
 *       be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Android adaptation of GLOMAP's PruneWeaklyConnectedImages and
 * EstablishStrongClusters at commit 0edb1b8435e0f9a594318908b81a31f078a51bf7.
 * See app/src/main/assets/licenses/glomap-reconstruction-pruning-license.txt.
 */
package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.CameraFeatureTrack
import java.util.ArrayDeque

/** Pure in-memory counterpart of GLOMAP's final covisibility clustering stage. */
internal object GlomapCovisibilityPruner {
    const val DEFAULT_MIN_NUM_IMAGES = 2
    const val DEFAULT_MIN_NUM_OBSERVATIONS = 0
    const val MIN_COVISIBLE_TRACKS_PER_PAIR = 5
    const val DEFAULT_STRONG_CLUSTER_FLOOR = 20.0
    const val WEAK_BRIDGE_FACTOR = 0.75
    const val MIN_WEAK_BRIDGE_PAIRS = 2
    const val MAX_BRIDGE_ITERATIONS = 10

    data class Result(
        /** Contains every requested frame; -1 means it belongs to no retained cluster. */
        val clusterIdByFrame: Map<Int, Int>,
        val clusterCount: Int,
        val strongThreshold: Double?,
        val eligiblePairWeights: Map<Pair<Int, Int>, Long>,
        val bridgeIterations: Int
    )

    /**
     * Builds covisibility from tracks with more than two observations, then
     * labels strong clusters and clusters connected by at least two weak pairs.
     *
     * The upstream stage assigns frame cluster IDs and invalidates inter-cluster
     * visibility edges; it does not unregister or delete frames or tracks.
     */
    fun prune(
        tracks: List<CameraFeatureTrack>,
        allFrameIds: Set<Int>,
        activeFrameIds: Set<Int> = allFrameIds,
        minNumImages: Int = DEFAULT_MIN_NUM_IMAGES,
        minNumObservations: Int = DEFAULT_MIN_NUM_OBSERVATIONS,
        minCovisibleTracksPerPair: Int = MIN_COVISIBLE_TRACKS_PER_PAIR,
        strongClusterFloor: Double = DEFAULT_STRONG_CLUSTER_FLOOR,
        weakBridgeFactor: Double = WEAK_BRIDGE_FACTOR,
        minWeakBridgePairs: Int = MIN_WEAK_BRIDGE_PAIRS,
        maxBridgeIterations: Int = MAX_BRIDGE_ITERATIONS
    ): Result {
        require(allFrameIds.containsAll(activeFrameIds))
        require(minNumImages >= 0 && minNumObservations >= 0)
        require(minCovisibleTracksPerPair > 0)
        require(strongClusterFloor.isFinite() && strongClusterFloor >= 0.0)
        require(weakBridgeFactor.isFinite() && weakBridgeFactor in 0.0..1.0)
        require(minWeakBridgePairs > 0 && maxBridgeIterations >= 0)

        val frameObservationCounts = HashMap<Int, Long>()
        val pairWeights = HashMap<Pair<Int, Int>, Long>()
        for (track in tracks) {
            if (track.observations.size <= 2) continue
            val observations = track.observations.filter { it.frameIndex in activeFrameIds }
            for (observation in observations) {
                frameObservationCounts[observation.frameIndex] =
                    (frameObservationCounts[observation.frameIndex] ?: 0L) + 1L
            }
            for (firstIndex in observations.indices) {
                val firstFrame = observations[firstIndex].frameIndex
                for (secondIndex in firstIndex + 1 until observations.size) {
                    val secondFrame = observations[secondIndex].frameIndex
                    if (firstFrame == secondFrame) continue
                    val pair = orderedPair(firstFrame, secondFrame)
                    pairWeights[pair] = (pairWeights[pair] ?: 0L) + 1L
                }
            }
        }

        val visiblePairWeights = pairWeights.filter { (pair, count) ->
            count >= minCovisibleTracksPerPair.toLong() &&
                (frameObservationCounts[pair.first] ?: 0L) >= minNumObservations.toLong() &&
                (frameObservationCounts[pair.second] ?: 0L) >= minNumObservations.toLong()
        }.toSortedMap(pairComparator)
        if (visiblePairWeights.isEmpty()) {
            return Result(
                clusterIdByFrame = allFrameIds.sorted().associateWith { -1 },
                clusterCount = 0,
                strongThreshold = null,
                eligiblePairWeights = emptyMap(),
                bridgeIterations = 0
            )
        }

        val sortedWeights = visiblePairWeights.values.sorted()
        val median = sortedWeights[sortedWeights.size / 2].toDouble()
        val sortedAbsoluteDeviations = sortedWeights
            .map { kotlin.math.abs(it.toDouble() - median) }
            .sorted()
        val medianAbsoluteDeviation = sortedAbsoluteDeviations[sortedAbsoluteDeviations.size / 2]
        val strongThreshold = maxOf(median - medianAbsoluteDeviation, strongClusterFloor)

        val disjointSet = DisjointSet()
        for ((pair, weight) in visiblePairWeights) {
            if (weight.toDouble() > strongThreshold) {
                disjointSet.union(pair.first, pair.second)
            }
        }

        var bridgeIterations = 0
        while (bridgeIterations < maxBridgeIterations) {
            bridgeIterations++
            val bridgeCounts = HashMap<Pair<Int, Int>, Int>()
            for ((pair, weight) in visiblePairWeights) {
                if (weight.toDouble() < weakBridgeFactor * strongThreshold) continue
                val firstRoot = disjointSet.find(pair.first)
                val secondRoot = disjointSet.find(pair.second)
                if (firstRoot == secondRoot) continue
                val roots = orderedPair(firstRoot, secondRoot)
                bridgeCounts[roots] = (bridgeCounts[roots] ?: 0) + 1
            }

            var merged = false
            for ((roots, count) in bridgeCounts.entries.sortedWith(
                compareBy<Map.Entry<Pair<Int, Int>, Int>> { it.key.first }
                    .thenBy { it.key.second }
            )) {
                if (count >= minWeakBridgePairs &&
                    disjointSet.union(roots.first, roots.second)
                ) {
                    merged = true
                }
            }
            if (!merged) break
        }

        val adjacency = HashMap<Int, MutableSet<Int>>()
        for ((pair, _) in visiblePairWeights) {
            if (disjointSet.find(pair.first) != disjointSet.find(pair.second)) continue
            adjacency.getOrPut(pair.first) { sortedSetOf() }.add(pair.second)
            adjacency.getOrPut(pair.second) { sortedSetOf() }.add(pair.first)
        }

        val visited = HashSet<Int>()
        val components = ArrayList<Set<Int>>()
        for (root in adjacency.keys.sorted()) {
            if (!visited.add(root)) continue
            val component = sortedSetOf<Int>()
            val queue = ArrayDeque<Int>()
            queue.add(root)
            while (queue.isNotEmpty()) {
                val frame = queue.removeFirst()
                component.add(frame)
                for (neighbor in adjacency[frame].orEmpty()) {
                    if (visited.add(neighbor)) queue.addLast(neighbor)
                }
            }
            if (component.size >= minNumImages) components += component
        }

        val orderedComponents = components.sortedWith(
            compareByDescending<Set<Int>> { it.size }
                .thenBy { it.minOrNull() ?: Int.MAX_VALUE }
        )
        val clusterIds = LinkedHashMap<Int, Int>(allFrameIds.size)
        allFrameIds.sorted().forEach { clusterIds[it] = -1 }
        orderedComponents.forEachIndexed { clusterId, component ->
            component.forEach { clusterIds[it] = clusterId }
        }

        return Result(
            clusterIdByFrame = clusterIds,
            clusterCount = orderedComponents.size,
            strongThreshold = strongThreshold,
            eligiblePairWeights = visiblePairWeights,
            bridgeIterations = bridgeIterations
        )
    }

    private fun orderedPair(first: Int, second: Int): Pair<Int, Int> =
        if (first <= second) first to second else second to first

    private val pairComparator = compareBy<Pair<Int, Int>> { it.first }.thenBy { it.second }

    private class DisjointSet {
        private val parent = HashMap<Int, Int>()

        fun find(value: Int): Int {
            val directParent = parent[value] ?: run {
                parent[value] = value
                return value
            }
            if (directParent == value) return value
            val root = find(directParent)
            parent[value] = root
            return root
        }

        fun union(first: Int, second: Int): Boolean {
            val firstRoot = find(first)
            val secondRoot = find(second)
            if (firstRoot == secondRoot) return false
            if (firstRoot < secondRoot) parent[secondRoot] = firstRoot
            else parent[firstRoot] = secondRoot
            return true
        }
    }
}
