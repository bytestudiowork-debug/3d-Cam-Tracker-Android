package com.trackstudio.app.tracking.glomap

import com.trackstudio.app.tracking.GlobalRotationAverager
import java.util.ArrayDeque
import java.util.TreeSet

/**
 * Android adaptation of GLOMAP's post-filter ViewGraph::KeepLargestConnectedComponents.
 * Only frames incident to a valid edge enter the upstream frame adjacency list.
 */
internal object GlomapRotationGraphReducer {
    data class Result(
        val frameIds: List<Int>,
        val edges: List<GlobalRotationAverager.Edge>
    )

    fun keepLargestConnectedComponent(
        registeredFrameIds: Collection<Int>,
        validEdges: List<GlobalRotationAverager.Edge>
    ): Result? {
        if (registeredFrameIds.isEmpty() || validEdges.isEmpty()) return null
        val registered = registeredFrameIds.toHashSet()
        val adjacency = HashMap<Int, MutableSet<Int>>()
        for (edge in validEdges) {
            val first = edge.firstFrame
            val second = edge.secondFrame
            if (first == second || first !in registered || second !in registered) continue
            adjacency.getOrPut(first) { HashSet() }.add(second)
            adjacency.getOrPut(second) { HashSet() }.add(first)
        }
        if (adjacency.isEmpty()) return null

        val unseen = TreeSet<Int>().apply { addAll(adjacency.keys) }
        var largestComponent = emptyList<Int>()
        while (unseen.isNotEmpty()) {
            val start = unseen.first()
            unseen.remove(start)
            val queue = ArrayDeque<Int>()
            val component = ArrayList<Int>()
            queue.addLast(start)
            while (queue.isNotEmpty()) {
                val current = queue.removeFirst()
                component.add(current)
                for (neighbor in adjacency[current].orEmpty().sorted()) {
                    if (unseen.remove(neighbor)) queue.addLast(neighbor)
                }
            }
            component.sort()
            // Pinned GLOMAP leaves equal-size ties to unordered-map iteration;
            // use the lowest frame id for stable Android results.
            if (component.size > largestComponent.size ||
                (component.size == largestComponent.size &&
                    component.first() < largestComponent.first())
            ) {
                largestComponent = component
            }
        }
        if (largestComponent.size < 2) return null

        val retained = largestComponent.toHashSet()
        val retainedEdges = validEdges.filter {
            it.firstFrame != it.secondFrame && it.firstFrame in retained &&
                it.secondFrame in retained
        }
        if (retainedEdges.isEmpty()) return null
        return Result(largestComponent, retainedEdges)
    }
}
