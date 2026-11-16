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
 * Android adaptation of NormalizeReconstruction in
 * glomap/processors/reconstruction_normalizer.cc from GLOMAP commit
 * 0edb1b8435e0f9a594318908b81a31f078a51bf7.
 */
package com.trackstudio.app.tracking.glomap

import kotlin.math.sqrt

internal object GlomapReconstructionNormalizer {
    const val DEFAULT_EXTENT = 10.0
    const val DEFAULT_P0 = 0.1
    const val DEFAULT_P1 = 0.9

    data class Result(
        val frameCentersById: Map<Int, DoubleArray>,
        val pointsByTrackId: Map<Int, DoubleArray>,
        val scale: Double,
        val translation: DoubleArray,
        val applied: Boolean
    ) {
        fun transformPoint(point: DoubleArray): DoubleArray {
            require(point.size == 3)
            return doubleArrayOf(
                scale * point[0] + translation[0],
                scale * point[1] + translation[1],
                scale * point[2] + translation[2]
            )
        }
    }

    fun normalize(
        frameCentersById: Map<Int, DoubleArray>,
        pointsByTrackId: Map<Int, DoubleArray>,
        fixedScale: Boolean = false,
        extent: Double = DEFAULT_EXTENT,
        p0: Double = DEFAULT_P0,
        p1: Double = DEFAULT_P1
    ): Result {
        require(extent.isFinite() && extent > 0.0)
        require(p0.isFinite() && p1.isFinite() && p0 in 0.0..1.0 && p1 in 0.0..1.0 && p0 <= p1)
        if (frameCentersById.isEmpty()) {
            return Result(
                frameCentersById = frameCentersById,
                pointsByTrackId = pointsByTrackId,
                scale = 1.0,
                translation = doubleArrayOf(0.0, 0.0, 0.0),
                applied = false
            )
        }

        val centers = frameCentersById.values.toList()
        centers.forEach { center ->
            require(center.size == 3 && center.all(Double::isFinite))
        }
        pointsByTrackId.values.forEach { point ->
            require(point.size == 3 && point.all(Double::isFinite))
        }

        val coordinatesX = FloatArray(centers.size) { centers[it][0].toFloat() }
        val coordinatesY = FloatArray(centers.size) { centers[it][1].toFloat() }
        val coordinatesZ = FloatArray(centers.size) { centers[it][2].toFloat() }
        require(coordinatesX.all(Float::isFinite) &&
            coordinatesY.all(Float::isFinite) &&
            coordinatesZ.all(Float::isFinite)
        )
        coordinatesX.sort()
        coordinatesY.sort()
        coordinatesZ.sort()

        val p0Index = if (centers.size > 3) (p0 * (centers.size - 1)).toInt() else 0
        val p1Index = if (centers.size > 3) {
            (p1 * (centers.size - 1)).toInt()
        } else {
            centers.size - 1
        }

        val minimum = doubleArrayOf(
            coordinatesX[p0Index].toDouble(),
            coordinatesY[p0Index].toDouble(),
            coordinatesZ[p0Index].toDouble()
        )
        val maximum = doubleArrayOf(
            coordinatesX[p1Index].toDouble(),
            coordinatesY[p1Index].toDouble(),
            coordinatesZ[p1Index].toDouble()
        )
        val mean = doubleArrayOf(0.0, 0.0, 0.0)
        for (index in p0Index..p1Index) {
            mean[0] += coordinatesX[index].toDouble()
            mean[1] += coordinatesY[index].toDouble()
            mean[2] += coordinatesZ[index].toDouble()
        }
        val selectedCount = (p1Index - p0Index + 1).toDouble()
        mean.indices.forEach { axis -> mean[axis] /= selectedCount }

        val dx = maximum[0] - minimum[0]
        val dy = maximum[1] - minimum[1]
        val dz = maximum[2] - minimum[2]
        val oldExtent = sqrt(dx * dx + dy * dy + dz * dz)
        val scale = if (!fixedScale && oldExtent >= Math.ulp(1.0)) extent / oldExtent else 1.0
        val translation = doubleArrayOf(
            -scale * mean[0],
            -scale * mean[1],
            -scale * mean[2]
        )

        fun transform(point: DoubleArray): DoubleArray = doubleArrayOf(
            scale * point[0] + translation[0],
            scale * point[1] + translation[1],
            scale * point[2] + translation[2]
        )

        return Result(
            frameCentersById = frameCentersById.mapValues { (_, center) -> transform(center) },
            pointsByTrackId = pointsByTrackId.mapValues { (_, point) -> transform(point) },
            scale = scale,
            translation = translation,
            applied = true
        )
    }
}
