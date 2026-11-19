package com.trackstudio.app.tracking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.KeyPoint
import org.opencv.core.Point3
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class SyntheticMultiFrameSequenceTest {
    @Test
    fun syntheticSequenceExercisesIncrementalAndGlobalGeometryStages() {
        val scene = createScene()
        val featureDescriptors = scene.points.indices.map { pointIndex ->
            RootSiftDescriptorMath.normalizeDescriptor(
                FloatArray(scene.points.size) { descriptorIndex ->
                    if (descriptorIndex == pointIndex) 1f else 0f
                }
            )
        }
        assertEquals(512.0, sqrt(featureDescriptors.first().sumOf { it.toDouble() * it }), 1e-4)

        val pairMatches = mutualMatches(featureDescriptors, featureDescriptors)
        assertEquals(scene.points.size, pairMatches.size)
        assertEquals(scene.points.indices.map { it to it }, pairMatches)

        verifyEssentialGeometry(scene, pairMatches)

        val incrementalTracks = buildIncrementalTracks(scene, pairMatches)
        assertEquals(scene.points.size, incrementalTracks.size)
        assertTrue(incrementalTracks.all { it.observations.size == scene.poses.size })

        val globalTracks = buildGlobalTracks(scene, pairMatches)
        assertEquals(scene.points.size, globalTracks.size)
        assertTrue(globalTracks.all { it.observations.size == scene.poses.size })

        val poseByFrame = scene.poses.associateBy { it.frameIndex }
        val incrementalEstimates = incrementalTracks.map { track ->
            triangulate(track, poseByFrame, scene.intrinsics)
        }
        val globalEstimates = globalTracks.map { track ->
            triangulate(track, poseByFrame, scene.intrinsics)
        }
        assertTrue(incrementalEstimates.all { it != null })
        assertTrue(globalEstimates.all { it != null })
        val incrementalMeanError = meanPointError(
            incrementalTracks.zip(incrementalEstimates).map { (track, estimate) ->
                distance3(estimate!!.pointWorld, scene.points[track.id])
            }
        )
        val globalMeanError = meanPointError(
            globalTracks.map { track ->
                val estimate = triangulate(track, poseByFrame, scene.intrinsics)!!
                distance3(estimate.pointWorld, scene.points[trackForPointIndex(track, scene)])
            }
        )
        assertTrue("incremental triangulation error=$incrementalMeanError", incrementalMeanError < 2e-3)
        assertTrue("global triangulation error=$globalMeanError", globalMeanError < 2e-3)

        val rotationEdges = allPairEdges(scene, pairMatches.size)
        val averagedRotations = GlobalRotationAverager.solve(
            frameIndices = scene.poses.indices.toList(),
            edges = rotationEdges
        )
        assertNotNull(averagedRotations)
        scene.rotations.forEachIndexed { frame, rotation ->
            assertTrue(
                "rotation error at frame $frame",
                rotationError(averagedRotations!!.rotationsByFrame.getValue(frame), rotation) < 0.01
            )
        }

        val incrementalTruth: (CameraFeatureTrack) -> DoubleArray = { track -> scene.points[track.id] }
        val incrementalInitial = perturbedLandmarks(incrementalTracks, incrementalTruth)
        val incrementalRefined = MobileBundleAdjuster.optimizeLandmarks(
            featureTracks = incrementalTracks,
            landmarks = incrementalInitial,
            poses = scene.poses,
            intrinsics = scene.intrinsics
        )
        val incrementalBefore = meanLandmarkError(
            incrementalInitial, incrementalTracks, incrementalTruth
        )
        val incrementalAfter = meanLandmarkError(
            incrementalRefined, incrementalTracks, incrementalTruth
        )
        assertTrue(
            "incremental refinement error $incrementalAfter was not below $incrementalBefore",
            incrementalAfter < incrementalBefore * 0.25
        )

        val globalTruth: (CameraFeatureTrack) -> DoubleArray = { track ->
            scene.points[trackForPointIndex(track, scene)]
        }
        val globalInitial = perturbedLandmarks(globalTracks, globalTruth)
        val globalRefined = MobileBundleAdjuster.optimizeLandmarks(
            featureTracks = globalTracks,
            landmarks = globalInitial,
            poses = scene.poses,
            intrinsics = scene.intrinsics
        )
        val globalBefore = meanLandmarkError(globalInitial, globalTracks, globalTruth)
        val globalAfter = meanLandmarkError(globalRefined, globalTracks, globalTruth)
        assertTrue(
            "global point refinement error $globalAfter was not below $globalBefore",
            globalAfter < globalBefore * 0.25
        )
    }

    private data class Scene(
        val intrinsics: PinholeIntrinsics,
        val centers: List<DoubleArray>,
        val rotations: List<DoubleArray>,
        val poses: List<CameraPose>,
        val points: List<DoubleArray>,
        val keypointsByFrame: List<Array<KeyPoint>>,
        val observationsByFrame: Map<Int, List<CameraFeatureObservation>>
    )

    private fun createScene(): Scene {
        val intrinsics = PinholeIntrinsics(700.0, 710.0, 1.5, 320.0, 240.0)
        val centers = listOf(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(0.28, 0.02, 0.01),
            doubleArrayOf(0.58, 0.06, 0.04),
            doubleArrayOf(0.92, 0.12, 0.07),
            doubleArrayOf(1.28, 0.20, 0.11)
        )
        val rotations = listOf(
            rotation(doubleArrayOf(0.0, 0.0, 0.0)),
            rotation(doubleArrayOf(0.002, 0.012, -0.005)),
            rotation(doubleArrayOf(-0.006, 0.020, 0.007)),
            rotation(doubleArrayOf(0.007, 0.030, 0.010)),
            rotation(doubleArrayOf(0.012, 0.040, 0.013))
        )
        val poses = centers.indices.map { frame ->
            val rotation = rotations[frame]
            val rotatedCenter = multiplyVector(rotation, centers[frame])
            CameraPose(
                frameIndex = frame,
                rotation = rotation,
                translation = DoubleArray(3) { -rotatedCenter[it] },
                status = if (frame == 0) CameraPoseStatus.REFERENCE else CameraPoseStatus.PNP_REGISTERED
            )
        }
        val points = (0 until 120).map { index ->
            val column = index % 12
            val row = index / 12
            doubleArrayOf(
                (column - 5.5) * 0.24,
                (row - 4.5) * 0.22,
                5.8 + 0.09 * (index % 13)
            )
        }
        val observationsByFrame = centers.indices.associateWith { frame ->
            points.map { point ->
                val relative = DoubleArray(3) { point[it] - centers[frame][it] }
                val cameraPoint = multiplyVector(rotations[frame], relative)
                val pixel = CameraProjectionMath.project(cameraPoint, intrinsics)
                CameraFeatureObservation(frame, pixel[0], pixel[1])
            }
        }
        val keypointsByFrame = centers.indices.map { frame ->
            observationsByFrame.getValue(frame).map { observation ->
                KeyPoint(observation.x.toFloat(), observation.y.toFloat(), 1f)
            }.toTypedArray()
        }
        return Scene(intrinsics, centers, rotations, poses, points, keypointsByFrame, observationsByFrame)
    }

    private fun mutualMatches(
        firstDescriptors: List<FloatArray>,
        secondDescriptors: List<FloatArray>
    ): List<Pair<Int, Int>> {
        fun rankedRows(query: List<FloatArray>, train: List<FloatArray>) =
            query.mapIndexed { queryIndex, descriptor ->
                train.mapIndexed { trainIndex, candidate ->
                    val normalizedL2Distance = sqrt(
                        descriptor.indices.sumOf { index ->
                            val difference = descriptor[index] - candidate[index]
                            difference.toDouble() * difference
                        }
                    ) / RootSiftDescriptorMath.DESCRIPTOR_NORM
                    RankedDescriptorMatch(queryIndex, trainIndex, normalizedL2Distance)
                }.sortedWith(
                    compareBy<RankedDescriptorMatch> { it.normalizedL2Distance }
                        .thenBy { it.trainIndex }
                )
                    .take(2)
            }
        return SiftMatchPolicy.selectMutualMatches(
            rankedRows(firstDescriptors, secondDescriptors),
            rankedRows(secondDescriptors, firstDescriptors)
        )
    }

    private fun buildIncrementalTracks(
        scene: Scene,
        matches: List<Pair<Int, Int>>
    ): List<CameraFeatureTrack> {
        val builder = FeatureTrackBuilder()
        builder.seedPair(0, 1, scene.keypointsByFrame[0], scene.keypointsByFrame[1], matches)
        for (frame in 2 until scene.poses.size) {
            builder.addInlierMatches(
                frame,
                scene.keypointsByFrame[frame - 1],
                scene.keypointsByFrame[frame],
                matches
            )
        }
        return builder.build(minimumObservations = scene.poses.size)
    }

    private fun buildGlobalTracks(
        scene: Scene,
        matches: List<Pair<Int, Int>>
    ): List<CameraFeatureTrack> {
        val builder = GlobalTrackGraphBuilder()
        for (frame in 0 until scene.poses.lastIndex) {
            builder.addVerifiedMatches(frame, frame + 1, matches)
        }
        builder.addVerifiedMatches(0, 2, matches)
        builder.addVerifiedMatches(0, 4, matches)
        builder.addVerifiedMatches(1, 3, matches)
        return builder.buildFromLocations(scene.observationsByFrame, scene.poses.indices.toSet())
    }

    private fun verifyEssentialGeometry(scene: Scene, matches: List<Pair<Int, Int>>) {
        val firstPose = scene.poses[0]
        val secondPose = scene.poses[1]
        val relativeRotation = multiply(secondPose.rotation, transpose(firstPose.rotation))
        val rotatedFirstTranslation = multiplyVector(relativeRotation, firstPose.translation)
        val relativeTranslation = DoubleArray(3) { secondPose.translation[it] - rotatedFirstTranslation[it] }
        val essential = multiply(skew(relativeTranslation), relativeRotation)
        val cameraMatrix = doubleArrayOf(
            scene.intrinsics.fx, scene.intrinsics.skew, scene.intrinsics.cx,
            0.0, scene.intrinsics.fy, scene.intrinsics.cy,
            0.0, 0.0, 1.0
        )
        val fundamental = EpipolarGeometryMath.fundamentalFromEssential(essential, cameraMatrix)
        assertNotNull(fundamental)
        for ((firstIndex, secondIndex) in matches) {
            val first = scene.observationsByFrame.getValue(0)[firstIndex]
            val second = scene.observationsByFrame.getValue(1)[secondIndex]
            val error = EpipolarGeometryMath.symmetricEpipolarError(
                fundamental!!, first.x, first.y, second.x, second.y
            )
            assertTrue("epipolar error=$error", error < 1e-3)
        }
    }

    private fun allPairEdges(scene: Scene, inlierCount: Int): List<GlobalRotationAverager.Edge> =
        buildList {
            for (first in scene.poses.indices) {
                for (second in first + 1 until scene.poses.size) {
                    add(
                        GlobalRotationAverager.Edge(
                            firstFrame = first,
                            secondFrame = second,
                            relativeRotation = multiply(
                                scene.rotations[second], transpose(scene.rotations[first])
                            ),
                            inlierCount = inlierCount
                        )
                    )
                }
            }
        }

    private fun triangulate(
        track: CameraFeatureTrack,
        posesByFrame: Map<Int, CameraPose>,
        intrinsics: PinholeIntrinsics
    ): MultiViewTriangulationEstimate? = MultiViewTriangulationMath.estimate(
        observations = track.observations.mapNotNull { observation ->
            posesByFrame[observation.frameIndex]?.let { pose ->
                TriangulationView(observation.x, observation.y, pose)
            }
        },
        intrinsics = intrinsics,
        minimumParallaxDegrees = 0.5,
        maximumAngularErrorDegrees = 0.1,
        minimumInlierRatio = 1.0,
        maximumReprojectionErrorPx = 0.05
    )

    private fun trackForPointIndex(track: CameraFeatureTrack, scene: Scene): Int {
        val first = track.observations.first { it.frameIndex == 0 }
        return scene.observationsByFrame.getValue(0).indices.minBy { index ->
            val candidate = scene.observationsByFrame.getValue(0)[index]
            val dx = candidate.x - first.x
            val dy = candidate.y - first.y
            dx * dx + dy * dy
        }
    }

    private fun meanPointError(errors: List<Double>): Double = errors.average()

    private fun perturbedLandmarks(
        tracks: List<CameraFeatureTrack>,
        truthPoint: (CameraFeatureTrack) -> DoubleArray
    ): List<CameraLandmark> = tracks.map { track ->
        val truth = truthPoint(track)
        val perturbed = doubleArrayOf(
            truth[0] + 0.03 * sin(track.id * 0.31),
            truth[1] + 0.025 * cos(track.id * 0.23),
            truth[2] + 0.05 * sin(track.id * 0.17)
        )
        CameraLandmark(
            trackId = track.id,
            position = Point3(perturbed[0], perturbed[1], perturbed[2]),
            observationCount = track.observations.size,
            meanReprojectionErrorPx = 5.0
        )
    }

    private fun meanLandmarkError(
        landmarks: List<CameraLandmark>,
        tracks: List<CameraFeatureTrack>,
        truthPoint: (CameraFeatureTrack) -> DoubleArray
    ): Double = meanPointError(landmarks.indices.map { index ->
        val landmark = landmarks[index]
        distance3(
            doubleArrayOf(landmark.position.x, landmark.position.y, landmark.position.z),
            truthPoint(tracks[index])
        )
    })

    private fun distance3(first: DoubleArray, second: DoubleArray): Double = sqrt(
        (first[0] - second[0]) * (first[0] - second[0]) +
            (first[1] - second[1]) * (first[1] - second[1]) +
            (first[2] - second[2]) * (first[2] - second[2])
    )

    private fun rotationError(first: DoubleArray, second: DoubleArray): Double {
        val difference = multiply(first, transpose(second))
        val cosine = ((difference[0] + difference[4] + difference[8] - 1.0) * 0.5)
            .coerceIn(-1.0, 1.0)
        return kotlin.math.acos(cosine)
    }

    private fun rotation(angleAxis: DoubleArray): DoubleArray {
        val angle = sqrt(angleAxis.sumOf { it * it })
        if (angle < 1e-12) return doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        val x = angleAxis[0] / angle
        val y = angleAxis[1] / angle
        val z = angleAxis[2] / angle
        val c = cos(angle)
        val s = sin(angle)
        val t = 1.0 - c
        return doubleArrayOf(
            t * x * x + c, t * x * y - s * z, t * x * z + s * y,
            t * x * y + s * z, t * y * y + c, t * y * z - s * x,
            t * x * z - s * y, t * y * z + s * x, t * z * z + c
        )
    }

    private fun transpose(matrix: DoubleArray) = doubleArrayOf(
        matrix[0], matrix[3], matrix[6],
        matrix[1], matrix[4], matrix[7],
        matrix[2], matrix[5], matrix[8]
    )

    private fun multiply(first: DoubleArray, second: DoubleArray): DoubleArray {
        val output = DoubleArray(9)
        for (row in 0..2) for (column in 0..2) for (axis in 0..2) {
            output[row * 3 + column] += first[row * 3 + axis] * second[axis * 3 + column]
        }
        return output
    }

    private fun multiplyVector(matrix: DoubleArray, vector: DoubleArray): DoubleArray = doubleArrayOf(
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2]
    )

    private fun skew(vector: DoubleArray) = doubleArrayOf(
        0.0, -vector[2], vector[1],
        vector[2], 0.0, -vector[0],
        -vector[1], vector[0], 0.0
    )
}
