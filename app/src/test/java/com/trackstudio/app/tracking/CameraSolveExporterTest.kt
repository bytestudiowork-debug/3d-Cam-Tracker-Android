package com.trackstudio.app.tracking

import org.junit.Assert.assertFalse
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point3

class CameraSolveExporterTest {
    @Test
    fun exportsCameraTrajectoryLandmarksAndSolveDiagnostics() {
        val result = CameraTrackingResult(
            poses = listOf(
                CameraPose(
                    frameIndex = 0,
                    rotation = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    translation = doubleArrayOf(-1.0, 2.0, 3.0),
                    status = CameraPoseStatus.REFERENCE
                )
            ),
            diagnostics = listOf(FrameTrackingDiagnostic(0, 120, 90, 80, "Global reconstruction registered")),
            processedFrames = 1,
            solvedFramePairs = 1,
            totalMatches = 90,
            totalInliers = 80,
            inlierRatio = 0.88,
            isCameraMotionEstimated = true,
            landmarks = listOf(CameraLandmark(7, Point3(4.0, 5.0, 6.0), 3, 0.75, 14.0)),
            registeredCameraCount = 1,
            globalPositioningApplied = true,
            globalPositioningInitialCost = 15.0,
            globalPositioningFinalCost = 2.5,
            globalPositioningIterations = 12,
            globalPositioningStatus = "Global solve completed",
            globalBundleAdjustmentApplied = true,
            globalBundleAdjustmentInitialCost = 2.5,
            globalBundleAdjustmentFinalCost = 0.8,
            globalBundleAdjustmentIterations = 5,
            globalBundleAdjustmentStatus = "Applied",
            globalBundleAdjustmentPositiveDepthRatio = 0.97,
            globalBundleAdjustmentPrincipalPointRefinementEnabled = true,
            jointBundleAdjustmentApplied = true,
            jointBundleAdjustmentInitialCost = 2.5,
            jointBundleAdjustmentFinalCost = 0.8,
            jointBundleAdjustmentIterations = 5,
            jointBundleAdjustmentStatus = "Applied",
            jointBundleAdjustmentPositiveDepthRatio = 0.97,
            jointBundleAdjustmentPrincipalPointRefinementEnabled = true,
            inputCameraCalibration = CameraCalibration(
                imageWidth = 1280,
                imageHeight = 720,
                fx = 700.0,
                fy = 710.0,
                cx = 640.0,
                cy = 360.0,
                distortionCoefficients = listOf(0.1, -0.02, 0.001, -0.001, 0.003)
            ),
            observationCoordinateModel = CameraObservationCoordinateModel.CALIBRATED_PINHOLE_PIXELS,
            refinedCameraIntrinsics = CameraIntrinsics(700.0, 0.0, 640.0, 710.0, 360.0),
            viewGraphEdgeCount = 4,
            viewGraphAcceptedEdgeCount = 3,
            globalRotationIterations = 8
        )

        val json = CameraSolveExporter.toJson(result)

        assertTrue(json.contains("\"coordinateConvention\": \"world-to-camera"))
        assertTrue(json.contains("\"rotationWorldToCameraRowMajor\": [1.0, 0.0, 0.0"))
        assertTrue(json.contains("\"translationWorldToCamera\": [-1.0, 2.0, 3.0]"))
        assertTrue(json.contains("\"centerWorld\": [1.0, -2.0, -3.0]"))
        assertTrue(json.contains("\"positionWorld\": [4.0, 5.0, 6.0]"))
        assertTrue(json.contains("\"viewGraphAcceptedEdgeCount\": 3"))
        assertTrue(json.contains("\"globalBundleAdjustmentApplied\": true"))
        assertTrue(json.contains("\"globalPositioningInitialCost\": 15.0"))
        assertTrue(json.contains("\"globalBundleAdjustmentInitialCost\": 2.5"))
        assertTrue(json.contains("\"globalBundleAdjustmentPositiveDepthRatio\": 0.97"))
        assertTrue(json.contains("\"globalBundleAdjustmentPrincipalPointRefinementEnabled\": true"))
        assertTrue(json.contains("\"jointBundleAdjustmentApplied\": true"))
        assertTrue(json.contains("\"jointBundleAdjustmentInitialCost\": 2.5"))
        assertTrue(json.contains("\"jointBundleAdjustmentIterations\": 5"))
        assertTrue(json.contains("\"jointBundleAdjustmentPositiveDepthRatio\": 0.97"))
        assertTrue(json.contains("\"observationCoordinateModel\": \"CALIBRATED_PINHOLE_PIXELS\""))
        assertTrue(json.contains("\"distortionCoefficients\": [0.1, -0.02, 0.001, -0.001, 0.003]"))
        assertTrue(json.contains("\"refinedCameraIntrinsics\": {\"fx\": 700.0, \"skew\": 0.0"))
        assertTrue(json.contains("\"effectiveCameraIntrinsics\": {\"fx\": 700.0, \"skew\": 0.0"))
        assertTrue(json.contains("\"status\": \"Global reconstruction registered\""))
        assertFalse(json.contains("NaN"))
        assertFalse(json.contains("Infinity"))
    }

    @Test
    fun exportsFeatureTracksWithStableObservationOrdering() {
        val result = CameraTrackingResult(
            poses = emptyList(),
            diagnostics = emptyList(),
            processedFrames = 3,
            solvedFramePairs = 0,
            totalMatches = 0,
            totalInliers = 0,
            inlierRatio = 0.0,
            isCameraMotionEstimated = false,
            featureTracks = listOf(
                CameraFeatureTrack(
                    id = 9,
                    observations = listOf(
                        CameraFeatureObservation(2, 22.5, 32.5, 4),
                        CameraFeatureObservation(0, 10.5, 20.5, 1),
                        CameraFeatureObservation(1, 16.5, 26.5, 3)
                    )
                ),
                CameraFeatureTrack(
                    id = 3,
                    observations = listOf(
                        CameraFeatureObservation(1, 40.0, 50.0, 7),
                        CameraFeatureObservation(0, 30.0, 35.0, 5)
                    )
                )
            )
        )

        val json = CameraSolveExporter.toJson(result)

        assertTrue(json.contains("\"featureTracks\": ["))
        assertTrue(json.indexOf("\"trackId\": 3") < json.indexOf("\"trackId\": 9"))
        assertTrue(
            json.indexOf("\"frameIndex\": 0, \"featureIndex\": 1") <
                json.indexOf("\"frameIndex\": 1, \"featureIndex\": 3")
        )
        assertTrue(json.contains("\"length\": 3"))
        assertTrue(json.contains("\"x\": 10.5, \"y\": 20.5"))
        assertTrue(json.contains("\"featureIndex\": 4"))
    }

    @Test
    fun excludesInvalidFeatureObservationsFromPortableTrackExport() {
        val result = CameraTrackingResult(
            poses = emptyList(),
            diagnostics = emptyList(),
            processedFrames = 2,
            solvedFramePairs = 0,
            totalMatches = 0,
            totalInliers = 0,
            inlierRatio = 0.0,
            isCameraMotionEstimated = false,
            featureTracks = listOf(
                CameraFeatureTrack(
                    id = 5,
                    observations = listOf(
                        CameraFeatureObservation(0, Double.NaN, 1.0, 0),
                        CameraFeatureObservation(1, 2.0, Double.POSITIVE_INFINITY, 1)
                    )
                ),
                CameraFeatureTrack(
                    id = 6,
                    observations = listOf(CameraFeatureObservation(0, 12.0, 13.0, 2))
                )
            )
        )

        val json = CameraSolveExporter.toJson(result)

        assertFalse(json.contains("\"trackId\": 5"))
        assertTrue(json.contains("\"trackId\": 6"))
        assertFalse(json.contains("NaN"))
        assertFalse(json.contains("Infinity"))
    }

    @Test
    fun refinedIntrinsicsUseOpenCvRowMajorMatrixOrder() {
        val matrix = CameraIntrinsics(700.0, 2.0, 640.0, 710.0, 360.0).toRowMajorMatrix()

        assertArrayEquals(
            doubleArrayOf(700.0, 2.0, 640.0, 0.0, 710.0, 360.0, 0.0, 0.0, 1.0),
            matrix,
            0.0
        )
    }

    @Test
    fun effectiveIntrinsicsFallBackToSuppliedCalibration() {
        val calibration = CameraCalibration(
            imageWidth = 640,
            imageHeight = 480,
            fx = 500.0,
            fy = 510.0,
            cx = 320.0,
            cy = 240.0,
            skew = 0.5,
            distortionCoefficients = listOf(0.1, -0.02, 0.001, -0.001)
        )
        val result = CameraTrackingResult(
            poses = emptyList(),
            diagnostics = emptyList(),
            processedFrames = 0,
            solvedFramePairs = 0,
            totalMatches = 0,
            totalInliers = 0,
            inlierRatio = 0.0,
            isCameraMotionEstimated = false,
            inputCameraCalibration = calibration
        )

        assertEquals(CameraIntrinsics(500.0, 0.5, 320.0, 510.0, 240.0), result.effectiveCameraIntrinsics)
    }

    @Test
    fun observationIntrinsicsDescribeRectifiedPixelsInsteadOfSourceSkew() {
        val calibration = CameraCalibration(
            imageWidth = 640,
            imageHeight = 480,
            fx = 500.0,
            fy = 510.0,
            cx = 320.0,
            cy = 240.0,
            skew = 5.0,
            distortionCoefficients = listOf(0.08, -0.02, 0.001, -0.001)
        )
        val observationIntrinsics = CameraIntrinsics(500.0, 0.0, 320.0, 510.0, 240.0)
        val result = CameraTrackingResult(
            poses = emptyList(),
            diagnostics = emptyList(),
            processedFrames = 0,
            solvedFramePairs = 0,
            totalMatches = 0,
            totalInliers = 0,
            inlierRatio = 0.0,
            isCameraMotionEstimated = false,
            inputCameraCalibration = calibration,
            observationCoordinateModel = CameraObservationCoordinateModel.CALIBRATED_PINHOLE_PIXELS,
            observationCameraIntrinsics = observationIntrinsics
        )

        val json = CameraSolveExporter.toJson(result)

        assertEquals(observationIntrinsics, result.effectiveCameraIntrinsics)
        assertTrue(json.contains("\"inputCameraCalibration\": {\"imageWidth\": 640"))
        assertTrue(json.contains("\"observationCameraIntrinsics\": {\"fx\": 500.0, \"skew\": 0.0"))
        assertTrue(json.contains("\"effectiveCameraIntrinsics\": {\"fx\": 500.0, \"skew\": 0.0"))
    }

    @Test
    fun nonFiniteOptionalCostsAreWrittenAsJsonNull() {
        val result = CameraTrackingResult(
            poses = emptyList(),
            diagnostics = emptyList(),
            processedFrames = 0,
            solvedFramePairs = 0,
            totalMatches = 0,
            totalInliers = 0,
            inlierRatio = 0.0,
            isCameraMotionEstimated = false
        )

        val json = CameraSolveExporter.toJson(result)

        assertTrue(json.contains("\"globalPositioningInitialCost\": null"))
        assertTrue(json.contains("\"globalPositioningFinalCost\": null"))
        assertTrue(json.contains("\"globalBundleAdjustmentInitialCost\": null"))
        assertTrue(json.contains("\"globalBundleAdjustmentFinalCost\": null"))
        assertTrue(json.contains("\"globalBundleAdjustmentPositiveDepthRatio\": null"))
        assertTrue(json.contains("\"globalBundleAdjustmentPrincipalPointRefinementEnabled\": false"))
        assertTrue(json.contains("\"jointBundleAdjustmentInitialCost\": null"))
        assertTrue(json.contains("\"jointBundleAdjustmentFinalCost\": null"))
        assertTrue(json.contains("\"jointBundleAdjustmentPositiveDepthRatio\": null"))
        assertTrue(json.contains("\"jointBundleAdjustmentPrincipalPointRefinementEnabled\": false"))
        assertTrue(json.contains("\"inputCameraCalibration\": null"))
        assertTrue(json.contains("\"refinedCameraIntrinsics\": null"))
        assertTrue(json.contains("\"effectiveCameraIntrinsics\": null"))
    }

    @Test
    fun cameraSpaceVectorUsesTransposeForWorldDirection() {
        val rotationWorldToCamera = doubleArrayOf(
            0.0, -1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0
        )

        val worldDirection = CameraPoseGeometry.cameraVectorToWorld(rotationWorldToCamera, 1.0, 0.0, 0.0)

        assertArrayEquals(doubleArrayOf(0.0, -1.0, 0.0), worldDirection, 1e-12)
    }
}
