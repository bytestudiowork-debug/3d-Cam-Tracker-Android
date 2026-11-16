package com.trackstudio.app.tracking

/** Serializes the completed reconstruction in a portable, VFX-friendly form. */
internal object CameraSolveExporter {
    fun toJson(result: CameraTrackingResult): String = buildString {
        writeJson(result, this)
    }

    fun writeJson(result: CameraTrackingResult, output: Appendable) {
        val poses = result.poses.asSequence()
            .filter { it.rotation.size == 9 && it.translation.size == 3 }
            .filter { pose -> pose.rotation.all(Double::isFinite) && pose.translation.all(Double::isFinite) }
            .sortedBy { it.frameIndex }
            .toList()
        val landmarks = result.landmarks.asSequence()
            .filter { landmark ->
                landmark.position.x.isFinite() && landmark.position.y.isFinite() && landmark.position.z.isFinite() &&
                    landmark.meanReprojectionErrorPx.isFinite()
            }
            .sortedBy { it.trackId }
            .toList()
        val diagnostics = result.diagnostics.sortedBy { it.frameIndex }
        fun line(value: String) { output.append(value).append('\n') }

        line("{")
        line("  \"schemaVersion\": 1,")
        line("  \"frameIndexBase\": 0,")
        line("  \"coordinateConvention\": \"world-to-camera: X_camera = R * X_world + t\",")
        line("  \"translationScale\": \"arbitrary reconstruction units; not metric\",")
        line("  \"summary\": {")
        line("    \"processedFrames\": ${result.processedFrames},")
        line("    \"registeredCameras\": ${result.registeredCameraCount},")
        line("    \"solvedFramePairs\": ${result.solvedFramePairs},")
        line("    \"totalMatches\": ${result.totalMatches},")
        line("    \"totalInliers\": ${result.totalInliers},")
        line("    \"inlierRatio\": ${jsonNumber(result.inlierRatio)},")
        line("    \"viewGraphEdgeCount\": ${result.viewGraphEdgeCount},")
        line("    \"viewGraphAcceptedEdgeCount\": ${result.viewGraphAcceptedEdgeCount},")
        line("    \"globalRotationIterations\": ${result.globalRotationIterations},")
        line("    \"globalPositioningApplied\": ${result.globalPositioningApplied},")
        line("    \"globalPositioningInitialCost\": ${jsonNumber(result.globalPositioningInitialCost)},")
        line("    \"globalPositioningFinalCost\": ${jsonNumber(result.globalPositioningFinalCost)},")
        line("    \"globalPositioningIterations\": ${result.globalPositioningIterations},")
        line("    \"globalPositioningStatus\": ${jsonString(result.globalPositioningStatus)},")
        line("    \"globalBundleAdjustmentApplied\": ${result.globalBundleAdjustmentApplied},")
        line("    \"globalBundleAdjustmentInitialCost\": ${jsonNumber(result.globalBundleAdjustmentInitialCost)},")
        line("    \"globalBundleAdjustmentFinalCost\": ${jsonNumber(result.globalBundleAdjustmentFinalCost)},")
        line("    \"globalBundleAdjustmentIterations\": ${result.globalBundleAdjustmentIterations},")
        line("    \"globalBundleAdjustmentStatus\": ${jsonString(result.globalBundleAdjustmentStatus)},")
        line("    \"globalBundleAdjustmentPositiveDepthRatio\": ${jsonNumber(result.globalBundleAdjustmentPositiveDepthRatio)},")
        line("    \"globalBundleAdjustmentPrincipalPointRefinementEnabled\": ${result.globalBundleAdjustmentPrincipalPointRefinementEnabled},")
        line("    \"jointBundleAdjustmentApplied\": ${result.jointBundleAdjustmentApplied},")
        line("    \"jointBundleAdjustmentInitialCost\": ${jsonNumber(result.jointBundleAdjustmentInitialCost)},")
        line("    \"jointBundleAdjustmentFinalCost\": ${jsonNumber(result.jointBundleAdjustmentFinalCost)},")
        line("    \"jointBundleAdjustmentIterations\": ${result.jointBundleAdjustmentIterations},")
        line("    \"jointBundleAdjustmentStatus\": ${jsonString(result.jointBundleAdjustmentStatus)},")
        line("    \"jointBundleAdjustmentPositiveDepthRatio\": ${jsonNumber(result.jointBundleAdjustmentPositiveDepthRatio)},")
        line("    \"jointBundleAdjustmentPrincipalPointRefinementEnabled\": ${result.jointBundleAdjustmentPrincipalPointRefinementEnabled},")
        line("    \"initialPairFrames\": ${result.initialPairFrames?.let { "[${it.first}, ${it.second}]" } ?: "null"}")
        line("  },")
        line("  \"observationCoordinateModel\": ${jsonString(result.observationCoordinateModel.name)},")
        val sourceCalibrationJson = result.inputCameraCalibration?.let { calibration ->
            "{\"imageWidth\": ${calibration.imageWidth}, \"imageHeight\": ${calibration.imageHeight}, " +
                "\"fx\": ${jsonNumber(calibration.fx)}, \"skew\": ${jsonNumber(calibration.skew)}, " +
                "\"cx\": ${jsonNumber(calibration.cx)}, \"fy\": ${jsonNumber(calibration.fy)}, " +
                "\"cy\": ${jsonNumber(calibration.cy)}, \"distortionCoefficients\": " +
                jsonArray(calibration.distortionCoefficients.toDoubleArray()) + "}"
        } ?: "null"
        line("  \"inputCameraCalibration\": $sourceCalibrationJson,")
        val intrinsics = result.refinedCameraIntrinsics
        val intrinsicsJson = intrinsics?.let {
            "{\"fx\": ${jsonNumber(it.fx)}, \"skew\": ${jsonNumber(it.skew)}, " +
                "\"cx\": ${jsonNumber(it.cx)}, \"fy\": ${jsonNumber(it.fy)}, " +
                "\"cy\": ${jsonNumber(it.cy)}}"
        } ?: "null"
        line("  \"refinedCameraIntrinsics\": $intrinsicsJson,")
        val effectiveIntrinsicsJson = result.effectiveCameraIntrinsics?.let {
            "{\"fx\": ${jsonNumber(it.fx)}, \"skew\": ${jsonNumber(it.skew)}, " +
                "\"cx\": ${jsonNumber(it.cx)}, \"fy\": ${jsonNumber(it.fy)}, " +
                "\"cy\": ${jsonNumber(it.cy)}}"
        } ?: "null"
        line("  \"effectiveCameraIntrinsics\": $effectiveIntrinsicsJson,")
        line("  \"cameraPoses\": [")
        poses.forEachIndexed { index, pose ->
            val center = CameraPoseGeometry.cameraCenterWorld(pose)
            output.append("    {\"frameIndex\": ${pose.frameIndex}, \"status\": ${jsonString(pose.status.name)}, ")
            output.append("\"rotationWorldToCameraRowMajor\": ${jsonArray(pose.rotation)}, ")
            output.append("\"translationWorldToCamera\": ${jsonArray(pose.translation)}, ")
            output.append("\"centerWorld\": ${jsonArray(center)}}")
            line(if (index == poses.lastIndex) "" else ",")
        }
        line("  ],")
        line("  \"landmarks\": [")
        landmarks.forEachIndexed { index, landmark ->
            output.append("    {\"trackId\": ${landmark.trackId}, \"positionWorld\": ${jsonArray(doubleArrayOf(landmark.position.x, landmark.position.y, landmark.position.z))}, ")
            output.append("\"observationCount\": ${landmark.observationCount}, ")
            output.append("\"meanReprojectionErrorPx\": ${jsonNumber(landmark.meanReprojectionErrorPx)}, ")
            output.append("\"triangulationAngleDegrees\": ${jsonNumber(landmark.triangulationAngleDegrees)}}")
            line(if (index == landmarks.lastIndex) "" else ",")
        }
        line("  ],")
        line("  \"diagnostics\": [")
        diagnostics.forEachIndexed { index, diagnostic ->
            output.append("    {\"frameIndex\": ${diagnostic.frameIndex}, \"detectedFeatures\": ${diagnostic.detectedFeatures}, ")
            output.append("\"matchedFeatures\": ${diagnostic.matchedFeatures}, \"inlierFeatures\": ${diagnostic.inlierFeatures}, ")
            output.append("\"status\": ${jsonString(diagnostic.status)}}")
            line(if (index == diagnostics.lastIndex) "" else ",")
        }
        line("  ]")
        line("}")
    }

    private fun jsonArray(values: DoubleArray): String =
        values.joinToString(prefix = "[", postfix = "]", separator = ", ", transform = ::jsonNumber)

    private fun jsonNumber(value: Double): String =
        if (value.isFinite()) value.toString() else "null"

    private fun jsonString(value: String): String = buildString {
        append('"')
        value.forEach { character ->
            when (character) {
                '\\' -> append("\\\\")
                '"' -> append("\\\"")
                '\b' -> append("\\b")
                '\u000C' -> append("\\f")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (character < ' ') append("\\u%04x".format(character.code)) else append(character)
            }
        }
        append('"')
    }
}
