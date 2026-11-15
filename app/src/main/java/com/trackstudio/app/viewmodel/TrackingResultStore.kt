package com.trackstudio.app.viewmodel

import android.content.Context
import android.net.Uri
import android.util.JsonReader
import android.util.JsonToken
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraFeatureObservation
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseStatus
import com.trackstudio.app.tracking.CameraSolveExporter
import com.trackstudio.app.tracking.CameraTrackingResult
import com.trackstudio.app.tracking.TrackingAnchorSegment
import com.trackstudio.app.util.FileUtils
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.Point3
import java.io.File
import java.io.InputStreamReader

/**
 * Stores the viewer-facing tracking result inside the project cache.
 * This keeps completed tracking overlays and solve diagnostics available
 * after the app is closed or the device is restarted.
 */
internal object TrackingResultStore {
    data class Snapshot(
        val pointsByFrame: Map<Int, List<CameraFeatureObservation>>,
        val anchorSegmentsByFrame: Map<Int, List<TrackingAnchorSegment>>,
        val summary: CameraTrackingSummary,
        val anchorX: Double,
        val anchorY: Double,
        val anchorZ: Double,
        val defaultAnchorX: Double,
        val defaultAnchorY: Double,
        val defaultAnchorZ: Double,
        val anchorAxisLength: Double,
        val cameraSolveResult: CameraTrackingResult? = null,
        val cameraSolveAvailable: Boolean = false,
        val cameraPoses: List<CameraPose> = emptyList(),
        val cameraLandmarks: List<CameraLandmark> = emptyList()
    )

    private data class CameraGeometry(
        val poses: List<CameraPose>,
        val landmarks: List<CameraLandmark>
    )

    private fun file(context: Context, projectId: String) =
        File(FileUtils.projectCacheDir(context, projectId), "tracking_result.json")

    private fun cameraSolveFile(context: Context, projectId: String) =
        File(FileUtils.projectCacheDir(context, projectId), "camera_solve.json")

    fun save(context: Context, projectId: String, snapshot: Snapshot) {
        if (projectId.isBlank()) return
        val json = JSONObject().apply {
            put("version", 1)
            put("summary", JSONObject().apply {
                put("registeredCameras", snapshot.summary.registeredCameras)
                put("totalFrames", snapshot.summary.totalFrames)
                put("landmarkCount", snapshot.summary.landmarkCount)
                put("inlierRatio", snapshot.summary.inlierRatio)
                put("initialPairFirst", snapshot.summary.initialPairFrames?.first ?: -1)
                put("initialPairSecond", snapshot.summary.initialPairFrames?.second ?: -1)
                put("meanReprojectionErrorPx", snapshot.summary.meanReprojectionErrorPx)
                put("globalPositioningApplied", snapshot.summary.globalPositioningApplied)
                put("globalPositioningInitialCost", snapshot.summary.globalPositioningInitialCost.takeIf { it.isFinite() } ?: JSONObject.NULL)
                put("globalPositioningFinalCost", snapshot.summary.globalPositioningFinalCost.takeIf { it.isFinite() } ?: JSONObject.NULL)
                put("globalPositioningIterations", snapshot.summary.globalPositioningIterations)
                put("globalPositioningStatus", snapshot.summary.globalPositioningStatus)
                put("viewGraphEdgeCount", snapshot.summary.viewGraphEdgeCount)
                put("viewGraphAcceptedEdgeCount", snapshot.summary.viewGraphAcceptedEdgeCount)
                put("globalRotationIterations", snapshot.summary.globalRotationIterations)
            })
            put("anchor", JSONObject().apply {
                put("x", snapshot.anchorX); put("y", snapshot.anchorY); put("z", snapshot.anchorZ)
                put("defaultX", snapshot.defaultAnchorX); put("defaultY", snapshot.defaultAnchorY); put("defaultZ", snapshot.defaultAnchorZ)
                put("axisLength", snapshot.anchorAxisLength)
            })
            put("points", JSONArray().apply {
                snapshot.pointsByFrame.toSortedMap().forEach { (frame, observations) ->
                    observations.forEach { observation ->
                        put(JSONArray().put(frame).put(observation.x).put(observation.y))
                    }
                }
            })
            put("anchorSegments", JSONArray().apply {
                snapshot.anchorSegmentsByFrame.toSortedMap().forEach { (frame, segments) ->
                    segments.forEach { segment ->
                        put(JSONArray().put(frame).put(segment.startX).put(segment.startY)
                            .put(segment.endX).put(segment.endY).put(segment.colorArgb))
                    }
                }
            })
        }
        val destination = file(context, projectId)
        val temporary = File(destination.parentFile, destination.name + ".tmp")
        temporary.writeText(json.toString())
        if (destination.exists() && !destination.delete()) {
            temporary.delete()
            error("Could not replace the previous tracking result.")
        }
        if (!temporary.renameTo(destination)) {
            temporary.delete()
            error("Could not save the tracking result.")
        }
        snapshot.cameraSolveResult?.let { cameraSolve ->
            writeCameraSolveAtomically(cameraSolveFile(context, projectId), cameraSolve)
        }
    }

    fun load(context: Context, projectId: String): Snapshot? {
        if (projectId.isBlank()) return null
        val source = file(context, projectId)
        if (!source.isFile) return null
        return runCatching {
            val json = JSONObject(source.readText())
            require(json.optInt("version", 0) == 1)
            val summaryJson = json.getJSONObject("summary")
            val first = summaryJson.optInt("initialPairFirst", -1)
            val second = summaryJson.optInt("initialPairSecond", -1)
            val initialPair = if (first >= 0 && second > first) first to second else null
            val summary = CameraTrackingSummary(
                registeredCameras = summaryJson.optInt("registeredCameras"),
                totalFrames = summaryJson.optInt("totalFrames"),
                landmarkCount = summaryJson.optInt("landmarkCount"),
                inlierRatio = summaryJson.optDouble("inlierRatio"),
                initialPairFrames = initialPair,
                meanReprojectionErrorPx = summaryJson.optDouble("meanReprojectionErrorPx"),
                globalPositioningApplied = summaryJson.optBoolean("globalPositioningApplied"),
                globalPositioningInitialCost = summaryJson.optDouble("globalPositioningInitialCost", Double.NaN),
                globalPositioningFinalCost = summaryJson.optDouble("globalPositioningFinalCost", Double.NaN),
                globalPositioningIterations = summaryJson.optInt("globalPositioningIterations"),
                globalPositioningStatus = summaryJson.optString("globalPositioningStatus", "Not run"),
                viewGraphEdgeCount = summaryJson.optInt("viewGraphEdgeCount"),
                viewGraphAcceptedEdgeCount = summaryJson.optInt("viewGraphAcceptedEdgeCount"),
                globalRotationIterations = summaryJson.optInt("globalRotationIterations")
            )
            val points = linkedMapOf<Int, MutableList<CameraFeatureObservation>>()
            val pointArray = json.optJSONArray("points") ?: JSONArray()
            for (i in 0 until pointArray.length()) {
                val row = pointArray.getJSONArray(i)
                val frame = row.getInt(0)
                points.getOrPut(frame) { mutableListOf() }.add(
                    CameraFeatureObservation(frame, row.getDouble(1), row.getDouble(2))
                )
            }
            val segments = linkedMapOf<Int, MutableList<TrackingAnchorSegment>>()
            val segmentArray = json.optJSONArray("anchorSegments") ?: JSONArray()
            for (i in 0 until segmentArray.length()) {
                val row = segmentArray.getJSONArray(i)
                val frame = row.getInt(0)
                segments.getOrPut(frame) { mutableListOf() }.add(
                    TrackingAnchorSegment(
                        row.getDouble(1).toFloat(), row.getDouble(2).toFloat(),
                        row.getDouble(3).toFloat(), row.getDouble(4).toFloat(),
                        row.getLong(5)
                    )
                )
            }
            val anchor = json.optJSONObject("anchor") ?: JSONObject()
            val cameraGeometry = runCatching { readCameraGeometry(cameraSolveFile(context, projectId)) }.getOrNull()
            Snapshot(
                points.mapValues { it.value.toList() },
                segments.mapValues { it.value.toList() },
                summary,
                anchor.optDouble("x"), anchor.optDouble("y"), anchor.optDouble("z"),
                anchor.optDouble("defaultX"), anchor.optDouble("defaultY"), anchor.optDouble("defaultZ"),
                anchor.optDouble("axisLength", 0.1),
                cameraSolveAvailable = cameraSolveFile(context, projectId).isFile,
                cameraPoses = cameraGeometry?.poses.orEmpty(),
                cameraLandmarks = cameraGeometry?.landmarks.orEmpty()
            )
        }.getOrNull()
    }

    fun exportCameraSolve(context: Context, projectId: String, destination: Uri) {
        require(projectId.isNotBlank()) { "The project has no saved camera solve." }
        val source = cameraSolveFile(context, projectId)
        require(source.isFile) { "The saved camera solve is unavailable. Run Auto Track again." }
        val output = context.contentResolver.openOutputStream(destination, "wt")
            ?: error("Could not open the selected export destination.")
        output.use { destinationStream ->
            source.inputStream().buffered().use { input -> input.copyTo(destinationStream) }
        }
    }

    private fun writeCameraSolveAtomically(destination: File, result: CameraTrackingResult) {
        val temporary = File(destination.parentFile, destination.name + ".tmp")
        temporary.bufferedWriter(Charsets.UTF_8).use { writer ->
            CameraSolveExporter.writeJson(result, writer)
        }
        if (destination.exists() && !destination.delete()) {
            temporary.delete()
            error("Could not replace the previous camera solve.")
        }
        if (!temporary.renameTo(destination)) {
            temporary.delete()
            error("Could not save the camera solve.")
        }
    }

    private fun readCameraGeometry(source: File): CameraGeometry? {
        if (!source.isFile) return null
        val poses = mutableListOf<CameraPose>()
        val landmarks = mutableListOf<CameraLandmark>()
        JsonReader(InputStreamReader(source.inputStream(), Charsets.UTF_8)).use { reader ->
            reader.beginObject()
            while (reader.hasNext()) {
                when (reader.nextName()) {
                    "cameraPoses" -> {
                        reader.beginArray()
                        while (reader.hasNext()) readCameraPose(reader)?.let(poses::add)
                        reader.endArray()
                    }
                    "landmarks" -> {
                        reader.beginArray()
                        while (reader.hasNext()) readLandmark(reader)?.let(landmarks::add)
                        reader.endArray()
                    }
                    else -> reader.skipValue()
                }
            }
            reader.endObject()
        }
        return CameraGeometry(poses.sortedBy { it.frameIndex }, landmarks)
    }

    private fun readCameraPose(reader: JsonReader): CameraPose? {
        var frameIndex = -1
        var status = CameraPoseStatus.RELATIVE_ESTIMATE
        var rotation = doubleArrayOf()
        var translation = doubleArrayOf()
        reader.beginObject()
        while (reader.hasNext()) {
            when (reader.nextName()) {
                "frameIndex" -> frameIndex = reader.nextInt()
                "status" -> status = runCatching { CameraPoseStatus.valueOf(reader.nextString()) }
                    .getOrDefault(CameraPoseStatus.RELATIVE_ESTIMATE)
                "rotationWorldToCameraRowMajor" -> rotation = readDoubleArray(reader)
                "translationWorldToCamera" -> translation = readDoubleArray(reader)
                else -> reader.skipValue()
            }
        }
        reader.endObject()
        if (frameIndex < 0 || rotation.size != 9 || translation.size != 3 ||
            rotation.any { !it.isFinite() } || translation.any { !it.isFinite() }
        ) return null
        return CameraPose(frameIndex, rotation, translation, status)
    }

    private fun readLandmark(reader: JsonReader): CameraLandmark? {
        var trackId = -1
        var position = doubleArrayOf()
        var observationCount = 0
        var meanError = Double.NaN
        var angle = 0.0
        reader.beginObject()
        while (reader.hasNext()) {
            when (reader.nextName()) {
                "trackId" -> trackId = reader.nextInt()
                "positionWorld" -> position = readDoubleArray(reader)
                "observationCount" -> observationCount = reader.nextInt()
                "meanReprojectionErrorPx" -> meanError = readNullableDouble(reader)
                "triangulationAngleDegrees" -> angle = readNullableDouble(reader)
                else -> reader.skipValue()
            }
        }
        reader.endObject()
        if (trackId < 0 || position.size != 3 || position.any { !it.isFinite() } ||
            observationCount < 0 || !meanError.isFinite()
        ) return null
        return CameraLandmark(trackId, Point3(position[0], position[1], position[2]), observationCount, meanError, angle)
    }

    private fun readDoubleArray(reader: JsonReader): DoubleArray {
        val values = mutableListOf<Double>()
        reader.beginArray()
        while (reader.hasNext()) values += readNullableDouble(reader)
        reader.endArray()
        return values.toDoubleArray()
    }

    private fun readNullableDouble(reader: JsonReader): Double =
        if (reader.peek() == JsonToken.NULL) {
            reader.nextNull()
            Double.NaN
        } else {
            reader.nextDouble()
        }
}
