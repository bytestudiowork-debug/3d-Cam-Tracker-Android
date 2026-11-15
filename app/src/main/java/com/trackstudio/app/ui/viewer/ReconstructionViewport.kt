package com.trackstudio.app.ui.viewer

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Dialog
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.DialogProperties
import com.trackstudio.app.tracking.CameraLandmark
import com.trackstudio.app.tracking.CameraPose
import com.trackstudio.app.tracking.CameraPoseGeometry
import com.trackstudio.app.ui.theme.AccentPrimary
import com.trackstudio.app.ui.theme.PanelBackground
import com.trackstudio.app.ui.theme.PanelSurfaceRaised
import com.trackstudio.app.ui.theme.TextPrimary
import com.trackstudio.app.ui.theme.TextSecondary
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

@Composable
internal fun ReconstructionViewportDialog(
    poses: List<CameraPose>,
    landmarks: List<CameraLandmark>,
    currentFrame: Int,
    onDismiss: () -> Unit
) {
    val scene = remember(poses, landmarks) {
        buildScene(poses, landmarks)
    }
    var yaw by remember { mutableFloatStateOf(0.62f) }
    var pitch by remember { mutableFloatStateOf(-0.32f) }
    var zoom by remember { mutableFloatStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(12.dp)
                .clip(RoundedCornerShape(12.dp))
                .background(PanelBackground)
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Column(Modifier.weight(1f)) {
                    Text("3D Reconstruction", color = TextPrimary, fontSize = 17.sp)
                    Text(
                        "${poses.size} camera poses · ${landmarks.size} 3D landmarks · frame ${currentFrame + 1}",
                        color = TextSecondary,
                        fontSize = 11.sp
                    )
                }
                ChromeButton(
                    text = "Reset View",
                    onClick = {
                        yaw = 0.62f
                        pitch = -0.32f
                        zoom = 1f
                        pan = Offset.Zero
                    }
                )
                ChromeButton(text = "Close", onClick = onDismiss)
            }

            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(9.dp))
                    .background(PanelSurfaceRaised)
            ) {
                Canvas(
                    modifier = Modifier
                        .fillMaxSize()
                        .pointerInput(Unit) {
                            awaitEachGesture {
                                val down = awaitFirstDown(requireUnconsumed = false)
                                var previousCentroid = down.position
                                var previousSpan = 0f
                                do {
                                    val event = awaitPointerEvent()
                                    val pressed = event.changes.filter { it.pressed }
                                    if (pressed.isNotEmpty()) {
                                        val centroid = Offset(
                                            pressed.map { it.position.x }.average().toFloat(),
                                            pressed.map { it.position.y }.average().toFloat()
                                        )
                                        if (pressed.size == 1) {
                                            val delta = centroid - previousCentroid
                                            yaw += delta.x * 0.008f
                                            pitch = (pitch + delta.y * 0.008f).coerceIn(-1.45f, 1.45f)
                                            previousSpan = 0f
                                        } else {
                                            val span = pressed.map {
                                                hypot(
                                                    (it.position.x - centroid.x).toDouble(),
                                                    (it.position.y - centroid.y).toDouble()
                                                )
                                            }.average().toFloat()
                                            pan += centroid - previousCentroid
                                            if (previousSpan > 0f && span > 0f) {
                                                zoom = (zoom * span / previousSpan).coerceIn(0.35f, 8f)
                                            }
                                            previousSpan = span
                                        }
                                        previousCentroid = centroid
                                        event.changes.forEach { change ->
                                            if (change.positionChanged()) change.consume()
                                        }
                                    }
                                } while (event.changes.any { it.pressed })
                            }
                        }
                ) {
                    val viewRadius = scene.radius
                    fun project(point: Vec3): Offset {
                        val x = point.x - scene.center.x
                        val y = point.y - scene.center.y
                        val z = point.z - scene.center.z
                        val cosYaw = kotlin.math.cos(yaw.toDouble())
                        val sinYaw = kotlin.math.sin(yaw.toDouble())
                        val rotatedX = cosYaw * x + sinYaw * z
                        val rotatedZ = -sinYaw * x + cosYaw * z
                        val cosPitch = kotlin.math.cos(pitch.toDouble())
                        val sinPitch = kotlin.math.sin(pitch.toDouble())
                        val rotatedY = cosPitch * y - sinPitch * rotatedZ
                        val depth = sinPitch * y + cosPitch * rotatedZ
                        val cameraDistance = viewRadius * 3.2
                        val denominator = max(viewRadius * 0.8, cameraDistance + depth)
                        val scale = (min(size.width, size.height).toDouble() * 0.86 * zoom.toDouble() / denominator).toFloat()
                        return Offset(
                            size.width * 0.5f + pan.x + (rotatedX * scale).toFloat(),
                            size.height * 0.5f + pan.y - (rotatedY * scale).toFloat()
                        )
                    }
                    fun drawWorldLine(start: Vec3, end: Vec3, color: Color, width: Float = 1.5f) {
                        drawLine(color, project(start), project(end), strokeWidth = width)
                    }

                    val axis = viewRadius * 0.18
                    drawWorldLine(scene.center, scene.center + Vec3(axis, 0.0, 0.0), Color(0xFFFF6B6B), 2f)
                    drawWorldLine(scene.center, scene.center + Vec3(0.0, axis, 0.0), Color(0xFF65D98B), 2f)
                    drawWorldLine(scene.center, scene.center + Vec3(0.0, 0.0, axis), Color(0xFF70A7FF), 2f)

                    scene.path.zipWithNext().forEach { (first, second) ->
                        drawWorldLine(first.center, second.center, AccentPrimary.copy(alpha = 0.42f), 1.2f)
                    }

                    scene.points.forEach { point ->
                        drawCircle(
                            color = Color(0xFFB4C7FF).copy(alpha = 0.82f),
                            radius = 1.8f,
                            center = project(point)
                        )
                    }

                    val selectedCamera = scene.cameraByFrame[currentFrame]
                    val camerasToDraw = if (selectedCamera != null && scene.cameras.none { it.frameIndex == currentFrame }) {
                        scene.cameras + selectedCamera
                    } else {
                        scene.cameras
                    }
                    camerasToDraw.forEach { camera ->
                        val selected = camera.frameIndex == currentFrame
                        val color = if (selected) Color(0xFFFFC857) else AccentPrimary
                        val frustum = cameraFrustum(camera, viewRadius)
                        val origin = camera.center
                        val corners = frustum
                        for (corner in corners) drawWorldLine(origin, corner, color.copy(alpha = 0.78f), 1.2f)
                        for (index in corners.indices) {
                            drawWorldLine(corners[index], corners[(index + 1) % corners.size], color.copy(alpha = 0.78f), 1.2f)
                        }
                        drawCircle(
                            color = color,
                            radius = if (selected) 4.5f else 2.8f,
                            center = project(origin)
                        )
                    }

                    drawCircle(
                        color = Color(0xFFFFC857),
                        radius = 5.5f,
                        center = project(scene.center),
                        style = Stroke(width = 1.3f)
                    )
                }

                if (scene.cameras.isEmpty() && scene.points.isEmpty()) {
                    Text(
                        "No 3D solve is available yet. Select Global and run Auto Track first.",
                        color = TextSecondary,
                        fontSize = 13.sp,
                        modifier = Modifier.align(Alignment.Center).padding(24.dp)
                    )
                }

                Row(
                    modifier = Modifier.align(Alignment.BottomStart).padding(10.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    LegendDot(Color(0xFFFFC857), "Current camera")
                    LegendDot(AccentPrimary, "Camera path")
                    LegendDot(Color(0xFFB4C7FF), "3D landmarks")
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("One finger: orbit · Two fingers: pan and zoom", color = TextSecondary, fontSize = 10.sp)
                Text("World-to-camera poses · relative scale", color = TextSecondary, fontSize = 10.sp)
            }
        }
    }
}

@Composable
private fun LegendDot(color: Color, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        Canvas(Modifier.size(7.dp)) { drawCircle(color) }
        Text(label, color = TextSecondary, fontSize = 9.sp)
    }
}

private data class Vec3(val x: Double, val y: Double, val z: Double) {
    operator fun plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
}

private data class ViewCamera(val frameIndex: Int, val center: Vec3, val rotation: DoubleArray)

private data class Scene(
    val center: Vec3,
    val radius: Double,
    val points: List<Vec3>,
    val path: List<ViewCamera>,
    val cameras: List<ViewCamera>,
    val cameraByFrame: Map<Int, ViewCamera>
)

private fun buildScene(poses: List<CameraPose>, landmarks: List<CameraLandmark>): Scene {
    val cameras = poses.asSequence()
        .filter { it.rotation.size == 9 && it.translation.size == 3 }
        .filter { pose -> pose.rotation.all { it.isFinite() } && pose.translation.all { it.isFinite() } }
        .map { pose -> ViewCamera(pose.frameIndex, cameraCenter(pose), pose.rotation) }
        .sortedBy { it.frameIndex }
        .toList()
    val pointStride = max(1, (landmarks.size + MAX_BOUNDS_POINTS - 1) / MAX_BOUNDS_POINTS)
    val points = landmarks.asSequence()
        .withIndex()
        .filter { it.index % pointStride == 0 }
        .map { it.value }
        .filter { it.observationCount >= 2 && it.meanReprojectionErrorPx.isFinite() && it.meanReprojectionErrorPx <= 6.0 }
        .map { Vec3(it.position.x, it.position.y, it.position.z) }
        .filter { it.x.isFinite() && it.y.isFinite() && it.z.isFinite() }
        .take(MAX_BOUNDS_POINTS)
        .toList()
    val bounds = points + cameras.map { it.center }
    if (bounds.isEmpty()) return Scene(Vec3(0.0, 0.0, 0.0), 1.0, emptyList(), emptyList(), emptyList(), emptyMap())

    val xs = bounds.map { it.x }.sorted()
    val ys = bounds.map { it.y }.sorted()
    val zs = bounds.map { it.z }.sorted()
    fun percentile(values: List<Double>, fraction: Double): Double =
        values[((values.size - 1) * fraction).toInt().coerceIn(values.indices)]
    val center = Vec3(
        (percentile(xs, 0.05) + percentile(xs, 0.95)) * 0.5,
        (percentile(ys, 0.05) + percentile(ys, 0.95)) * 0.5,
        (percentile(zs, 0.05) + percentile(zs, 0.95)) * 0.5
    )
    val distances = bounds.map { point ->
        val x = point.x - center.x
        val y = point.y - center.y
        val z = point.z - center.z
        sqrt(x * x + y * y + z * z)
    }.sorted()
    val radius = percentile(distances, 0.95).coerceAtLeast(1e-3)
    val visiblePoints = sampleEvenly(points, MAX_DISPLAY_POINTS)
    val path = sampleEvenly(cameras, MAX_PATH_CAMERAS)
    val shownCameras = sampleEvenly(cameras, MAX_DISPLAY_CAMERAS)
    return Scene(center, radius, visiblePoints, path, shownCameras, cameras.associateBy { it.frameIndex })
}

private fun cameraCenter(pose: CameraPose): Vec3 {
    val center = CameraPoseGeometry.cameraCenterWorld(pose)
    return Vec3(
        center[0],
        center[1],
        center[2]
    )
}

private fun cameraFrustum(camera: ViewCamera, sceneRadius: Double): List<Vec3> {
    val size = sceneRadius * 0.055
    val halfWidth = size * 0.42
    val halfHeight = size * 0.30
    fun worldOffset(x: Double, y: Double, z: Double): Vec3 {
        val offset = CameraPoseGeometry.cameraVectorToWorld(camera.rotation, x, y, z)
        return Vec3(
            offset[0],
            offset[1],
            offset[2]
        )
    }
    return listOf(
        camera.center + worldOffset(-halfWidth, -halfHeight, size),
        camera.center + worldOffset(halfWidth, -halfHeight, size),
        camera.center + worldOffset(halfWidth, halfHeight, size),
        camera.center + worldOffset(-halfWidth, halfHeight, size)
    )
}

private fun <T> sampleEvenly(values: List<T>, maximum: Int): List<T> {
    if (values.size <= maximum) return values
    if (maximum <= 1) return listOf(values.first())
    return (0 until maximum).map { index ->
        values[(index.toDouble() * (values.lastIndex) / (maximum - 1)).toInt()]
    }
}

private const val MAX_DISPLAY_POINTS = 6_000
private const val MAX_BOUNDS_POINTS = 20_000
private const val MAX_PATH_CAMERAS = 1_000
private const val MAX_DISPLAY_CAMERAS = 100
