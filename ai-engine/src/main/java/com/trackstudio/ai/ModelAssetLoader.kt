package com.trackstudio.ai

import android.content.Context
import java.io.File
import java.io.FileOutputStream

/**
 * Copies uncompressed model assets into private app storage without exposing
 * partially written files to ONNX Runtime sessions.
 */
internal object ModelAssetLoader {
    fun copyAsset(context: Context, assetPath: String): File {
        require(assetPath.isNotBlank() && !assetPath.startsWith("/") &&
            assetPath.split('/').none { it == ".." }
        ) { "Invalid model asset path." }

        val directory = File(context.filesDir, "trackstudio-models")
        if (!directory.exists() && !directory.mkdirs()) {
            throw IllegalStateException("Unable to create the model directory.")
        }

        val updateStamp = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime
        val target = File(directory, assetPath.substringAfterLast('/') + "-" + updateStamp)
        val expectedLength = context.assets.openFd(assetPath).use { it.length }
        if (target.isFile && target.length() == expectedLength) return target

        val temporary = File.createTempFile(target.name, ".partial", directory)
        try {
            context.assets.open(assetPath).use { input ->
                FileOutputStream(temporary).use { output ->
                    input.copyTo(output)
                    output.fd.sync()
                }
            }
            check(temporary.length() == expectedLength) {
                "The copied feature model asset has an unexpected size."
            }

            if (target.exists() && !target.delete()) {
                throw IllegalStateException("Unable to replace an incomplete model asset.")
            }
            if (!temporary.renameTo(target)) {
                if (target.isFile && target.length() == expectedLength) return target
                throw IllegalStateException("Unable to atomically install the feature model asset.")
            }
            return target
        } finally {
            temporary.delete()
        }
    }
}
