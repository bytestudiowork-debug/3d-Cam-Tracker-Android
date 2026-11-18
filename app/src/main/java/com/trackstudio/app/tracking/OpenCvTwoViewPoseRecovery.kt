package com.trackstudio.app.tracking

import org.opencv.calib3d.Calib3d
import org.opencv.core.Mat

/**
 * COLMAP-style essential-pose recovery without OpenCV's implicit far-point cutoff.
 *
 * COLMAP chooses an essential pose from positive-depth triangulations; OpenCV's
 * short `recoverPose` overload additionally drops points beyond 50 baseline units.
 * Translation from an essential matrix has arbitrary unit scale, so that fixed
 * cutoff changes the accepted inlier set. Use the explicit overload with the
 * largest finite threshold and leave practical parallax rejection to seed policy.
 */
internal object OpenCvTwoViewPoseRecovery {
    fun recover(
        essential: Mat,
        firstPoints: Mat,
        secondPoints: Mat,
        cameraMatrix: Mat,
        rotation: Mat,
        translation: Mat,
        mask: Mat
    ): Int = Calib3d.recoverPose(
        essential,
        firstPoints,
        secondPoints,
        cameraMatrix,
        rotation,
        translation,
        MAX_TRIANGULATED_DEPTH,
        mask
    )

    private const val MAX_TRIANGULATED_DEPTH = Double.MAX_VALUE
}
