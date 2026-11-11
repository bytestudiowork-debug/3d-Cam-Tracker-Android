# Android geometry port status

Status date: 2026-11-11.

## Implemented in the app

- The Kotlin global-positioning initializer, seeded random generator and
  ONLY_POINTS problem builder are present.
- Android CMake builds Eigen 3.4.0 and Ceres 2.2.0 for the app's supported
  ABIs. SuiteSparse, CUDA and LAPACK are disabled; EigenSparse is enabled.
- The JNI target `trackstudio_glomap` contains a point-to-camera global
  positioning solve using direction residuals, per-observation scales,
  robust loss, sparse Schur solving and a scale gauge constraint.
- The JNI bundle-adjustment path jointly refines camera centres and 3D
  points using reprojection residuals while holding camera rotations fixed.
- The existing global reconstruction engine uses its current OpenCV feature
  extraction and matching path, then runs the global geometry stages.

## Not yet complete

- The native solver files are focused Android implementations, not a direct
  build of the upstream COLMAP/GLOMAP libraries.
- The current bundle-adjustment path is not the full upstream optimizer:
  rotations and camera intrinsics are held fixed.
- The native solvers still need deterministic numerical parity tests against
  the pinned upstream implementation.
- The AI feature module now has an adapter that converts learned feature
  observations into the frame representation consumed by the existing
  global geometry engine. The existing OpenCV path remains the default.
  No model weights are packaged, so actual learned inference is not yet
  available in the app.
- Loop-closure scoring from learned frame signatures and complete device
  validation remain outstanding.

## Build validation

The release APK workflow completed successfully on commit
`a588ab5f0cf1d278da1d9da83c48d3b95e1e42fd` on 2026-11-11. The dedicated
Ceres sparse-solver smoke workflow has been enabled for
`feature/ai-engine`; its host and Android ABI checks are tracked by GitHub
Actions.

## Architecture boundary

The Custom and Global engines remain separate from the AI Engine. The next integration stage is to package and validate compatible model
weights, then compare learned-feature reconstruction against the existing
OpenCV baseline. The adapter does not replace the OpenCV path or change the
Custom engine.
