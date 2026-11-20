# Android global-solver dependency status

Last reviewed: 2026-11-20.

## Pinned source revisions

- GLOMAP: `99806d0869f802fad218516a2e027793e7ca687d`
- COLMAP revision declared by that GLOMAP source: `b6b7b54eca6078070f73a3f0a084f79c629a6f10`
- Separate later COLMAP reference examined by this project: `e6644e744488a13f3aba82958500db9066341205`
- PoseLib: `7e9f5f53372e43f89655040d4dfc4a00e5ace11c`
- Ceres: `2.2.0`
- Eigen: `3.4.0`

Do not mix headers from the later COLMAP reference with code or libraries built against GLOMAP's declared COLMAP dependency revision.

## Current Android build configuration

The app's `app/src/main/cpp/CMakeLists.txt` now builds the global native solver target and configures:

- Ceres 2.2.0 through FetchContent.
- Eigen 3.4.0 as the sparse linear-algebra backend.
- `EIGENSPARSE=ON` and `SUITESPARSE=OFF`.
- `CUDA=OFF`, `LAPACK=OFF`, `ACCELERATESPARSE=OFF`, and unused test/example/benchmark targets disabled.
- PoseLib at the pinned revision above.
- A native `trackstudio_reconstruction` target containing global positioning, relative-pose estimation, view-graph calibration, and global bundle adjustment.

The main Android workflow builds the release app with the configured production ABIs and runs native numeric, Kotlin unit, and emulator instrumentation tests. A dedicated sparse-solver workflow also builds Ceres on the host and cross-builds Ceres plus a real `SPARSE_SCHUR` smoke executable for `arm64-v8a` and `armeabi-v7a`.

The Android cross-build step compiles the smoke executable; it does not execute that Android binary on a physical device. Physical-device solver validation remains an explicit release gate.

## Why EigenSparse is selected

The pinned GLOMAP global-positioning path selects Ceres `SPARSE_SCHUR`. Its source also assigns `CLUSTER_TRIDIAGONAL`, but that preconditioner applies to `ITERATIVE_SCHUR`, not the selected direct sparse solver. Keep the algorithm's sparse Schur choice and do not silently switch to an iterative solver.

The upstream desktop dependency request for SuiteSparse is not used in the Android build. Ceres is configured with EigenSparse and SuiteSparse disabled, avoiding the GPL/commercial SuiteSparse components. This is a dependency-selection/build adaptation; it does not authorize changing the residuals, robust losses, parameter ordering, gauge constraints, or solver objective.

## Implemented solver behavior

The native global-positioning implementation preserves the relevant track-based objective and settings:

- BATA point-to-camera residuals.
- Scaled Huber loss with threshold 0.1.
- Scale lower bound 1e-5.
- Scale, point, and camera parameter ordering for Schur elimination.
- First active observation scale fixed for scale gauge and first active camera center fixed for translation gauge.
- `SPARSE_SCHUR`, maximum 100 iterations, and function tolerance 1e-5.

The native global bundle-adjustment implementation is a separate joint reprojection solve used by the Android global engine. It has its own loss selection, parameter masks, gauge constraints, and dense/sparse Schur size policy. Do not conflate its configuration with the point-to-camera positioning stage.

## Remaining validation

- Verify deterministic numeric parity against the exact pinned upstream source for representative positioning and bundle-adjustment fixtures.
- Run physical-device tests on both production ABIs where devices are available.
- Record the complete transitive dependency and licence inventory; fail CI if a prohibited GPL component enters the Android dependency graph.
- Keep the source pin and Android build configuration synchronized whenever the upstream source revision changes.

The native solver and JNI integration are implemented; this file is a status and validation record, not a future implementation plan.
