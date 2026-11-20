# Android reconstruction port status

Audit date: 2026-11-20.

## Current implementation

The Android project contains an integrated global reconstruction path. It is a focused mobile adaptation of selected upstream components, not a build of the complete desktop applications.

### Global positioning

Implemented in native C++ and exposed through JNI:

- `global_positioning_core.cpp` builds the point-to-camera BATA objective with one scale parameter per observation.
- Uses scaled Huber losses with threshold 0.1 and scale lower bound 1e-5.
- Uses Ceres `SPARSE_SCHUR`, the upstream scale/point/camera parameter ordering, a 100-iteration maximum, and function tolerance 1e-5.
- Fixes the first active camera center to establish the translation origin and fixes the first active observation scale for the scale gauge.
- Rejects disconnected active camera components and malformed/non-finite input before solving.
- `glomap_positioner.cpp` exposes the solver through JNI, including progress and cancellation callbacks.
- `GlomapNativeSolver.kt` packs stable input order, validates the returned result, and maps solved values back to frame and track IDs.

The Kotlin initializer and problem builder prepare the source-compatible random initialization and point-to-camera observations. They are no longer the end of the positioning path: the native Ceres solve is implemented and called by the global engine.

### Relative pose and view-graph calibration

- `glomap_relative_pose_estimator_core.cpp` adapts the pinned relative-pose estimation path and calls the pinned PoseLib dependency.
- `glomap_view_graph_calibrator.cpp` exposes native Ceres view-graph focal calibration.
- The global engine uses these components while constructing and filtering its verified view graph.

### Global bundle adjustment

- `global_bundle_adjuster_core.cpp` implements the native joint reprojection solve for camera rotations, camera centers, 3D points, and shared focal lengths.
- It supports Huber, Soft-L1, and trivial losses, parameter masks, fixed/refinable rotations and translations, and bounded focal-length optimization.
- The principal point remains fixed in the current global-engine configuration.
- The solver uses dense Schur for smaller reconstructions and sparse Schur with Jacobi preconditioning for larger reconstructions. It checks solver usability, cost behavior, finite outputs, valid rotations, and positive-depth observations.
- `glomap_bundle_adjuster.cpp` and `GlomapGlobalBundleAdjuster.kt` provide the JNI/Kotlin boundary and validate packed results.
- The global engine schedules bundle-adjustment passes and runs post-retriangulation refinement.

### Engine integration

- `GlobalReconstructionEngine.kt` orchestrates feature extraction, pair verification, view-graph calibration/rotation, track construction and filtering, global positioning, triangulation, bundle adjustment, and post-retriangulation refinement.
- The global path is exposed through `CameraReconstructionEngine` using `ReconstructionMode.GLOBAL`.
- The existing incremental engine remains available through `ReconstructionMode.CUSTOM`; it is not replaced by the global path.
- Selected COLMAP-derived track management, image selection, triangulation, and observation-filtering components remain in the Kotlin tracking packages. Native mobile-specific estimation and solver bridges are kept in the C++ targets.

## Android native dependency configuration

The app CMake target currently:

- Pins Ceres 2.2.0 and builds it for the active Android ABI.
- Uses Eigen 3.4.0 and the EigenSparse backend.
- Disables SuiteSparse, CUDA, LAPACK, and other unused optional backends.
- Pins PoseLib at `7e9f5f53372e43f89655040d4dfc4a00e5ace11c`.
- Builds the `trackstudio_reconstruction` shared library with Ceres and PoseLib.
- Builds the separate camera-estimation and proxy libraries.

The app uses Android NDK 27.0.12077973, CMake 3.22.1, C++17, minSdk 26, and the `arm64-v8a` and `armeabi-v7a` production ABIs.

## Validation already present

- Native numeric tests run in the main Android CI workflow.
- Kotlin release unit tests run in CI.
- Engine instrumentation tests run on an x86_64 Android emulator.
- The release APK is built for the configured production ABIs.
- Instrumentation tests cover the native global solver/JNI path with synthetic scenes, progress callbacks, cost reduction, gauge behavior, and bundle-adjustment constraints.
- A separate Ceres smoke workflow builds the selected sparse backend on the host and cross-builds Ceres plus the smoke executable for both Android ABIs.

The successful Android CI run for commit `755aae1ee1d762592d7c3258aa2acf5a7afb0bac` confirms the current branch's configured build and test workflow passed. It is not a substitute for real-device reconstruction validation or a complete source-by-source differential test against the upstream executables.

## Remaining work

1. Run repeatable end-to-end footage tests on physical Android devices, including low-memory and 32-bit devices; record reconstruction success, camera/landmark counts, reprojection error, memory, and solve time.
2. Expand deterministic differential fixtures for the ported geometry, track, positioning, calibration, and bundle-adjustment stages against the exact pinned upstream revisions.
3. Complete the source-to-upstream revision and licence inventory for every adapted component and transitive dependency. Keep GPL SuiteSparse components disabled.
4. Profile the full global path on representative phone footage and address measured allocation, threading, and memory bottlenecks without changing the selected algorithms.
5. Continue migrating only proven hot-path components where native execution materially helps. Keep the current custom engine as a fallback until physical-device validation is complete.

Do not introduce replacement algorithms. Any missing stage must be sourced from the pinned COLMAP/GLOMAP implementation, with only the Android build and platform integration adapted.
