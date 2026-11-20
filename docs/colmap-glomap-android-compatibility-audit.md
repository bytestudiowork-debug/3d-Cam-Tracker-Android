# COLMAP and GLOMAP Android integration audit

Audit date: 2026-11-20.

## Source pins and relationship

- GLOMAP source: `colmap/glomap@99806d0869f802fad218516a2e027793e7ca687d`.
- COLMAP revision fetched by that GLOMAP source's `thirdparty/CMakeLists.txt`: `colmap/colmap@b6b7b54eca6078070f73a3f0a084f79c629a6f10`.
- Separate later COLMAP reference examined by this project: `colmap/colmap@e6644e744488a13f3aba82958500db9066341205`.
- PoseLib: `7e9f5f53372e43f89655040d4dfc4a00e5ace11c`.

The GLOMAP-matched COLMAP revision is the compatibility baseline. Do not combine headers from the later COLMAP reference with libraries built from the matching revision.

## Current implementation boundary

TrackStudio is not building either complete desktop application on Android. It ports selected algorithms and data-management components into the existing Kotlin/C++ mobile architecture.

The current Android native CMake configuration builds:

- `trackstudio_reconstruction`: Ceres-backed global positioning, relative-pose estimation through the pinned PoseLib dependency, view-graph calibration, and global bundle adjustment.
- `trackstudio_camera`: native camera-estimation and feature-extraction components.
- `trackstudio_native`: the proxy engine.

The Kotlin global engine orchestrates the full mobile reconstruction flow and uses selected COLMAP/GLOMAP-derived track, image-selection, filtering, triangulation, view-graph, and reconstruction stages. The custom incremental engine remains available separately. This is an adapted subset, not a drop-in Android build of the full COLMAP/GLOMAP source trees.

## Dependency and licence decisions

The app CMake configuration pins Ceres 2.2.0, Eigen 3.4.0, and PoseLib at the revision above. Ceres is configured with EigenSparse enabled and SuiteSparse disabled. CUDA, LAPACK, and unused optional backends are disabled for the Android build.

The pinned upstream GLOMAP desktop build requests SuiteSparse. Its CHOLMOD supernodal and SPQR components carry GPL/commercial licensing conditions. TrackStudio must not enable those components for its proprietary distribution without the appropriate commercial licence. The EigenSparse configuration keeps the selected `SPARSE_SCHUR` solver path without adding SuiteSparse.

Retain all upstream copyright and licence notices for copied or adapted files and all third-party dependencies. BSD licensing of an upstream project does not change the independent licence of its transitive dependencies.

## Build and test evidence

The main Android CI workflow runs native numeric tests, Kotlin release unit tests, x86_64 emulator instrumentation tests, and a release APK build. The production ABI filters are `arm64-v8a` and `armeabi-v7a`.

The dedicated Ceres smoke workflow builds a host solver and cross-builds Ceres plus a real sparse-Schur smoke executable for both Android ABIs. Cross-compilation is not the same as running the binary on a physical Android device.

Commit `755aae1ee1d762592d7c3258aa2acf5a7afb0bac` passed the main Android CI workflow on 2026-11-20. This establishes that the current configured CI checks passed; it does not establish full upstream numerical equivalence or physical-device reliability.

## Required release gates

1. Keep the GLOMAP-matched COLMAP revision and all dependency pins explicit.
2. Maintain a complete transitive dependency/licence inventory and reject prohibited GPL components in CI.
3. Keep both production ABIs building with the app's NDK, API level, and C++ runtime.
4. Add deterministic differential fixtures for the adapted estimation, track, positioning, and bundle-adjustment stages.
5. Validate complete reconstruction on physical Android devices with representative footage; capture accuracy, runtime, and memory metrics.
6. Preserve the current custom engine as a fallback until the global path passes device validation.

Only port missing functionality from the pinned upstream implementations. Do not substitute unrelated algorithms to make an incomplete stage appear finished.
