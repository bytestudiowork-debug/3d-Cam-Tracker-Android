# COLMAP + GLOMAP Android integration audit

Audit date: 2026-11-05.

## Source pins and relationship

- GLOMAP source: `colmap/glomap@99806d0869f802fad218516a2e027793e7ca687d`.
- COLMAP revision fetched by that GLOMAP source's `thirdparty/CMakeLists.txt`:
  `colmap/colmap@b6b7b54eca6078070f73a3f0a084f79c629a6f10`
  (COLMAP 3.14.0 development snapshot, 2025-11-20).
- Separate later COLMAP reference examined by this project:
  `colmap/colmap@e6644e744488a13f3aba82958500db9066341205`
  (4.3.0 development snapshot).

Do not combine headers from the later COLMAP revision with libraries built
from the GLOMAP-matched COLMAP revision. First get the matching pair building;
then assess whether the later COLMAP source should replace or supplement it.

## Confirmed blockers and decisions

1. **GPL dependency request in upstream GLOMAP build.** The pinned GLOMAP
   `cmake/FindDependencies.cmake` requires
   `find_package(Ceres REQUIRED COMPONENTS SuiteSparse)`; its vcpkg manifest
   requests Ceres' `suitesparse` feature and SuiteSparse itself. Ceres'
   official documentation states that the required SuiteSparse CHOLMOD
   supernodal and SPQR components are GPL/commercial. Do not enable this
   dependency for TrackStudio without a commercial licence.

2. **No-GPL sparse backend candidate.** Ceres supports Eigen sparse
   Cholesky as a backend for `SPARSE_SCHUR`. The proposed build is Ceres
   `EigenSparse` with GLOMAP's package requirement changed to
   `EigenSparse`; remove SuiteSparse from the GLOMAP dependency manifest.
   GLOMAP's track-based path sets `SPARSE_SCHUR` and also assigns
   `CLUSTER_TRIDIAGONAL`, but Ceres defines that preconditioner for
   `ITERATIVE_SCHUR` only. GLOMAP explicitly says not to use iterative
   solvers. Preserve the field assignment for source parity, but test the
   actual `SPARSE_SCHUR` + EigenSparse path; do not claim the cluster
   preconditioner is being applied.

3. **CMake version mismatch.** The app uses Android CMake 3.22.1. The pinned
   GLOMAP top-level file requires CMake 3.28. Ceres 2.3 requires CMake 3.22+.
   A build-system decision is needed: install Android CMake 3.28 for the
   native build, or create a narrowly scoped Android subproject build that
   avoids GLOMAP's newer top-level configuration. Keep the app's current
   version unchanged until the chosen route builds in CI.

4. **COLMAP is not a single small dependency.** GLOMAP links COLMAP, Eigen,
   Ceres, PoseLib, OpenMP and Boost. COLMAP's desktop build adds many optional
   dependencies and targets. Android should compile only the COLMAP core
   targets needed by GLOMAP; GUI, OpenGL, CUDA, MVS, download support and
   unused desktop modules should be disabled where the dependency graph
   permits. Determine the actual transitive target closure from the pinned
   sources rather than guessing.

5. **Licence boundary.** The pinned COLMAP and GLOMAP projects are BSD-3-Clause.
   Ceres is BSD-3-Clause. Their dependencies have independent licences and
   must be inventoried individually. BSD upstream licences do not make a
   build with GPL dependencies proprietary-safe by themselves. Retain required
   copyright and licence notices in the app's third-party notices.

## Build/validation gate

Before adding upstream COLMAP/GLOMAP source to the production app:
- record exact commits for COLMAP, GLOMAP, Ceres, Eigen, Abseil, PoseLib and
  any additional dependencies;
- generate a dependency/licence inventory and fail CI if a GPL component is
  included;
- build each dependency for both `arm64-v8a` and `armeabi-v7a` using the
  app's NDK/API/STL settings;
- run the Ceres `SPARSE_SCHUR` + EigenSparse smoke test on both ABIs;
- build the selected COLMAP core and GLOMAP sources;
- compare a deterministic fixture against the pinned upstream implementation.

## Source files inspected

- GLOMAP root `CMakeLists.txt`, `cmake/FindDependencies.cmake`,
  `thirdparty/CMakeLists.txt`, `glomap/CMakeLists.txt`, and `vcpkg.json`.
- GLOMAP global-positioning and bundle-adjustment solver option setup.
- COLMAP root `CMakeLists.txt` at both recorded COLMAP revisions.
- COLMAP controller target graph at the later reference revision.
- Ceres official installation, Android build, component and solver guidance.

## Implementation status update (2026-11-11)

The app now builds Eigen 3.4.0 and Ceres 2.2.0 through its Android CMake
configuration. SuiteSparse, CUDA and LAPACK are disabled; Ceres uses the
Eigen sparse backend. The app builds a JNI target named
`trackstudio_glomap` from two focused native solver files.

This is a partial Android implementation, not a build of the upstream
COLMAP/GLOMAP libraries. The positioner implements the ONLY_POINTS
direction-constraint solve. The bundle-adjustment JNI path currently holds
camera rotations fixed and jointly refines camera centres and 3D points.
The Kotlin global reconstruction path still obtains features through its
existing OpenCV implementation. The separate AI module is not connected to
that geometry path yet, and it has no packaged model weights.

The successful release workflow on 2026-11-11 built the Android app with
the native target. A dedicated Ceres sparse-solver smoke workflow has now
been enabled for `feature/ai-engine`; its two-ABI validation is tracked
separately. Upstream COLMAP/GLOMAP source integration and deterministic
parity testing remain open gates.
