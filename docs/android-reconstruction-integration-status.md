# Android Reconstruction Integration Status

Status date: 2026-11-11.

## Locked architecture

- The fast and accurate feature pipelines are separate selectable profiles.
- Learned components are limited to perception, feature extraction and feature matching.
- Camera geometry, pose estimation, triangulation, reconstruction and optimization use proven upstream implementations.
- Preserve upstream algorithm source and behavior. Android-specific work belongs in build configuration, JNI boundaries and narrow platform adapters.
- Remove a dependency only after the selected source target graph proves it is unused.
- The Custom and Global engines remain unchanged. The AI Engine remains a separate selectable engine.
- Do not add experimental reconstruction algorithms or redesign established solver mathematics.

## Current repository state

The current Android native implementation is a partial fallback, not a complete reconstruction pipeline.

- The position solver is a focused direction-constraint implementation.
- The bundle-adjustment implementation optimizes camera centres and 3D points while holding camera rotations fixed.
- The current feature adapter can pass learned observations into the existing geometry path, but actual model assets and device validation are still outstanding.
- Existing native code must not be presented as a direct upstream implementation.

## Upstream source audit

The current upstream source tree exposes separate global and incremental reconstruction pipelines.

- The global pipeline estimates camera rotations over the view graph, filters inconsistent edges, establishes tracks and performs global positioning and refinement. It depends on a sufficiently connected, reliable match graph and good camera focal-length priors. It can be faster on large well-connected datasets but is more sensitive to outliers.
- The incremental pipeline registers cameras progressively, triangulates observations, performs local and global bundle adjustment, and supports iterative refinement. It is the more robust, established candidate for an ordered video sequence.
- The final pipeline choice must be based on the app's actual video footage and deterministic comparisons, not assumptions. The incremental path is the first candidate for sequential VFX footage; the global path remains a candidate when matching-graph quality and calibration are strong.

### Source target dependencies observed

The current upstream build separates the relevant code into core target groups:

- Reconstruction pipeline code depends on scene data, estimators and utility code.
- Estimators depend on geometry, math, feature types, sensors, scene structures, optimization helpers, cost functions, solver implementations, Eigen, Ceres and PoseLib.
- Scene storage depends on sensor, feature, geometry and utility types, Eigen and SQLite.
- Optimization currently links a sparse matrix backend and may use OpenMP.
- The desktop build exposes optional GUI, OpenGL, CUDA, multi-view stereo, download, ONNX, CGAL and other features.

This is a dependency map, not permission to remove libraries. In particular, the sparse backend and any libraries exposed through public headers must remain until the exact Android target closure proves they can be excluded without changing upstream algorithms or ABI layout.

## Required next steps

1. Pin one exact upstream revision and record its licence and dependency revisions.
2. Compare the global and incremental paths on the same representative video sequences.
3. Select the reconstruction path using tracking continuity, recovered camera poses, reprojection error, completeness, CPU time and memory.
4. Build the selected upstream source targets for arm64-v8a and armeabi-v7a.
5. Keep upstream algorithm files unchanged. Make only narrowly scoped Android build and platform integration changes.
6. Add deterministic parity fixtures that run against both the upstream reference build and the Android build.
7. Verify dependency licences before shipping and retain all required third-party notices.
8. Keep the existing Custom and Global engines available until the new engine passes device validation.

## Acceptance gates

- Same input produces numerically consistent results between the reference and Android builds within documented tolerances.
- Camera rotations, translations, intrinsics, 3D points and reprojection residuals are compared.
- Both Android ABIs build successfully.
- Long-sequence, low-texture, motion-blur and low-light footage is tested.
- CPU usage, peak RAM, latency, thermal throttling and tracking continuity are measured.
- No production claim is made until real-device tests pass.
