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
- The first native geometry slice now includes the upstream two-view DLT and multi-view bearing-DLT triangulation algorithms. The incremental diagnostic triangulator seeds from a high-parallax pair, selects reprojection-consistent observations, and solves from up to 32 views. It is not yet the full robust multi-view RANSAC estimator or production incremental mapper.
- Existing native code must not be presented as a direct upstream implementation.

## Upstream source baseline

- Initial source baseline: release 4.2.1, tag commit `bd1fcf6`.
- Preserve the original upstream source files and their algorithm behavior.
- The full source revision and all transitive dependency revisions must be recorded in the third-party inventory before the production port begins.

### Reconstruction path decision

Use the incremental reconstruction pipeline as the primary path for ordered video footage.

Reasons:
- It registers cameras progressively, which fits ordered frame sequences.
- It triangulates observations and performs local as well as global bundle adjustment.
- It is the more robust, established option when matches contain outliers or camera calibration is imperfect.
- The global alternative can be faster for large, well-connected match graphs, but is more sensitive to outliers and depends on reliable focal-length priors.

The global path is not part of the initial production integration. Reconsider it only if repeatable tests on the app's own footage show a material advantage without sacrificing camera accuracy or tracking continuity.

### Source target dependencies observed

The selected reconstruction code is organized into core target groups:

- Reconstruction pipeline code depends on scene data, estimators and utility code.
- Estimators depend on geometry, math, feature types, sensors, scene structures, optimization helpers, cost functions, solver implementations, Eigen, Ceres and PoseLib.
- Scene storage depends on sensor, feature, geometry and utility types, Eigen and SQLite.
- Optimization currently links a sparse matrix backend and may use OpenMP.
- The desktop build exposes optional GUI, OpenGL, CUDA, multi-view stereo, download, ONNX, CGAL and other features.

This is a dependency map, not permission to remove libraries. In particular, the sparse backend and any libraries exposed through public headers must remain until the exact Android target closure proves they can be excluded without changing upstream algorithms or ABI layout.

## Required implementation sequence

1. Pin the full source commit and record its licence and dependency revisions.
2. Build the selected core source targets for arm64-v8a and armeabi-v7a.
3. Connect both feature profiles to the same upstream feature/match data contract.
4. Feed the resulting database/scene graph into the unchanged incremental reconstruction pipeline.
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
