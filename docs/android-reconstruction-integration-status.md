# Android Reconstruction Integration Status

Status date: 2026-11-12.

## Locked architecture

- Fast and accurate feature pipelines remain separate selectable profiles.
- Learned components are limited to perception, feature extraction and feature matching.
- Camera geometry, pose estimation, triangulation, reconstruction and optimization use proven upstream implementations.
- Preserve source algorithms and behavior. Android-specific work belongs in build configuration, JNI boundaries and narrow platform adapters.
- Remove dependencies only after the selected source target graph proves they are unused.
- Custom and Global engines remain unchanged. AI Engine remains separate.

## Current repository state

- Native geometry now contains the pinned two-view and multi-view triangulation source, robust multi-view track estimator, sampler, inlier support, and projection residual source.
- The diagnostic landmark path invokes the robust estimator instead of selecting a seed pair and applying a custom inlier heuristic.
- The Android compatibility camera adapter currently supports calibrated pinhole intrinsics. Other camera models are not yet connected.
- Existing position and bundle-adjustment solvers remain partial implementations; they are not the complete mapping pipeline.
- Model assets and on-device model validation remain outstanding.
- Current pose sequence still comes from the existing pairwise-motion stage with arbitrary per-step translation scale. Produced landmarks remain diagnostic until camera registration and incremental mapping are integrated.

## Source baseline

- Pinned source revision: `137f21680c79d3f2fffbb1d11deb367162aa6f83`.
- License: BSD-3-Clause. Required notices remain in `UPSTREAM_LICENSES.txt`.
- Original source paths and revision are recorded in `THIRD_PARTY_SOURCE_REVISION.txt`.
- The selected source is isolated from desktop UI, database, and optional compute dependencies through narrow Android compatibility headers.

## Remaining integration

1. Connect actual incremental camera registration and mapping.
2. Connect feature profiles to native feature and match structures.
3. Integrate full local and global bundle adjustment using unchanged source.
4. Add deterministic parity tests against the reference build.
5. Validate intrinsics, pose accuracy, point coordinates, reprojection residuals, and tracking continuity on both Android ABIs and real devices.
6. Measure CPU, peak RAM, latency, and thermal behavior before enabling production defaults.

## Acceptance gates

- Same input produces numerically consistent results between reference and Android builds within documented tolerances.
- Camera rotations, translations, intrinsics, 3D points and reprojection residuals are compared.
- Both Android ABIs build successfully.
- Long-sequence, low-texture, motion-blur and low-light footage is tested.
- CPU usage, peak RAM, latency, thermal throttling and tracking continuity are measured.
- No production claim is made until real-device tests pass.
