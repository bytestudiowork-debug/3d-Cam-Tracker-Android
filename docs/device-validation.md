# Device validation checklist

## Test package

Use the `trackstudio-release-apk` artifact attached to the successful **Build optimized release APK** run on the `android-tracking-engine` branch.

The APK is signed with the CI debug key so it can be installed for testing. It is not a production signing configuration.

## Before testing

- Use an Android device running Android 8.0 (API 26) or newer.
- Keep at least 2 GB of free storage for source footage, project data, and generated output.
- Start with a short, well-lit clip (roughly 10–20 seconds) with visible scene texture and clear camera movement.
- Avoid a clip with rapid motion blur, a moving subject filling the frame, or a featureless wall for the first test.
- Keep the original clip unchanged so results can be compared.

## Install

1. Download the APK artifact from the successful GitHub Actions run.
2. Transfer the APK to the phone and install it. Android may ask you to allow installation from that file source.
3. Open TrackStudio and create a project.
4. Import the test clip using the system file picker.

## First-pass test

1. Confirm the imported clip opens and displays a frame.
2. Play, pause, and scrub to the beginning, middle, and end.
3. Reopen the project and confirm the source clip is still linked.
4. Run tracking on the short clip using the existing CUSTOM mode first.
5. Confirm progress advances and the solve finishes or reports a clear error; note any stage where progress stops.
6. Inspect the camera path and reconstructed points for obvious jumps or collapse.
7. Export the solve report if the app exposes that action, and preserve it with the source clip for review.

Do not judge reconstruction quality from a single clip. A useful test set includes a slow pan, a forward/backward camera move, and a clip with a modest orbit around a textured stationary scene.

## Record results

For each run, record:

- Device model and Android version
- App version shown by the installed build
- Video resolution, frame rate, duration, and file size
- Selected reconstruction mode
- Whether playback and scrubbing worked
- Whether tracking completed, elapsed time, and the last visible progress phase if it stalled
- Whether the camera path and point cloud looked stable
- Any crash, error message, or exported report

If the app crashes or tracking stalls, capture a bug report or Android logcat around the failure and attach the exported solve report and a short sample clip where possible.

## What CI does and does not prove

The API 27 compatibility workflow checks Android 8.1 runtime behavior using an x86_64 emulator. The regular workflow checks Android 14 and builds the release APK for the configured ARM phone ABIs.

Neither emulator run proves performance or camera/decoder behavior on a particular physical phone. Physical-device validation remains required, especially for native library loading on ARM, sustained memory use, thermal throttling, and reconstruction quality.
