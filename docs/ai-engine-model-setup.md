# AI Engine model assets

The AI Engine uses two generated ONNX assets stored in the APK:

- `app/src/main/assets/ai/extractor.onnx`
- `app/src/main/assets/ai/matcher.onnx`

The binaries are not committed to source control. On the `feature/ai-engine` branch, the Android build workflow downloads the pinned, checksum-verified source weights, exports both assets, validates their CPU inference and dynamic input dimensions, then packages them into the release APK.

The Android runtime uses ONNX Runtime on CPU with bounded thread counts. The model files are copied from APK assets to the app's private files directory before the inference sessions are opened. Model assets are refreshed after an app update.

If asset export or validation fails, the build fails rather than packaging placeholder models. The AI Engine must not be considered production-ready until real-device tests confirm extraction quality, matching stability, memory use, and processing speed across the supported Android device range.

The existing Custom and Global modes remain available independently of these assets.
