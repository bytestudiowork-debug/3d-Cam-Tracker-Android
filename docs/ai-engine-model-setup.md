# AI Engine model asset

The AI Engine loads its optional TensorFlow Lite asset from:

`app/src/main/assets/ai/feature_model.tflite`

The asset is not committed to this repository. This keeps model binaries out of source control and allows the release asset to be updated independently from the Android code.

## Configure the build

1. Publish the approved model archive as a stable HTTPS ZIP containing exactly one `.tflite` file.
2. Add a repository Actions variable named `AI_FEATURE_MODEL_ARCHIVE_URL` with that archive URL.
3. Push to `feature/ai-engine`, or run the Android build workflow manually.

The workflow downloads the archive, extracts the model into the app assets directory, runs the unit tests, and builds the release APK. The download script can also be run locally:

```bash
AI_FEATURE_MODEL_ARCHIVE_URL="https://your-approved-host/model-archive.zip" \
  bash scripts/fetch_ai_feature_model.sh
./gradlew testReleaseUnitTest assembleRelease
```

The archive must contain exactly one TensorFlow Lite model. The runtime validates input and output tensor shapes and data types before inference. If the asset is absent, the AI Engine remains unavailable and the existing Custom and Global engines continue to work.

Do not enable the AI Engine for production footage until device inference and same-footage comparisons against the Global engine have passed.
