# Build

## Local build

Requirements: JDK 17, Android SDK 34, NDK 27.0.12077973, and CMake 3.22.1.

```bash
./gradlew testReleaseUnitTest
./gradlew assembleRelease
```

The release APK is written to:

```
app/build/outputs/apk/release/
```

## GitHub Actions

The workflow at `.github/workflows/build.yml` runs on pushes, pull requests,
and manual dispatch. It runs native numeric tests, tracking unit tests,
Android emulator instrumentation tests, and a release APK build. Successful
branch and manual runs upload the release APK as a workflow artifact.

## Device validation

The standard CI workflow runs the engine test suite on an Android 14 emulator. A separate Android 8.1 (API 27) compatibility workflow exercises the instrumentation suite on an older Android runtime.

For hands-on phone testing, see [the device validation checklist](docs/device-validation.md). The release APK is CI-signed for installation testing; it is not a Play Store release.

## Technical notes

Architecture, solver design, dependency audits, and porting notes are kept in
the `docs/` directory.