# Keep JNI-facing data classes and the native bridge intact.
-keep class com.trackstudio.app.data.native.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}

# C++ resolves this Kotlin callback by its exact JVM class and method name.
-keep interface com.trackstudio.app.tracking.glomap.GlomapSolverProgressListener {
    void onIteration(int, int);
}
-keep class com.trackstudio.app.tracking.glomap.GlomapNativeSolver { *; }
