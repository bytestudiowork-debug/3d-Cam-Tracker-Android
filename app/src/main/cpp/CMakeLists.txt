cmake_minimum_required(VERSION 3.22.1)
project("trackstudio_native" LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include(FetchContent)

# Ceres 2.2.0 is built for the active Android ABI by the same NDK toolchain
# as TrackStudio. Eigen is header-only; Ceres uses its EigenSparse backend.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MINIGLOG ON CACHE BOOL "" FORCE)
set(GFLAGS OFF CACHE BOOL "" FORCE)
set(SUITESPARSE OFF CACHE BOOL "" FORCE)
set(EIGENSPARSE ON CACHE BOOL "" FORCE)
set(CUDA OFF CACHE BOOL "" FORCE)
set(LAPACK OFF CACHE BOOL "" FORCE)
set(ACCELERATESPARSE OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Populate Eigen's headers without configuring its optional desktop targets.
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()

# Provide the minimal Eigen3 config package required by Ceres.
set(Eigen3_DIR "${CMAKE_BINARY_DIR}/eigen3-config" CACHE PATH "Eigen 3.4 package directory" FORCE)
file(MAKE_DIRECTORY "${Eigen3_DIR}")
file(WRITE "${Eigen3_DIR}/Eigen3Config.cmake"
"set(Eigen3_FOUND TRUE)\n"
"set(Eigen3_VERSION 3.4.0)\n"
"set(Eigen3_INCLUDE_DIRS \"${eigen_SOURCE_DIR}\")\n"
"if(NOT TARGET Eigen3::Eigen)\n"
"  add_library(Eigen3::Eigen INTERFACE IMPORTED)\n"
"  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${eigen_SOURCE_DIR}\")\n"
"endif()\n"
)
file(WRITE "${Eigen3_DIR}/Eigen3ConfigVersion.cmake"
"set(PACKAGE_VERSION 3.4.0)\n"
"if(PACKAGE_FIND_VERSION VERSION_LESS_EQUAL PACKAGE_VERSION)\n"
"  set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
"  if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)\n"
"    set(PACKAGE_VERSION_EXACT TRUE)\n"
"  endif()\n"
"endif()\n"
)

# PoseLib 2.0.5 is pinned by GLOMAP at 7e9f5f53372e43f89655040d4dfc4a00e5ace11c.
set(BUILD_TESTS OFF CACHE BOOL "Do not build PoseLib tests" FORCE)
set(WITH_BENCHMARK OFF CACHE BOOL "Do not build PoseLib benchmarks" FORCE)
set(PYTHON_PACKAGE OFF CACHE BOOL "Do not build PoseLib Python bindings" FORCE)
set(MARCH_NATIVE OFF CACHE BOOL "Use portable Android instruction sets" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build PoseLib statically" FORCE)
FetchContent_Declare(
        PoseLib
        GIT_REPOSITORY https://github.com/PoseLib/PoseLib.git
        GIT_TAG 7e9f5f53372e43f89655040d4dfc4a00e5ace11c
)
FetchContent_MakeAvailable(PoseLib)
# PoseLib adds -Werror unconditionally; keep its warnings visible but non-fatal.
target_compile_options(PoseLib PRIVATE -Wno-error)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build Android dependencies as shared libraries" FORCE)

# Avoid duplicate uninstall target creation between dependencies.
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Do not add Ceres uninstall target" FORCE)

FetchContent_Declare(
        ceres
        GIT_REPOSITORY https://ceres-solver.googlesource.com/ceres-solver
        GIT_TAG 2.2.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ceres)

if(NOT TARGET Ceres::ceres)
    message(FATAL_ERROR "Ceres 2.2.0 did not export Ceres::ceres")
endif()

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

add_library(trackstudio_reconstruction SHARED
        glomap_positioner.cpp
        global_positioning_core.cpp
        glomap_relative_pose_estimator_core.cpp
        glomap_relative_pose_estimator_jni.cpp
        glomap_view_graph_calibrator.cpp
        view_graph_calibration_core.cpp
        glomap_bundle_adjuster.cpp
        global_bundle_adjuster_core.cpp)
target_link_libraries(trackstudio_reconstruction PRIVATE Ceres::ceres PoseLib ${log-lib})
target_compile_options(trackstudio_reconstruction PRIVATE -O3 -fno-fast-math)

add_library(trackstudio_camera SHARED
        absolute_pose_loransac_core.cpp
        absolute_pose_loransac_jni.cpp
        camera_pose_refiner.cpp
        camera_pose_refiner_core.cpp
        epnp_pose_estimator_jni.cpp
        epnp_pose_estimator_core.cpp
        p3p_pose_estimator_core.cpp
        least_absolute_deviations_core.cpp
        native_l1_solver_jni.cpp
        native_sift_extractor_jni.cpp
        sift_feature_core.cpp
        sift_core/sift.c
        sift_core/imopv.c
        sift_core/mathop.c
        sift_core/generic.c
        sift_core/host.c
        sift_core/random.c)
set(SIFT_PORTABLE_C_SOURCES
        sift_core/sift.c
        sift_core/imopv.c
        sift_core/mathop.c
        sift_core/generic.c
        sift_core/host.c
        sift_core/random.c)
set_source_files_properties(${SIFT_PORTABLE_C_SOURCES} PROPERTIES
        COMPILE_DEFINITIONS "VL_DISABLE_SSE2;VL_DISABLE_AVX;VL_DISABLE_OPENMP;VL_DISABLE_THREADS"
        COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(sift_feature_core.cpp PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off")
target_link_libraries(trackstudio_camera PRIVATE Ceres::ceres m)
target_compile_options(trackstudio_camera PRIVATE -O3 -fno-fast-math)

add_library(trackstudio_native SHARED proxy_engine.cpp)
target_include_directories(trackstudio_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(trackstudio_native PRIVATE
        ${log-lib}
        ${jnigraphics-lib}
)

target_compile_options(trackstudio_native PRIVATE
        -O3
        -funroll-loops
)

if(SUITESPARSE OR NOT EIGENSPARSE)
    message(FATAL_ERROR "TrackStudio requires Ceres EigenSparse with SuiteSparse disabled")
endif()
