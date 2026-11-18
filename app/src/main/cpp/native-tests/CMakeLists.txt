cmake_minimum_required(VERSION 3.16)
project(tracking_solver_numeric_tests LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

enable_testing()

include(FetchContent)
set(BUILD_TESTS OFF CACHE BOOL "Do not build PoseLib tests" FORCE)
set(WITH_BENCHMARK OFF CACHE BOOL "Do not build PoseLib benchmarks" FORCE)
set(PYTHON_PACKAGE OFF CACHE BOOL "Do not build PoseLib Python bindings" FORCE)
set(MARCH_NATIVE OFF CACHE BOOL "Use portable instruction sets" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build PoseLib statically" FORCE)
FetchContent_Declare(
        PoseLib
        GIT_REPOSITORY https://github.com/PoseLib/PoseLib.git
        GIT_TAG 7e9f5f53372e43f89655040d4dfc4a00e5ace11c
)
FetchContent_MakeAvailable(PoseLib)
target_compile_options(PoseLib PRIVATE -Wno-error)

add_executable(tracking_solver_numeric_tests
        tracking_solver_numeric_tests.cpp
        ../absolute_pose_loransac_core.cpp
        ../colmap_p4pf_absolute_pose_core.cpp
        ../camera_pose_refiner_core.cpp
        ../epnp_pose_estimator_core.cpp
        ../p3p_pose_estimator_core.cpp
        ../glomap_relative_pose_estimator_core.cpp
        ../global_positioning_core.cpp
        ../global_bundle_adjuster_core.cpp
        ../view_graph_calibration_core.cpp
        ../least_absolute_deviations_core.cpp)
target_include_directories(tracking_solver_numeric_tests PRIVATE ..)
target_link_libraries(tracking_solver_numeric_tests PRIVATE Ceres::ceres PoseLib)
target_compile_options(tracking_solver_numeric_tests PRIVATE -O2 -fno-fast-math)
add_test(NAME tracking_solver_numeric_tests COMMAND tracking_solver_numeric_tests)

add_executable(colmap_initial_pair_differential_tests
        colmap_initial_pair_differential_tests.cpp
        colmap_initial_pair_differential.cpp
        colmap_five_point_differential.cpp
        fundamental_matrix_solver_differential.cpp
        ../two_view_epipolar_estimator_core.cpp
        ../two_view_epipolar_geometry_utils.cpp
        ../two_view_essential_five_point.cpp
        ../two_view_fundamental_solvers.cpp)
target_link_libraries(colmap_initial_pair_differential_tests PRIVATE Eigen3::Eigen)
target_include_directories(colmap_initial_pair_differential_tests PRIVATE ..)
target_compile_options(colmap_initial_pair_differential_tests PRIVATE -O2 -fno-fast-math)
add_test(NAME colmap_initial_pair_differential_tests
        COMMAND colmap_initial_pair_differential_tests)

add_executable(homography_differential_tests
        homography_differential_tests.cpp
        homography_differential.cpp
        colmap_initial_pair_differential.cpp)
target_link_libraries(homography_differential_tests PRIVATE Eigen3::Eigen)
target_include_directories(homography_differential_tests PRIVATE ..)
target_compile_options(homography_differential_tests PRIVATE -O2 -fno-fast-math)
add_test(NAME homography_differential_tests COMMAND homography_differential_tests)

add_executable(two_view_dispatch_differential_tests
        two_view_dispatch_differential_tests.cpp
        two_view_dispatch_differential.cpp
        homography_differential.cpp
        colmap_initial_pair_differential.cpp
        colmap_five_point_differential.cpp
        fundamental_matrix_solver_differential.cpp
        ../two_view_epipolar_estimator_core.cpp
        ../two_view_epipolar_geometry_utils.cpp
        ../two_view_essential_five_point.cpp
        ../two_view_fundamental_solvers.cpp)
target_link_libraries(two_view_dispatch_differential_tests PRIVATE Eigen3::Eigen)
target_include_directories(two_view_dispatch_differential_tests PRIVATE ..)
target_compile_options(two_view_dispatch_differential_tests PRIVATE -O2 -fno-fast-math)
add_test(NAME two_view_dispatch_differential_tests
        COMMAND two_view_dispatch_differential_tests)

add_executable(fundamental_matrix_differential_tests
        fundamental_matrix_differential_tests.cpp
        fundamental_matrix_solver_differential.cpp)
target_link_libraries(fundamental_matrix_differential_tests PRIVATE Eigen3::Eigen)
target_compile_options(fundamental_matrix_differential_tests PRIVATE
        -O2 -fno-fast-math)
add_test(NAME fundamental_matrix_differential_tests
        COMMAND fundamental_matrix_differential_tests)

set(SIFT_PORTABLE_C_SOURCES
        ../sift_core/sift.c
        ../sift_core/imopv.c
        ../sift_core/mathop.c
        ../sift_core/generic.c
        ../sift_core/host.c
        ../sift_core/random.c)
add_executable(camera_sift_reference_tests
        camera_sift_reference_tests.cpp
        ../sift_feature_core.cpp
        ${SIFT_PORTABLE_C_SOURCES})
target_include_directories(camera_sift_reference_tests PRIVATE
        ..)
set_source_files_properties(${SIFT_PORTABLE_C_SOURCES} PROPERTIES
        COMPILE_DEFINITIONS "VL_DISABLE_SSE2;VL_DISABLE_AVX;VL_DISABLE_OPENMP;VL_DISABLE_THREADS"
        COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(../sift_feature_core.cpp PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off")
target_link_libraries(camera_sift_reference_tests PRIVATE m)
target_compile_options(camera_sift_reference_tests PRIVATE -O2 -fno-fast-math -ffp-contract=off)
add_test(NAME camera_sift_reference_tests COMMAND camera_sift_reference_tests)
