cmake_minimum_required(VERSION 3.16)
project(tracking_solver_numeric_tests LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Ceres REQUIRED)

enable_testing()

add_executable(tracking_solver_numeric_tests
        tracking_solver_numeric_tests.cpp
        ../camera_pose_refiner_core.cpp
        ../epnp_pose_estimator_core.cpp
        ../global_positioning_core.cpp
        ../global_bundle_adjuster_core.cpp
        ../view_graph_calibration_core.cpp
        ../least_absolute_deviations_core.cpp)
target_include_directories(tracking_solver_numeric_tests PRIVATE ..)
target_link_libraries(tracking_solver_numeric_tests PRIVATE Ceres::ceres)
target_compile_options(tracking_solver_numeric_tests PRIVATE -O2 -fno-fast-math)
add_test(NAME tracking_solver_numeric_tests COMMAND tracking_solver_numeric_tests)

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
