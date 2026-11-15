cmake_minimum_required(VERSION 3.16)
project(tracking_solver_numeric_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Ceres REQUIRED)

enable_testing()

add_executable(tracking_solver_numeric_tests
        tracking_solver_numeric_tests.cpp
        ../camera_pose_refiner_core.cpp
        ../global_bundle_adjuster_core.cpp
        ../least_absolute_deviations_core.cpp)
target_include_directories(tracking_solver_numeric_tests PRIVATE ..)
target_link_libraries(tracking_solver_numeric_tests PRIVATE Ceres::ceres)
target_compile_options(tracking_solver_numeric_tests PRIVATE -O2 -fno-fast-math)
add_test(NAME tracking_solver_numeric_tests COMMAND tracking_solver_numeric_tests)
