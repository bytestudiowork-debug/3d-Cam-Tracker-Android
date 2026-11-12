cmake_minimum_required(VERSION 3.22)
project(trackstudio_ceres_sparse_smoke LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Ceres 2.2 REQUIRED CONFIG)
add_executable(ceres_sparse_smoke main.cpp)
target_link_libraries(ceres_sparse_smoke PRIVATE Ceres::ceres)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
set(RECONSTRUCTION_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../app/src/main/cpp)
add_executable(triangulation_smoke triangulation_smoke.cpp
 ${RECONSTRUCTION_SOURCE_DIR}/upstream_geometry.cpp
 ${RECONSTRUCTION_SOURCE_DIR}/upstream_inlier_support.cpp
 ${RECONSTRUCTION_SOURCE_DIR}/colmap/estimators/triangulation.cc
 ${RECONSTRUCTION_SOURCE_DIR}/colmap/optim/combination_sampler.cc
 ${RECONSTRUCTION_SOURCE_DIR}/colmap/scene/projection.cc)
target_include_directories(triangulation_smoke PRIVATE ${RECONSTRUCTION_SOURCE_DIR})
target_link_libraries(triangulation_smoke PRIVATE Eigen3::Eigen)
