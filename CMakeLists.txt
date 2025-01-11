cmake_minimum_required (VERSION 3.14) # CMake minimum version required. It is always good to ensure!
project (PhaseFieldBoram) # name of your project
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(pf-boram-jr main.cpp)
target_link_libraries(pf-boram-jr PUBLIC Eigen3::Eigen)