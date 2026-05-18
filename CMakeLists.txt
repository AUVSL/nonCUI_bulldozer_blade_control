cmake_minimum_required(VERSION 3.15)
project(bulldozer_blade_control CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build with optimizations by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(bulldozer main.cpp)
target_link_libraries(bulldozer PRIVATE Eigen3::Eigen)

# Suppress Eigen deprecation warnings on MSVC
if(MSVC)
  target_compile_options(bulldozer PRIVATE /W3 /wd4996)
else()
  target_compile_options(bulldozer PRIVATE -O3 -Wall -Wextra)
endif()
