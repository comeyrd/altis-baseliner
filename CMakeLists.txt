cmake_minimum_required(VERSION 3.15)
project(altis-baseliner VERSION 1.0.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#TEMPORARY FIX AS THROUGH PRESETS IT DOES NOT WORK
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(FOUND_DEVICE_COMPILER OFF)
set(BLAS_LIBRARIES "" )
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    enable_language(CUDA)
    find_package(CUDAToolkit 12 REQUIRED)
    set(HAS_CUDA ON)
    set(FOUND_DEVICE_COMPILER ON)
endif()

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(HAS_HIP ON)
    set(FOUND_DEVICE_COMPILER ON)
endif()

if(NOT FOUND_DEVICE_COMPILER)
    message(FATAL_ERROR "Requires at least one device device compiler")
endif()

include(FetchContent)
FetchContent_Declare(
  baseliner
  GIT_REPOSITORY https://github.com/comeyrd/gpu-kernel-baseliner.git
  GIT_TAG        v1.0
)
FetchContent_MakeAvailable(baseliner)
include(${baseliner_SOURCE_DIR}/cmake/utils.cmake)

#version interface
add_library(altis-baseliner INTERFACE)

baseliner_enable_git_version(altis-baseliner)

add_library(benchmark INTERFACE)
target_link_libraries(benchmark INTERFACE baseliner::baseliner)
# Add subdirectories
add_subdirectory(altis)
set(DUMMY_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/dummy_main.cpp")
file(WRITE ${DUMMY_SOURCE} " ")
add_executable(altis_exec ${DUMMY_SOURCE})
target_link_libraries(altis_exec PRIVATE benchmark)
