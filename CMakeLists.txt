cmake_minimum_required(VERSION 3.15)
project(altis-baseliner VERSION 1.0.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  baseliner
  GIT_REPOSITORY https://github.com/comeyrd/gpu-kernel-baseliner.git
  GIT_TAG        develop 
)
#  GIT_TAG        v0.2.1 

FetchContent_MakeAvailable(baseliner)
include(${baseliner_SOURCE_DIR}/cmake/utils.cmake)


#TEMPORARY FIX AS THROUGH PRESETS IT DOES NOT WORK
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Create a "version" interface
add_library(altis-baseliner INTERFACE)

# Run your function on this interface
baseliner_enable_git_version(altis-baseliner)

add_subdirectory(altis)
