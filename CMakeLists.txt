cmake_minimum_required(VERSION 3.15)
project(altis-baseliner VERSION 1.0.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  baseliner
  GIT_REPOSITORY https://github.com/comeyrd/gpu-kernel-baseliner.git
  GIT_TAG        v0.3
)

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
option(COMBINED_BUILD "Build everything into one executable" ON)

set_property(GLOBAL PROPERTY APP_OBJECT_TARGETS "")

add_subdirectory(altis)

if(COMBINED_BUILD)
    get_property(ALL_OBJ_TARGETS GLOBAL PROPERTY APP_OBJECT_TARGETS)
    
    add_executable(altis_exec)
    
    foreach(obj_target IN LISTS ALL_OBJ_TARGETS)
        target_sources(altis_exec PRIVATE $<TARGET_OBJECTS:${obj_target}>)
    endforeach()

    target_link_libraries(altis_exec PRIVATE baseliner::baseliner altis-baseliner)
endif()

