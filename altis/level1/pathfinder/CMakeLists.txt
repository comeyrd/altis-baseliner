cmake_minimum_required(VERSION 3.15)

add_library(pathfinderLib OBJECT PathFinder.cu main.cu)
target_include_directories(pathfinderLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS pathfinderLib)
target_link_libraries(pathfinderLib PUBLIC baseliner::baseliner altis-baseliner)

if(NOT COMBINED_BUILD)
    add_executable(pathfinderLib $<TARGET_OBJECTS:pathfinderLib>)
    target_link_libraries(pathfinderLib PRIVATE baseliner::baseliner altis-baseliner)
endif()