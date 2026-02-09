cmake_minimum_required(VERSION 3.15)

add_library(sortLib OBJECT SortKernel.cu main.cu)
target_include_directories(sortLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(GLOBAL APPEND PROPERTY APP_OBJECT_TARGETS sortLib)
target_link_libraries(sortLib PUBLIC baseliner::baseliner altis-baseliner)


if(NOT COMBINED_BUILD)
    add_executable(sortLib $<TARGET_OBJECTS:sortLib>)
    target_link_libraries(sortLib PRIVATE baseliner::baseliner altis-baseliner)
endif()