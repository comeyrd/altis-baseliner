cmake_minimum_required(VERSION 3.15)

add_executable(sortLib SortKernel.cu main.cu)
target_include_directories(sortLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sortLib PRIVATE baseliner::baseliner altis-baseliner)
message("here")