cmake_minimum_required(VERSION 3.15)

add_executable(pathfinderLib PathFinder.cu main.cu)
target_include_directories(pathfinderLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pathfinderLib PRIVATE baseliner::baseliner altis-baseliner)
message("here")