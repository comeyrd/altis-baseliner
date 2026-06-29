add_library(pathFinder OBJECT)
target_include_directories(pathFinder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pathFinder PUBLIC baseliner::baseliner altis-baseliner)
if(HAS_CUDA)
  target_sources(pathFinder PRIVATE cuda/PathFinder.cu)
endif()

if(HAS_HIP)
  target_sources(pathFinder PRIVATE hip/PathFinder.hip)
endif()

target_sources(benchmark INTERFACE $<TARGET_OBJECTS:pathFinder>)