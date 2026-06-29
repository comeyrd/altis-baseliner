add_library(Gups OBJECT)
target_include_directories(Gups PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(Gups PUBLIC baseliner::baseliner altis-baseliner)
if(HAS_CUDA)
  target_sources(Gups PRIVATE cuda/Gups.cu)
endif()

if(HAS_HIP)
  target_sources(Gups PRIVATE hip/Gups.hip)
endif()

target_sources(benchmark INTERFACE $<TARGET_OBJECTS:Gups>)