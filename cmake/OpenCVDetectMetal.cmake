#
# Detects Apple Metal framework
#

if(NOT APPLE)
  message(STATUS "Metal: Not an Apple platform, disabling.")
  return()
endif()

find_library(Metal_LIBRARY Metal)
find_library(MetalPerformanceShaders_LIBRARY MetalPerformanceShaders)
find_library(CoreGraphics_LIBRARY CoreGraphics)

if(Metal_LIBRARY AND MetalPerformanceShaders_LIBRARY AND CoreGraphics_LIBRARY)
  set(HAVE_METAL 1 CACHE INTERNAL "Metal support")
  set(METAL_LIBRARIES ${Metal_LIBRARY} ${MetalPerformanceShaders_LIBRARY} ${CoreGraphics_LIBRARY})
  list(APPEND OPENCV_LINKER_LIBS ${METAL_LIBRARIES})
  message(STATUS "Metal: YES")
  message(STATUS "  Metal library: ${Metal_LIBRARY}")
  message(STATUS "  MetalPerformanceShaders library: ${MetalPerformanceShaders_LIBRARY}")
  message(STATUS "  CoreGraphics library: ${CoreGraphics_LIBRARY}")
else()
  set(HAVE_METAL 0 CACHE INTERNAL "Metal support")
  message(STATUS "Metal: NO")
endif()