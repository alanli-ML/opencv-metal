if(NOT HAVE_METAL)
  ocv_module_disable(metalimgproc)
endif()

set(the_description "Metal-accelerated Advanced Image Processing")

ocv_add_module(metalimgproc opencv_core opencv_imgproc)

if(HAVE_METAL)
  # Set ARC flags for all Metal .mm files
  set_source_files_properties(src/grabcut.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set_source_files_properties(src/gmm.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set_source_files_properties(src/gmm_kernels.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set_source_files_properties(src/gmm_gpu.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set_source_files_properties(src/graphcut.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  set_source_files_properties(src/util.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

  set_source_files_properties(src/clustering.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

ocv_set_module_sources(
    HEADERS include/opencv2/metalimgproc.hpp
    SOURCES src/precomp.hpp src/gmm_internal.hpp src/gmm_kernels.hpp 
            src/grabcut.mm src/gmm.mm src/gmm_kernels.mm src/gmm_gpu.mm 
            src/util.mm src/clustering.mm
            src/graphcut.mm src/graphcut_internal.hpp
)

ocv_module_include_directories()
ocv_create_module()

if(HAVE_METAL)
  # Link Metal frameworks after module creation
  find_library(METAL_LIBRARY Metal)
  find_library(METALPERFORMANCESHADERS_LIBRARY MetalPerformanceShaders)
  find_library(FOUNDATION_LIBRARY Foundation)
  find_library(COREGRAPHICS_LIBRARY CoreGraphics)
  
  target_link_libraries(${the_module} PRIVATE ${METAL_LIBRARY} ${METALPERFORMANCESHADERS_LIBRARY} ${FOUNDATION_LIBRARY} ${COREGRAPHICS_LIBRARY})
endif()

option(ENABLE_FULLGPU_GRABCUT "Enable full GPU GraphCut tests" ON)

ocv_add_accuracy_tests()

# Conditional Metal Graph Cut Support - Phase 3 Implementation
target_compile_definitions(${the_module} PRIVATE USE_METAL_GRAPHCUT=1)
if(BUILD_TESTS AND TARGET opencv_test_metalimgproc)
  target_compile_definitions(opencv_test_metalimgproc PRIVATE USE_METAL_GRAPHCUT=1)
endif()

if(HAVE_METAL)
  ocv_add_perf_tests()
endif() 

include_directories(${OpenCV_SOURCE_DIR}/modules/imgproc/include) 