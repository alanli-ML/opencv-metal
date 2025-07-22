if(NOT HAVE_METAL)
  ocv_module_disable(metalfilters)
endif()

set(the_description "Metal-accelerated Image Filtering")

ocv_add_module(metalfilters opencv_core opencv_imgproc)

if(HAVE_METAL)
  set_source_files_properties(src/filtering.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
endif()

ocv_set_module_sources(
    HEADERS include/opencv2/metalfilters.hpp
    SOURCES src/precomp.hpp src/filtering.mm
)

ocv_module_include_directories()
ocv_create_module()

if(HAVE_METAL)
  # Link Metal frameworks after module creation
  find_library(METAL_LIBRARY Metal)
  find_library(METALPERFORMANCESHADERS_LIBRARY MetalPerformanceShaders)
  find_library(FOUNDATION_LIBRARY Foundation)
  
  target_link_libraries(${the_module} PRIVATE ${METAL_LIBRARY} ${METALPERFORMANCESHADERS_LIBRARY} ${FOUNDATION_LIBRARY})
endif()

if(HAVE_METAL)
    ocv_add_accuracy_tests(
        FILES test/test_metalfilters.cpp
        DEPENDS_ON opencv_imgproc
    )

    ocv_add_perf_tests(
        FILES perf/perf_metalfilters.cpp
        DEPENDS_ON opencv_imgproc
    )
endif()