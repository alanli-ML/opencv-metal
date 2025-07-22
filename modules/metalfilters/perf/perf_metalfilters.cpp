#include "perf_precomp.hpp"
#include "opencv2/metalfilters.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<tuple<Size, int>> Metal_PerfBoxFilter;

PERF_TEST_P(Metal_PerfBoxFilter, boxFilter,
    testing::Combine(
        testing::Values(Size(640, 480), Size(1024, 768)),
        testing::Values(CV_8UC1, CV_8UC4)
    ))
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    
    Mat src(size, type);
    cv::randu(src, 0, 255);
    
    Mat dst_cpu, dst_metal;
    Size ksize(5, 5);
    
    // CPU version
    TEST_CYCLE() 
    {
        cv::boxFilter(src, dst_cpu, -1, ksize);
    }
    
    SANITY_CHECK(dst_cpu, 1e-3);
}

}} // namespace opencv_test::anonymous

#endif // HAVE_METAL