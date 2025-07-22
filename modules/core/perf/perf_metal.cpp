#include "perf_precomp.hpp"
#include "opencv2/core/metal.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<tuple<Size, MatType>> Core_Arithm;

PERF_TEST_P(Core_Arithm, Add,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(CV_8UC4, CV_32FC1, CV_32FC4)
    )
)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src1 = randomMat(sz, type, 0, 255);
    Mat src2 = randomMat(sz, type, 0, 255);

    cv::metal::MetalMat d_src1(src1);
    cv::metal::MetalMat d_src2(src2);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::add(d_src1, d_src2, d_dst);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Core_Arithm, Subtract,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(CV_8UC4, CV_32FC1, CV_32FC4)
    )
)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src1 = randomMat(sz, type, 0, 255);
    Mat src2 = randomMat(sz, type, 0, 255);

    cv::metal::MetalMat d_src1(src1);
    cv::metal::MetalMat d_src2(src2);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::subtract(d_src1, d_src2, d_dst);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Core_Arithm, Multiply,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(CV_8UC4, CV_32FC1, CV_32FC4)
    )
)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src1 = randomMat(sz, type, 0, 255);
    Mat src2 = randomMat(sz, type, 0, 255);

    cv::metal::MetalMat d_src1(src1);
    cv::metal::MetalMat d_src2(src2);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::multiply(d_src1, d_src2, d_dst);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Core_Arithm, Divide,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(CV_8UC4, CV_32FC1, CV_32FC4)
    )
)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src1 = randomMat(sz, type, 1, 255);
    Mat src2 = randomMat(sz, type, 1, 255);

    cv::metal::MetalMat d_src1(src1);
    cv::metal::MetalMat d_src2(src2);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::divide(d_src1, d_src2, d_dst);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace
#endif // HAVE_METAL