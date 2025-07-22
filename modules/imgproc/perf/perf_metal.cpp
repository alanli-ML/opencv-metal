#include "perf_precomp.hpp"
#include "opencv2/imgproc/metal.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

// Performance test for GaussianBlur
typedef perf::TestBaseWithParam<tuple<Size, int>> Imgproc_GaussianBlur;
PERF_TEST_P(Imgproc_GaussianBlur, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(3, 7)
    )
)
{
    Size sz = get<0>(GetParam());
    int ksize_val = get<1>(GetParam());
    Size ksize(ksize_val, ksize_val);
    int type = CV_32FC1;
    double sigma = 1.2;

    Mat src(sz, type);
    randu(src, 0, 1);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::GaussianBlur(d_src, d_dst, ksize, sigma);
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for Sobel
typedef perf::TestBaseWithParam<tuple<Size, int, int>> Imgproc_Sobel;
PERF_TEST_P(Imgproc_Sobel, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(1, 0),
        testing::Values(0, 1)
    )
)
{
    Size sz = get<0>(GetParam());
    int dx = get<1>(GetParam());
    int dy = get<2>(GetParam());
    int type = CV_32FC1;
    int ksize = 3;

    Mat src(sz, type);
    randu(src, 0, 1);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::Sobel(d_src, d_dst, -1, dx, dy, ksize);
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for resize
typedef perf::TestBaseWithParam<tuple<Size, int, double>> Imgproc_Resize;
PERF_TEST_P(Imgproc_Resize, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR),
        testing::Values(0.5, 2.0)
    )
)
{
    Size sz = get<0>(GetParam());
    int interpolation = get<1>(GetParam());
    double scale = get<2>(GetParam());
    int type = CV_8UC4;
    Size dsize(cvRound(sz.width * scale), cvRound(sz.height * scale));

    Mat src(sz, type);
    randu(src, 0, 255);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::resize(d_src, d_dst, dsize, 0, 0, interpolation);
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for chained operations
typedef perf::TestBaseWithParam<Size> Imgproc_ChainedOps;

PERF_TEST_P(Imgproc_ChainedOps, Sync,
    testing::Values(szVGA, sz720p, sz1080p)
)
{
    Size sz = GetParam();
    int type = CV_32FC1;
    Size dsize(sz.width / 2, sz.height / 2);
    Size ksize(5, 5);
    double sigma = 1.2;

    Mat src(sz, type);
    randu(src, 0, 1);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_resized, d_blurred, d_sobel;

    TEST_CYCLE()
    {
        cv::metal::resize(d_src, d_resized, dsize, 0, 0, INTER_LINEAR);
        cv::metal::GaussianBlur(d_resized, d_blurred, ksize, sigma);
        cv::metal::Sobel(d_blurred, d_sobel, -1, 1, 0, 3);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Imgproc_ChainedOps, AsyncStream,
    testing::Values(szVGA, sz720p, sz1080p)
)
{
    Size sz = GetParam();
    int type = CV_32FC1;
    Size dsize(sz.width / 2, sz.height / 2);
    Size ksize(5, 5);
    double sigma = 1.2;

    Mat src(sz, type);
    randu(src, 0, 1);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_resized, d_blurred, d_sobel;
    cv::metal::Stream stream;

    TEST_CYCLE()
    {
        cv::metal::resize(d_src, d_resized, dsize, 0, 0, INTER_LINEAR, stream);
        cv::metal::GaussianBlur(d_resized, d_blurred, ksize, sigma, stream);
        cv::metal::Sobel(d_blurred, d_sobel, -1, 1, 0, 3, stream);
        stream.commitAndWait();
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for bilateralFilter
typedef perf::TestBaseWithParam<tuple<Size, int>> Imgproc_BilateralFilter;
PERF_TEST_P(Imgproc_BilateralFilter, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p),
        testing::Values(5, 9)
    )
)
{
    Size sz = get<0>(GetParam());
    int ksize = get<1>(GetParam());
    int type = CV_8UC4;
    float sigma_color = 15;
    float sigma_spatial = 15;

    Mat src(sz, type);
    randu(src, 0, 255);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::bilateralFilter(d_src, d_dst, ksize, sigma_color, sigma_spatial);
    }

    SANITY_CHECK_NOTHING();
}


}} // namespace
#endif // HAVE_METAL