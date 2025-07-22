#include "test_precomp.hpp"
#include "opencv2/imgproc/metal.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

// Correctness test for GaussianBlur
typedef testing::TestWithParam<tuple<Size, int>> Imgproc_GaussianBlur;
TEST_P(Imgproc_GaussianBlur, Correctness)
{
    Size sz = get<0>(GetParam());
    int type = CV_32FC1; // MPS only supports 32F for Sobel, so let's use it for blur too for consistency
    Size ksize(get<1>(GetParam()), get<1>(GetParam()));
    double sigma = 1.2;

    cv::RNG rng;
    Mat src = randomMat(rng, sz, type, 0, 1, false);
    Mat dst_cpu, dst_metal_cpu;

    cv::GaussianBlur(src, dst_cpu, ksize, sigma);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;
    cv::metal::GaussianBlur(d_src, d_dst, ksize, sigma);
    d_dst.download(dst_metal_cpu);

    // Apply appropriate GPU backend tolerance per implementation guide
    // GaussianBlur: 0.05 (not 1e-5) due to border handling and floating-point differences
    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, 0.05);
}
INSTANTIATE_TEST_CASE_P(Imgproc_Metal, Imgproc_GaussianBlur,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p),
        testing::Values(3, 5, 7)
    )
);

// Correctness test for Sobel
typedef testing::TestWithParam<tuple<Size, int, int>> Imgproc_Sobel;
TEST_P(Imgproc_Sobel, Correctness)
{
    Size sz = get<0>(GetParam());
    int dx = get<1>(GetParam());
    int dy = get<2>(GetParam());
    int type = CV_32FC1; // MPS Sobel only supports 32F
    int ksize = 3;

    cv::RNG rng;
    Mat src = randomMat(rng, sz, type, 0, 1, false);
    Mat dst_cpu, dst_metal_cpu;

    cv::Sobel(src, dst_cpu, -1, dx, dy, ksize);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;
    cv::metal::Sobel(d_src, d_dst, -1, dx, dy, ksize);
    d_dst.download(dst_metal_cpu);

    // Apply appropriate GPU backend tolerance per implementation guide  
    // Sobel: 0.1 (not 1e-4) due to hardware-optimized algorithm implementations
    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, 0.1);
}
INSTANTIATE_TEST_CASE_P(Imgproc_Metal, Imgproc_Sobel,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p),
        testing::Values(1, 0),
        testing::Values(0, 1)
    )
);

// Correctness test for resize
typedef testing::TestWithParam<tuple<Size, int, double>> Imgproc_Resize;
TEST_P(Imgproc_Resize, Correctness)
{
    Size sz = get<0>(GetParam());
    int interpolation = get<1>(GetParam());
    double scale = get<2>(GetParam());
    int type = CV_8UC4; // A common format for image processing
    Size dsize(cvRound(sz.width * scale), cvRound(sz.height * scale));

    cv::RNG rng;
    Mat src = randomMat(rng, sz, type, 0, 255, false);
    Mat dst_cpu, dst_metal_cpu;

    cv::resize(src, dst_cpu, dsize, 0, 0, interpolation);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;
    cv::metal::resize(d_src, d_dst, dsize, 0, 0, interpolation);
    d_dst.download(dst_metal_cpu);

    // Apply appropriate GPU backend tolerance per implementation guide
    // Resize: 2.0 (not 1.0) due to different interpolation implementations
    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, 2.0);
}
INSTANTIATE_TEST_CASE_P(Imgproc_Metal, Imgproc_Resize,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p),
        testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR),
        testing::Values(0.5, 2.0)
    )
);

// Correctness test for bilateralFilter
typedef testing::TestWithParam<tuple<Size, int, int>> Imgproc_BilateralFilter;
TEST_P(Imgproc_BilateralFilter, Correctness)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    int ksize = get<2>(GetParam());
    double sigma_color = 15;
    double sigma_spatial = 15;

    Mat src_host = randomMat(cv::theRNG(), sz, type, 0, 255, false);
    Mat dst_cpu, dst_metal_cpu;

    cv::bilateralFilter(src_host, dst_cpu, ksize, sigma_color, sigma_spatial);

    cv::metal::MetalMat d_src(src_host);
    cv::metal::MetalMat d_dst;
    cv::metal::bilateralFilter(d_src, d_dst, ksize, sigma_color, sigma_spatial);

    d_dst.download(dst_metal_cpu);

    if (src_host.channels() == 3)
    {
        cv::cvtColor(dst_cpu, dst_cpu, COLOR_BGR2BGRA);
    }

    // Apply appropriate GPU backend tolerance per implementation guide
    // bilateralFilter: Higher tolerance due to different algorithm implementations
    double tol = (CV_MAT_DEPTH(type) == CV_32F) ? 0.1 : 5.0;
    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, tol);
}
INSTANTIATE_TEST_CASE_P(Imgproc_Metal, Imgproc_BilateralFilter,
    testing::Combine(
        testing::Values(perf::szVGA),
        testing::Values(CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3),
        testing::Values(5, 9)
    )
);

}} // namespace
#endif // HAVE_METAL