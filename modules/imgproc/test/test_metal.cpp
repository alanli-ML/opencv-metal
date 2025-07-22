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

    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, 1e-5);
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

    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, 1e-4);
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

    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, 1.0);
}
INSTANTIATE_TEST_CASE_P(Imgproc_Metal, Imgproc_Resize,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p),
        testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR),
        testing::Values(0.5, 2.0)
    )
);

}} // namespace
#endif // HAVE_METAL