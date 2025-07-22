#include "test_precomp.hpp"
#include "opencv2/metalfilters.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

PARAM_TEST_CASE(Metal_FilterBox, cv::Size, MatType, int, Point, int)
{
    int type;
    cv::Size size;
    cv::Size ksize;
    Point anchor;
    int borderType;

    virtual void SetUp()
    {
        size = get<0>(GetParam());
        type = get<1>(GetParam());
        ksize = Size(get<2>(GetParam()), get<2>(GetParam()));
        anchor = get<3>(GetParam());
        borderType = get<4>(GetParam());
    }
};

CUDA_TEST_P(Metal_FilterBox, Accuracy)
{
    Mat src = randomMat(size, type, 0, 255);

    Scalar borderVal = Scalar::all(0);
    if (borderType == BORDER_CONSTANT)
    {
        // Test that non-zero border value throws an exception
        Scalar nonZeroBorderVal = Scalar::all(128);
        ASSERT_THROW(cv::metal::createBoxFilter(src.type(), src.type(), ksize, anchor, borderType, nonZeroBorderVal), cv::Exception);
    }

    Ptr<cv::metal::Filter> boxFilter = cv::metal::createBoxFilter(src.type(), src.type(), ksize, anchor, borderType, borderVal);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;
    boxFilter->apply(d_src, d_dst);

    Mat dst_metal;
    d_dst.download(dst_metal);

    Mat dst_cpu;
    cv::boxFilter(src, dst_cpu, -1, ksize, anchor, true, borderType);

    // Apply appropriate GPU backend tolerance per implementation guide
    // For box filters, use tolerance of 2.0 due to potential border handling differences
    EXPECT_MAT_NEAR(dst_cpu, dst_metal, 2.0);
}

INSTANTIATE_TEST_CASE_P(Imgproc_Metal, Metal_FilterBox,
    testing::Combine(
        testing::Values(szVGA, sz720p),
        testing::Values(CV_8UC1, CV_8UC4, CV_32FC1),
        testing::Values(3, 5, 7),
        testing::Values(Point(-1, -1)),
        testing::Values((int)BORDER_REPLICATE, (int)BORDER_REFLECT, (int)BORDER_REFLECT_101, (int)BORDER_CONSTANT)
    )
);

}} // namespace
#endif // HAVE_METAL