#include "test_precomp.hpp"
#include "opencv2/core/metal.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<Size, MatType>> MetalMatTransferTest;

TEST_P(MetalMatTransferTest, UploadDownload)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src = randomMat(sz, type, -128, 128);
    cv::metal::MetalMat d_src(src);
    Mat dst;
    d_src.download(dst);

    EXPECT_MAT_NEAR(src, dst, 0);
}

INSTANTIATE_TEST_CASE_P(Core_Metal, MetalMatTransferTest,
    testing::Combine(
        testing::Values(szODD, szVGA, sz720p),
        testing::Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4)
    )
);

TEST(Core_Metal, CloneAndROI)
{
    Size sz(256, 256);
    int type = CV_8UC4;
    Mat src = randomMat(sz, type, 0, 255);
    cv::metal::MetalMat d_src(src);

    // Test clone
    cv::metal::MetalMat d_clone = d_src.clone();
    Mat clone_host;
    d_clone.download(clone_host);
    EXPECT_MAT_NEAR(src, clone_host, 0);

    // Test ROI
    Rect roi(32, 64, 100, 120);
    Mat src_roi = src(roi);
    cv::metal::MetalMat d_roi = d_src(roi);
    Mat roi_host;
    d_roi.download(roi_host);
    EXPECT_MAT_NEAR(src_roi, roi_host, 0);
}

TEST(Core_Metal, WrapExternalTexture)
{
    Size sz(128, 128);
    int type = CV_8UC4;
    Mat src = randomMat(sz, type, 0, 255);

    // Create a MetalMat to get a valid MTLTexture
    cv::metal::MetalMat d_src(src);
    id<MTLTexture> texture = (id<MTLTexture>)d_src.texture();
    ASSERT_TRUE(texture != nil);

    // Wrap the existing texture in a new MetalMat
    cv::metal::MetalMat d_wrapped((__bridge id)texture);
    ASSERT_FALSE(d_wrapped.empty());
    EXPECT_EQ(d_wrapped.rows(), sz.height);
    EXPECT_EQ(d_wrapped.cols(), sz.width);
    EXPECT_EQ(d_wrapped.type(), type);

    // Download and verify
    Mat dst;
    d_wrapped.download(dst);
    EXPECT_MAT_NEAR(src, dst, 0);
}

enum ArithmOp { ADD, SUB, MUL, DIV };
void PrintTo(const ArithmOp& op, std::ostream* os)
{
    switch (op)
    {
        case ADD: *os << "ADD"; break;
        case SUB: *os << "SUB"; break;
        case MUL: *os << "MUL"; break;
        case DIV: *os << "DIV"; break;
    }
}

typedef testing::TestWithParam<tuple<Size, MatType, ArithmOp>> Core_ArithmTest;

TEST_P(Core_ArithmTest, Correctness)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    ArithmOp op = get<2>(GetParam());

    Mat src1 = randomMat(sz, type, 1, 255);
    Mat src2 = randomMat(sz, type, 1, 255);
    Mat dst_cpu, dst_metal_cpu;

    switch (op)
    {
        case ADD: cv::add(src1, src2, dst_cpu); break;
        case SUB: cv::subtract(src1, src2, dst_cpu); break;
        case MUL: cv::multiply(src1, src2, dst_cpu); break;
        case DIV: cv::divide(src1, src2, dst_cpu); break;
    }

    cv::metal::MetalMat d_src1(src1);
    cv::metal::MetalMat d_src2(src2);
    cv::metal::MetalMat d_dst;

    switch (op)
    {
        case ADD: cv::metal::add(d_src1, d_src2, d_dst); break;
        case SUB: cv::metal::subtract(d_src1, d_src2, d_dst); break;
        case MUL: cv::metal::multiply(d_src1, d_src2, d_dst); break;
        case DIV: cv::metal::divide(d_src1, d_src2, d_dst); break;
    }

    d_dst.download(dst_metal_cpu);

    double tol = (CV_MAT_DEPTH(type) == CV_32F) ? 1e-5 : 1.0;
    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, tol);
}

INSTANTIATE_TEST_CASE_P(Core_Metal, Core_ArithmTest,
    testing::Combine(
        testing::Values(szVGA, sz720p),
        testing::Values(CV_8UC4, CV_32FC1),
        testing::Values(ADD, SUB, MUL, DIV)
    )
);


}} // namespace
#endif // HAVE_METAL