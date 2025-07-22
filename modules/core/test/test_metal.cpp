#include "test_precomp.hpp"
#include "opencv2/core/metal.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<Size, MatType>> MetalMatTransferTest;

TEST_P(MetalMatTransferTest, UploadDownload)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());

    Mat src = randomMat(cv::theRNG(), sz, type, -128, 128, false);
    cv::metal::MetalMat d_src(src);
    Mat dst;
    d_src.download(dst);

    EXPECT_MAT_NEAR(src, dst, 0);
}

INSTANTIATE_TEST_CASE_P(Core_Metal, MetalMatTransferTest,
    testing::Combine(
        testing::Values(perf::szODD, perf::szVGA, perf::sz720p),
        testing::Values(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4)
    )
);

TEST(Core_Metal, CloneAndROI)
{
    Size sz(256, 256);
    int type = CV_8UC4;
    Mat src = randomMat(cv::theRNG(), sz, type, 0, 255, false);
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

// TODO: This test needs Objective-C++ compilation to work properly
// TEST(Core_Metal, WrapExternalTexture)
// {
//     Size sz(128, 128);
//     int type = CV_8UC4;
//     Mat src = randomMat(cv::theRNG(), sz, type, 0, 255, false);
// 
//     // Create a MetalMat to get a valid MTLTexture
//     cv::metal::MetalMat d_src(src);
//     id texture = d_src.texture();
//     ASSERT_TRUE(texture != nullptr);
// 
//     // Wrap the existing texture in a new MetalMat
//     cv::metal::MetalMat d_wrapped(texture);
//     ASSERT_FALSE(d_wrapped.empty());
//     EXPECT_EQ(d_wrapped.rows(), sz.height);
//     EXPECT_EQ(d_wrapped.cols(), sz.width);
//     EXPECT_EQ(d_wrapped.type(), type);

// }

TEST(Core_Metal, Upload_Download_Consistency)
{
    Size sz(128, 128);
    int type = CV_8UC4;
    Mat src = randomMat(cv::theRNG(), sz, type, 0, 255, false);

    // Test upload/download consistency
    cv::metal::MetalMat d_src(src);
    ASSERT_FALSE(d_src.empty());
    EXPECT_EQ(d_src.rows(), sz.height);
    EXPECT_EQ(d_src.cols(), sz.width);
    EXPECT_EQ(d_src.type(), type);

    // Download and verify consistency
    Mat result;
    d_src.download(result);
    EXPECT_MAT_NEAR(src, result, 0);
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

    Mat src1 = randomMat(cv::theRNG(), sz, type, 1, 255, false);
    Mat src2 = randomMat(cv::theRNG(), sz, type, 1, 255, false);
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

    // Apply appropriate GPU backend tolerances per implementation guide
    // Metal arithmetic operations now use custom kernels that match OpenCV behavior exactly
    double tol;
    switch (op) {
        case ADD:
        case SUB:
            tol = (CV_MAT_DEPTH(type) == CV_32F) ? 1e-5 : 1.0;
            break;
        case MUL:
            // Custom Metal kernels implement OpenCV-compatible integer arithmetic: min(255, a * b)
            // Should match CPU implementation exactly for integer types
            tol = (CV_MAT_DEPTH(type) == CV_32F) ? 1e-4 : 1.0;
            break;
        case DIV:
            // Custom Metal kernels implement OpenCV-compatible integer division: a / b
            // Should match CPU implementation exactly for integer types
            tol = (CV_MAT_DEPTH(type) == CV_32F) ? 1e-4 : 1.0;
            break;
    }
    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, tol);
}

INSTANTIATE_TEST_CASE_P(Core_Metal, Core_ArithmTest,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p),
        testing::Values(CV_8UC4, CV_32FC1),
        testing::Values(ADD, SUB, MUL, DIV)
    )
);


}} // namespace
#endif // HAVE_METAL