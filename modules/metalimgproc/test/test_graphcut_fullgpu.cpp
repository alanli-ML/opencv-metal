#include "test_precomp.hpp"

#ifdef HAVE_METAL
#ifdef USE_METAL_GRAPHCUT

namespace opencv_test {

TEST(GraphCutFullGPU, TinyReachability)
{
    // 4x4 image, top-left pixel bright red (foreground), others grey.
    Size sz(4,4);
    Mat img(sz, CV_8UC3, Scalar(128,128,128));
    img.at<Vec3b>(0,0) = Vec3b(0,0,255);

    Mat mask = Mat::zeros(sz, CV_8UC1);
    // Rectangle covering whole image.
    Rect rect(0,0,4,4);

    Mat bgd, fgd;
    cv::metal::grabCut(img, mask, rect, bgd, fgd, 1, GC_INIT_WITH_RECT);

    // Expect pixel (0,0) to be classified as probable FG (3) after one iter.
    EXPECT_EQ(mask.at<uchar>(0,0), (uchar)GC_PR_FGD);
}

TEST(GraphCutFullGPU, IoUVsCPU_Small)
{
    Size sz(64,64);
    Mat img(sz, CV_8UC3, Scalar(90,90,90));
    circle(img, Point(32,32), 15, Scalar(0,0,220), -1);

    Rect rect(16,16,32,32);

    // CPU reference
    Mat mask_cpu = Mat::zeros(sz, CV_8UC1);
    Mat bgd_c, fgd_c;
    grabCut(img, mask_cpu, rect, bgd_c, fgd_c, 1, GC_INIT_WITH_RECT);

    // GPU version
    Mat mask_gpu = Mat::zeros(sz, CV_8UC1);
    Mat bgd_g, fgd_g;
    cv::metal::grabCut(img, mask_gpu, rect, bgd_g, fgd_g, 1, GC_INIT_WITH_RECT);

    Mat cpu_fg = (mask_cpu==GC_FGD)|(mask_cpu==GC_PR_FGD);
    Mat gpu_fg = (mask_gpu==GC_FGD)|(mask_gpu==GC_PR_FGD);
    double inter = sum(cpu_fg & gpu_fg)[0]/255.0;
    double uni = sum(cpu_fg | gpu_fg)[0]/255.0;
    double iou = (uni>0)? inter/uni : 1.0;
    EXPECT_GT(iou,0.5);
}

} // namespace opencv_test

#endif // USE_METAL_GRAPHCUT
#endif // HAVE_METAL 