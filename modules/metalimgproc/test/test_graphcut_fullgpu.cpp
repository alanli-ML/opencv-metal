#if 0 // Completely disabled due to Metal header compilation issues
#include "test_precomp.hpp"

#ifdef HAVE_METAL
#ifdef USE_METAL_GRAPHCUT

namespace opencv_test {

TEST(GraphCutFullGPU, SimpleSegmentationOnly)
{
    // Test the simple segmentation kernel independently from the full push-relabel algorithm
    cv::Size testSize(4, 4);
    
    // Create simple test data
    cv::Mat img(testSize, CV_8UC3, cv::Scalar(128, 128, 128));
    img.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 255);    // Blue pixel (should be foreground)
    img.at<cv::Vec3b>(3, 3) = cv::Vec3b(255, 0, 0);    // Red pixel (should be background)
    
    // Upload to Metal
    cv::metal::MetalMat d_img;
    d_img.upload(img);
    
    // Create simple unary terms (background and foreground costs)
    cv::Mat bg_cost(testSize, CV_32F, cv::Scalar(1.0f));  // Background cost
    cv::Mat fg_cost(testSize, CV_32F, cv::Scalar(2.0f));  // Foreground cost
    
    // Make center pixels prefer foreground, outer pixels prefer background
    bg_cost.at<float>(1, 1) = 2.0f; fg_cost.at<float>(1, 1) = 0.5f;
    bg_cost.at<float>(1, 2) = 2.0f; fg_cost.at<float>(1, 2) = 0.5f;
    bg_cost.at<float>(2, 1) = 2.0f; fg_cost.at<float>(2, 1) = 0.5f;
    bg_cost.at<float>(2, 2) = 2.0f; fg_cost.at<float>(2, 2) = 0.5f;
    
    cv::metal::MetalMat d_bg_cost, d_fg_cost;
    d_bg_cost.upload(bg_cost);
    d_fg_cost.upload(fg_cost);
    
    // Create pairwise terms (all zeros for simplicity)
    cv::Mat zero_weights(testSize, CV_32F, cv::Scalar(0.0f));
    cv::metal::MetalMat d_left, d_top, d_tl, d_tr;
    d_left.upload(zero_weights);
    d_top.upload(zero_weights);
    d_tl.upload(zero_weights);
    d_tr.upload(zero_weights);
    
    // Test MetalGraphCut directly
    cv::metal::Stream stream;
    
    try {
        cv::metal::MetalGraphCut graphcut(testSize, stream);
        
        // Build graph
        graphcut.buildGraph(d_bg_cost, d_fg_cost, d_left, d_top, d_tl, d_tr, 1.0);
        
        // Solve (this will use simple segmentation for now)
        graphcut.solve(1);
        
        // Get segmentation
        cv::metal::MetalMat d_mask;
        graphcut.getSegmentation(d_mask);
        
        stream.commitAndWait();
        
        // Download result
        cv::Mat result_mask;
        d_mask.download(result_mask, stream, true);
        
        std::cout << "Simple segmentation test completed successfully" << std::endl;
        std::cout << "Result mask size: " << result_mask.size() << std::endl;
        
        // Print the result for debugging
        for (int y = 0; y < result_mask.rows; y++) {
            for (int x = 0; x < result_mask.cols; x++) {
                std::cout << (int)result_mask.at<uchar>(y, x) << " ";
            }
            std::cout << std::endl;
        }
        
        // Basic sanity check - should have some foreground and background pixels
        int fg_count = 0, bg_count = 0;
        for (int y = 0; y < result_mask.rows; y++) {
            for (int x = 0; x < result_mask.cols; x++) {
                uchar val = result_mask.at<uchar>(y, x);
                if (val == 3) fg_count++;  // probable FG
                else if (val == 2) bg_count++;  // probable BG
            }
        }
        
        std::cout << "FG pixels: " << fg_count << ", BG pixels: " << bg_count << std::endl;
        EXPECT_GT(fg_count, 0);
        EXPECT_GT(bg_count, 0);
        
    } catch (const cv::Exception& e) {
        std::cout << "Exception in simple segmentation test: " << e.what() << std::endl;
        FAIL() << "Simple segmentation should not throw exceptions";
    }
}

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
#endif // #if 0 - disabled file 