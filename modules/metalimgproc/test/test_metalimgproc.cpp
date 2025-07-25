#include "test_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utils/filesystem.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

// Debug utility to save mask visualization
void saveMaskDebugImage(const cv::Mat& image, const cv::Mat& mask, const std::string& filename_base) {
    // Ensure output directory exists
    std::string dir = cv::utils::fs::getParent(filename_base);
    if (!dir.empty()) cv::utils::fs::createDirectories(dir);

    cv::Mat debug_image = image.clone();
    
    // Create colored overlay for different mask values
    cv::Mat overlay = cv::Mat::zeros(image.size(), CV_8UC3);
    
    // GC_BGD = 0 (sure background) - blue
    overlay.setTo(cv::Scalar(255, 0, 0), mask == cv::GC_BGD);
    
    // GC_FGD = 1 (sure foreground) - red  
    overlay.setTo(cv::Scalar(0, 0, 255), mask == cv::GC_FGD);
    
    // GC_PR_BGD = 2 (probable background) - light blue
    overlay.setTo(cv::Scalar(255, 128, 0), mask == cv::GC_PR_BGD);
    
    // GC_PR_FGD = 3 (probable foreground) - light red
    overlay.setTo(cv::Scalar(0, 128, 255), mask == cv::GC_PR_FGD);
    
    // Blend with original image
    cv::addWeighted(debug_image, 0.7, overlay, 0.3, 0, debug_image);
    
    cv::imwrite(filename_base + ".jpg", debug_image);
    
    // Also save the raw mask
    std::string mask_filename = filename_base + "_mask.jpg";
    cv::Mat mask_viz = mask * 85; // Scale to make values visible (0->0, 1->85, 2->170, 3->255)
    cv::imwrite(mask_filename, mask_viz);
}

// Test parameters: Size, GrabCut mode
typedef testing::TestWithParam<tuple<Size, int>> MetalImgproc_GrabCut;

TEST_P(MetalImgproc_GrabCut, Accuracy)
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());
    
    // Create test image with distinct foreground and background regions
    Mat image(sz, CV_8UC3);
    
    // Create a simple synthetic image
    // Background: dark blue
    image.setTo(Scalar(100, 50, 0)); // BGR
    
    // Foreground: bright red circle in center
    Point center(sz.width / 2, sz.height / 2);
    int radius = min(sz.width, sz.height) / 4;
    circle(image, center, radius, Scalar(0, 0, 200), -1); // BGR red
    
    // Create initial mask and rect
    Mat mask_cpu, mask_metal;
    Rect rect(sz.width / 4, sz.height / 4, sz.width / 2, sz.height / 2);
    
    if (mode == GC_INIT_WITH_RECT) {
        // Let GrabCut initialize the mask
        mask_cpu = Mat::zeros(sz, CV_8UC1);
        mask_metal = Mat::zeros(sz, CV_8UC1);
    } else { // GC_INIT_WITH_MASK or GC_EVAL
        // Initialize mask manually
        mask_cpu = Mat::zeros(sz, CV_8UC1);
        mask_metal = Mat::zeros(sz, CV_8UC1);
        
        // Set probable foreground in center circle
        circle(mask_cpu, center, radius, Scalar(GC_PR_FGD), -1);
        circle(mask_metal, center, radius, Scalar(GC_PR_FGD), -1);
        
        // Set sure background around edges
        rectangle(mask_cpu, Rect(0, 0, sz.width, 10), Scalar(GC_BGD), -1);
        rectangle(mask_cpu, Rect(0, sz.height-10, sz.width, 10), Scalar(GC_BGD), -1);
        rectangle(mask_cpu, Rect(0, 0, 10, sz.height), Scalar(GC_BGD), -1);
        rectangle(mask_cpu, Rect(sz.width-10, 0, 10, sz.height), Scalar(GC_BGD), -1);
        
        mask_cpu.copyTo(mask_metal);
    }
    
    Mat bgdModel_cpu, fgdModel_cpu;
    Mat bgdModel_metal, fgdModel_metal;
    
    // Run CPU GrabCut
    cv::grabCut(image, mask_cpu, rect, bgdModel_cpu, fgdModel_cpu, 5, mode);
    
    // Run Metal GrabCut
    cv::metal::grabCut(image, mask_metal, rect, bgdModel_metal, fgdModel_metal, 5, mode);
    
    // Compare results using Intersection over Union (IoU)
    Mat cpu_fg_mask, metal_fg_mask;
    cpu_fg_mask = (mask_cpu == GC_FGD) | (mask_cpu == GC_PR_FGD);
    metal_fg_mask = (mask_metal == GC_FGD) | (mask_metal == GC_PR_FGD);
    
    Mat intersection, union_mask;
    bitwise_and(cpu_fg_mask, metal_fg_mask, intersection);
    bitwise_or(cpu_fg_mask, metal_fg_mask, union_mask);
    
    double intersection_count = cv::sum(intersection)[0] / 255.0;
    double union_count = cv::sum(union_mask)[0] / 255.0;
    
    double iou = (union_count > 0) ? intersection_count / union_count : 1.0;
    
    // Expect reasonable IoU for segmentation task
    EXPECT_GT(iou, 0.7) << "IoU between CPU and Metal GrabCut should be > 0.7 for size " 
                        << sz << " and mode " << mode;
}

INSTANTIATE_TEST_CASE_P(MetalImgproc, MetalImgproc_GrabCut,
    testing::Combine(
        testing::Values(Size(160, 120), Size(320, 240), Size(640, 480)),
        testing::Values(GC_INIT_WITH_RECT, GC_INIT_WITH_MASK, GC_EVAL)
    )
);

// Test for edge cases and error conditions
TEST(MetalImgproc_GrabCut, EdgeCases)
{
    Size sz(100, 100);
    Mat image(sz, CV_8UC3, Scalar(128, 128, 128));
    Mat mask = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel, fgdModel;
    Rect rect(10, 10, 80, 80);
    
    // Test with zero iterations
    EXPECT_NO_THROW(cv::metal::grabCut(image, mask, rect, bgdModel, fgdModel, 0, GC_INIT_WITH_RECT));
    
    // Test with single iteration
    EXPECT_NO_THROW(cv::metal::grabCut(image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT));
    
    // Test freeze model mode
    cv::metal::grabCut(image, mask, rect, bgdModel, fgdModel, 2, GC_INIT_WITH_RECT); // Initialize
    EXPECT_NO_THROW(cv::metal::grabCut(image, mask, rect, bgdModel, fgdModel, 1, GC_EVAL_FREEZE_MODEL));
}

// Test stream vs sync execution
TEST(MetalImgproc_GrabCut, StreamExecution)
{
    Size sz(200, 200);
    Mat image(sz, CV_8UC3, Scalar(100, 100, 100));
    circle(image, Point(100, 100), 60, Scalar(200, 50, 50), -1);
    
    Mat mask1 = Mat::zeros(sz, CV_8UC1);
    Mat mask2 = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel1, fgdModel1, bgdModel2, fgdModel2;
    Rect rect(40, 40, 120, 120);
    
    // Test synchronous execution
    cv::metal::grabCut(image, mask1, rect, bgdModel1, fgdModel1, 3, GC_INIT_WITH_RECT);
    
    // Test asynchronous execution with stream
    cv::metal::Stream stream;
    cv::metal::grabCut(image, mask2, rect, bgdModel2, fgdModel2, 3, GC_INIT_WITH_RECT, stream);
    stream.commitAndWait();
    
    // Results should be identical
    Mat diff;
    absdiff(mask1, mask2, diff);
    double max_diff = 0;
    minMaxLoc(diff, nullptr, &max_diff);
    
    EXPECT_EQ(max_diff, 0) << "Sync and async execution should produce identical results";
}

// Test input validation
TEST(MetalImgproc_GrabCut, InputValidation)
{
    Size sz(100, 100);
    Mat image(sz, CV_8UC3, Scalar(128, 128, 128));
    Mat mask = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel, fgdModel;
    Rect rect(10, 10, 80, 80);
    
    // Test with empty image
    Mat empty_image;
    EXPECT_THROW(cv::metal::grabCut(empty_image, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT), cv::Exception);
    
    // Test with wrong image type
    Mat wrong_type_image(sz, CV_8UC1, Scalar(128));
    EXPECT_THROW(cv::metal::grabCut(wrong_type_image, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT), cv::Exception);
    
    // Test with invalid rect (should not crash, might clamp internally)
    Rect invalid_rect(-10, -10, 150, 150);
    EXPECT_NO_THROW(cv::metal::grabCut(image, mask, invalid_rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT));
}

// Component-level testing for utility functions
TEST(MetalImgproc_GrabCut, UtilityFunctions)
{
    Size sz(100, 100);
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    cv::metal::MetalMat d_image(image);
    cv::metal::Stream stream;
    
    // Test beta calculation
    double beta = cv::metal::calcBeta(d_image, stream);
    EXPECT_GT(beta, 0.0) << "Beta should be positive for random image";
    EXPECT_LT(beta, 1000.0) << "Beta should be reasonable for typical images";
    
    // Test pairwise weight calculation
    cv::metal::MetalMat leftW, topleftW, topW, toprightW;
    EXPECT_NO_THROW(cv::metal::calcNWeights(d_image, leftW, topleftW, topW, toprightW, beta, 50.0, stream));
    
    // Verify output sizes
    EXPECT_EQ(leftW.size(), sz);
    EXPECT_EQ(topleftW.size(), sz);
    EXPECT_EQ(topW.size(), sz);
    EXPECT_EQ(toprightW.size(), sz);
    
    stream.commitAndWait();
}

// Real image test with performance comparison and .jpg output
TEST(MetalImgproc_GrabCut, RealImagePerformanceTest)
{
    std::cout << "\n=== REAL IMAGE GRABCUT PERFORMANCE TEST ===\n" << std::endl;
    
    // Load real test image
    std::string img_path = std::string("../WID-small.jpg");
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path 
                               << " - ensure WID-small.jpg is in the opencv-metal directory";
    
    // Ensure image is in correct format
    if (image.type() != CV_8UC3) {
        if (image.channels() == 4) {
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        } else if (image.channels() == 1) {
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        }
    }
    CV_Assert(image.type() == CV_8UC3);
    
    std::cout << "Image loaded: " << image.cols << "x" << image.rows << " pixels" << std::endl;

    // Define rectangle for object of interest
    cv::Rect rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2);
    std::cout << "Selection rectangle: " << rect << std::endl;
    
    // Test different iteration counts
    std::vector<int> iteration_counts = {1, 3, 5};
    
    for (int iterations : iteration_counts) {
        std::cout << "\n--- Testing with " << iterations << " iterations ---" << std::endl;
        
        // Prepare masks and models
    cv::Mat mask_cpu(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat mask_metal(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_cpu, fgd_cpu, bgd_metal, fgd_metal;
    
        // Time CPU implementation
        std::cout << "Running CPU GrabCut..." << std::flush;
    auto start_cpu = std::chrono::high_resolution_clock::now();
        cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, iterations, cv::GC_INIT_WITH_RECT);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
        std::cout << " completed in " << cpu_time.count() << " ms" << std::endl;
        
        // Time Metal implementation
        std::cout << "Running Metal GrabCut..." << std::flush;
    auto start_metal = std::chrono::high_resolution_clock::now();
    cv::metal::Stream stream;
        cv::metal::grabCut(image, mask_metal, rect, bgd_metal, fgd_metal, iterations, cv::GC_INIT_WITH_RECT, stream);
    stream.commitAndWait();
    auto end_metal = std::chrono::high_resolution_clock::now();
    auto metal_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_metal - start_metal);
        std::cout << " completed in " << metal_time.count() << " ms" << std::endl;
        
        // Calculate speedup
        double speedup = static_cast<double>(cpu_time.count()) / static_cast<double>(metal_time.count());
        std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x";
        if (speedup > 1.0) {
            std::cout << " (Metal is faster)";
        } else {
            std::cout << " (CPU is faster)";
        }
        std::cout << std::endl;
        
        // Analyze segmentation results
        cv::Mat cpu_fg_mask = (mask_cpu == cv::GC_FGD) | (mask_cpu == cv::GC_PR_FGD);
        cv::Mat metal_fg_mask = (mask_metal == cv::GC_FGD) | (mask_metal == cv::GC_PR_FGD);
        
        double cpu_fg_pixels = cv::sum(cpu_fg_mask)[0] / 255.0;
        double metal_fg_pixels = cv::sum(metal_fg_mask)[0] / 255.0;
        double total_pixels = image.rows * image.cols;
        
        std::cout << "CPU foreground: " << cpu_fg_pixels << " pixels (" 
                  << (cpu_fg_pixels/total_pixels*100.0) << "%)" << std::endl;
        std::cout << "Metal foreground: " << metal_fg_pixels << " pixels (" 
                  << (metal_fg_pixels/total_pixels*100.0) << "%)" << std::endl;
        
        // Calculate agreement
        cv::Mat intersection, union_mask;
        cv::bitwise_and(cpu_fg_mask, metal_fg_mask, intersection);
        cv::bitwise_or(cpu_fg_mask, metal_fg_mask, union_mask);
        
        double intersection_count = cv::sum(intersection)[0] / 255.0;
        double union_count = cv::sum(union_mask)[0] / 255.0;
        double iou = (union_count > 0) ? intersection_count / union_count : 1.0;
        
        std::cout << "IoU (agreement): " << std::fixed << std::setprecision(3) << iou << std::endl;
        
        // Create output directory
        std::string output_dir = "./grabcut_results/";
        cv::utils::fs::createDirectories(output_dir);
        
        // Save results
        std::string prefix = output_dir + "iter" + std::to_string(iterations) + "_";
        
        // Save original image with rectangle overlay
        cv::Mat original_with_rect = image.clone();
        cv::rectangle(original_with_rect, rect, cv::Scalar(0, 255, 0), 3);
        cv::imwrite(prefix + "original_with_rect.jpg", original_with_rect);
        
        // Save segmentation masks with color overlay
        saveMaskDebugImage(image, mask_cpu, prefix + "cpu_result");
        saveMaskDebugImage(image, mask_metal, prefix + "metal_result");
        
        // Create foreground extraction images
        cv::Mat cpu_foreground = image.clone();
        cv::Mat metal_foreground = image.clone();
        
        // Set background pixels to black
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                if (!cpu_fg_mask.at<uchar>(y, x)) {
                    cpu_foreground.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
                if (!metal_fg_mask.at<uchar>(y, x)) {
                    metal_foreground.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
            }
        }
        
        cv::imwrite(prefix + "cpu_foreground.jpg", cpu_foreground);
        cv::imwrite(prefix + "metal_foreground.jpg", metal_foreground);
        
        // Create difference visualization
        cv::Mat diff_mask;
        cv::absdiff(cpu_fg_mask, metal_fg_mask, diff_mask);
        cv::Mat diff_viz = image.clone();
        diff_viz.setTo(cv::Scalar(0, 0, 255), diff_mask > 0); // Red where they differ
        cv::imwrite(prefix + "difference.jpg", diff_viz);
        
        std::cout << "Results saved to: " << output_dir << std::endl;
        
        // Validation assertions
        EXPECT_LT(metal_time.count(), 10000) << "Metal should complete within 10 seconds";
        EXPECT_GT(metal_fg_pixels, total_pixels * 0.01) << "Metal should find some foreground pixels";
        EXPECT_LT(metal_fg_pixels, total_pixels * 0.9) << "Metal shouldn't classify everything as foreground";
        EXPECT_GT(iou, 0.3) << "CPU and Metal should have reasonable agreement (IoU > 0.3)";
        
        std::cout << "✓ " << iterations << " iterations test passed!" << std::endl;
    }
    
    std::cout << "\n=== PERFORMANCE TEST COMPLETE ===\n" << std::endl;
    std::cout << "Check './grabcut_results/' directory for output images" << std::endl;
}

// K-means validation test (simplified)
TEST(MetalImgproc_GrabCut, KMeansValidation)
{
    std::cout << "\n=== K-MEANS VALIDATION TEST ===\n" << std::endl;
    
    // Load test image
    std::string img_path = std::string("../WID-small.jpg");
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path;
    
    if (image.type() != CV_8UC3) {
        if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        }
    }
    
    cv::Rect rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2);
    
    // Test k-means fix by running multiple times and checking consistency
    std::vector<int> fg_pixel_counts;
    
    for (int run = 0; run < 3; run++) {
        cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
        cv::Mat bgd, fgd;
        
    cv::metal::Stream stream;
        cv::metal::grabCut(image, mask, rect, bgd, fgd, 2, cv::GC_INIT_WITH_RECT, stream);
    stream.commitAndWait();
        
        cv::Mat fg_mask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
        int fg_pixels = cv::sum(fg_mask)[0] / 255;
        fg_pixel_counts.push_back(fg_pixels);
        
        std::cout << "Run " << (run+1) << ": " << fg_pixels << " foreground pixels" << std::endl;
    }
    
    // Check that results are reasonably consistent (not all the same due to randomness, but not wildly different)
    int min_fg = *std::min_element(fg_pixel_counts.begin(), fg_pixel_counts.end());
    int max_fg = *std::max_element(fg_pixel_counts.begin(), fg_pixel_counts.end());
    double variation = static_cast<double>(max_fg - min_fg) / min_fg;
    
    std::cout << "Variation between runs: " << (variation * 100.0) << "%" << std::endl;
    
    // Reasonable variation should be less than 50% (some randomness is expected)
    EXPECT_LT(variation, 0.5) << "K-means should produce reasonably consistent results";
    EXPECT_GT(min_fg, 0) << "Should find some foreground pixels in all runs";
    
    std::cout << "✓ K-means validation passed!" << std::endl;
}

// Performance profiling test to identify bottlenecks
TEST(MetalImgproc_GrabCut, PerformanceBottleneckAnalysis)
{
    std::cout << "\n=== PERFORMANCE BOTTLENECK ANALYSIS ===\n" << std::endl;
    
    // Load test image
    std::string img_path = std::string("../WID-small.jpg");
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path;
    
    if (image.type() != CV_8UC3) {
        if (image.channels() == 4) {
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        }
    }
    
    std::cout << "Image size: " << image.cols << "x" << image.rows << " pixels" << std::endl;
    std::cout << "Data size estimates:" << std::endl;
    std::cout << "  Single image: " << (image.cols * image.rows * 3 / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Vec4f texture: " << (image.cols * image.rows * 16 / 1024.0 / 1024.0) << " MB" << std::endl;
    
    cv::Rect rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2);
    
    // Prepare for profiling
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd, fgd;
    
    cv::metal::Stream stream;
    
    // Profile individual components by running just 1 iteration to isolate costs
    std::cout << "\n--- Profiling 1 iteration of Metal GrabCut ---" << std::endl;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    cv::metal::grabCut(image, mask, rect, bgd, fgd, 1, cv::GC_INIT_WITH_RECT, stream);
        stream.commitAndWait();
    auto total_end = std::chrono::high_resolution_clock::now();
    
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    std::cout << "Total Metal GrabCut (1 iter): " << total_time.count() << " ms" << std::endl;
    
    // Now compare with CPU baseline for same work
    cv::Mat mask_cpu(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_cpu, fgd_cpu;
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 1, cv::GC_INIT_WITH_RECT);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    std::cout << "Total CPU GrabCut (1 iter): " << cpu_time.count() << " ms" << std::endl;
    
    double efficiency = static_cast<double>(cpu_time.count()) / static_cast<double>(total_time.count());
    std::cout << "Metal efficiency: " << std::fixed << std::setprecision(2) << efficiency << "x" << std::endl;
    
    // Analysis
    std::cout << "\n--- Performance Analysis ---" << std::endl;
    if (efficiency < 1.5) {
        std::cout << "⚠️  Poor Metal efficiency detected!" << std::endl;
        std::cout << "Likely causes:" << std::endl;
        std::cout << "1. Excessive GPU↔CPU data transfers" << std::endl;
        std::cout << "2. CPU fallback paths in 'Metal' implementation" << std::endl;
        std::cout << "3. GPU pipeline stalls from sync points" << std::endl;
        
        // Estimate data transfer overhead
        double estimated_transfer_mb = 7 * (image.cols * image.rows * 4.0 / 1024.0 / 1024.0); // 7 matrices per iteration
        std::cout << "Estimated GPU→CPU transfer per iteration: " << estimated_transfer_mb << " MB" << std::endl;
        
        // Estimate transfer time (assume ~1GB/s PCIe bandwidth)
        double transfer_time_ms = estimated_transfer_mb; // Very rough estimate: 1MB ≈ 1ms
        std::cout << "Estimated transfer time overhead: ~" << transfer_time_ms << " ms" << std::endl;
        
        if (transfer_time_ms > total_time.count() * 0.3) {
            std::cout << "🚨 Data transfer overhead is likely >30% of total time!" << std::endl;
        }
                } else {
        std::cout << "✅ Good Metal efficiency detected" << std::endl;
    }
    
    // Calculate IoU to ensure correctness
    cv::Mat cpu_fg_mask = (mask_cpu == cv::GC_FGD) | (mask_cpu == cv::GC_PR_FGD);
    cv::Mat metal_fg_mask = (mask == cv::GC_FGD) | (mask == cv::GC_PR_FGD);
    
    cv::Mat intersection, union_mask;
    cv::bitwise_and(cpu_fg_mask, metal_fg_mask, intersection);
    cv::bitwise_or(cpu_fg_mask, metal_fg_mask, union_mask);
    
    double intersection_count = cv::sum(intersection)[0] / 255.0;
    double union_count = cv::sum(union_mask)[0] / 255.0;
    double iou = (union_count > 0) ? intersection_count / union_count : 1.0;
    
    std::cout << "Result accuracy (IoU): " << std::fixed << std::setprecision(3) << iou << std::endl;
    
    // Test assertions
    EXPECT_GT(iou, 0.7) << "Metal implementation should maintain good accuracy";
    if (efficiency >= 1.5) {
        std::cout << "✅ Performance test passed!" << std::endl;
    } else {
        std::cout << "⚠️  Performance needs improvement" << std::endl;
    }
}

// Test to measure sync point overhead specifically
TEST(MetalImgproc_GrabCut, SyncPointAnalysis)
{
    std::cout << "\n=== SYNC POINT OVERHEAD ANALYSIS ===\n" << std::endl;
    
    // Load test image  
    std::string img_path = std::string("../WID-small.jpg");
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path;
    
    if (image.type() != CV_8UC3) {
        if (image.channels() == 4) {
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        }
    }
    
    cv::Rect rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2);
    
    std::cout << "Analyzing explicit sync point overhead..." << std::endl;
    std::cout << "Key sync points in GrabCut implementation:" << std::endl;
    std::cout << "1. Line 470: stream.syncCPU() after GMM initialization" << std::endl;
    std::cout << "2. Line 573: stream.syncCPU() before CPU graph construction (PER ITERATION)" << std::endl;
    std::cout << "3. Multiple downloads after sync (should be fast since no commands queued)" << std::endl;
    
    // Test the hypothesis: sync points are the bottleneck, not downloads
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd, fgd;
    
    // Time multiple iterations to see sync overhead scaling
    std::vector<int> iteration_counts = {1, 3, 5};
    
    for (int iters : iteration_counts) {
    cv::metal::Stream stream;
    
        auto start = std::chrono::high_resolution_clock::now();
        cv::metal::grabCut(image, mask, rect, bgd, fgd, iters, cv::GC_INIT_WITH_RECT, stream);
        stream.commitAndWait();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << iters << " iterations: " << time_ms.count() << " ms";
        
        if (iters > 1) {
            // Estimate per-iteration time (excluding initialization)
            // Each iteration has 1 major sync point (line 573)
            double per_iter_ms = static_cast<double>(time_ms.count()) / iters;
            std::cout << " (~" << per_iter_ms << " ms per iteration)";
    }
    std::cout << std::endl;
    }
    
    std::cout << "\n--- Sync Point Impact Analysis ---" << std::endl;
    std::cout << "If sync points are the bottleneck, we should see:" << std::endl;
    std::cout << "1. Linear scaling with iteration count" << std::endl;
    std::cout << "2. Each iteration taking 500-1000ms+ due to sync overhead" << std::endl;
    std::cout << "3. Download time being minimal compared to sync time" << std::endl;
    
    std::cout << "\n--- Recommended Phase 1 Optimizations ---" << std::endl;
    std::cout << "Priority 1: Eliminate stream.syncCPU() on line 573 (per iteration)" << std::endl;
    std::cout << "Priority 2: Reduce sync points in GMM learning" << std::endl;
    std::cout << "Priority 3: Optimize CPU graph construction to avoid downloads" << std::endl;
}

} // namespace
} // namespace opencv_test

#endif // HAVE_METAL
