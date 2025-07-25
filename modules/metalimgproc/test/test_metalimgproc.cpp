#include "test_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/imgproc/grabcut_shared.hpp"

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
    
    // Debug output for failing cases
    if (iou <= 0.7) {
        std::cout << "Low IoU detected: " << iou << " for size " << sz << " mode " << mode << std::endl;
        std::cout << "CPU foreground pixels: " << cv::sum(cpu_fg_mask)[0] / 255.0 << std::endl;
        std::cout << "Metal foreground pixels: " << cv::sum(metal_fg_mask)[0] / 255.0 << std::endl;
        std::cout << "Intersection pixels: " << intersection_count << std::endl;
        std::cout << "Union pixels: " << union_count << std::endl;
    }
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

// Performance comparison test (light version for unit tests)
TEST(MetalImgproc_GrabCut, BasicPerformance)
{
    Size sz(400, 300);
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    Mat mask_cpu = Mat::zeros(sz, CV_8UC1);
    Mat mask_metal = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel_cpu, fgdModel_cpu, bgdModel_metal, fgdModel_metal;
    Rect rect(50, 50, 300, 200);
    
    // Time CPU version
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cv::grabCut(image, mask_cpu, rect, bgdModel_cpu, fgdModel_cpu, 3, GC_INIT_WITH_RECT);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    
    // Time Metal version
    auto start_metal = std::chrono::high_resolution_clock::now();
    cv::metal::grabCut(image, mask_metal, rect, bgdModel_metal, fgdModel_metal, 3, GC_INIT_WITH_RECT);
    auto end_metal = std::chrono::high_resolution_clock::now();
    
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    auto metal_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_metal - start_metal);
    
    std::cout << "CPU GrabCut time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "Metal GrabCut time: " << metal_time.count() << " ms" << std::endl;
    
    // Metal should complete in reasonable time (not a strict performance requirement for unit tests)
    EXPECT_LT(metal_time.count(), 5000) << "Metal GrabCut should complete within 5 seconds for 400x300 image";
}

// Add real photo comparison test
TEST(MetalImgproc_GrabCut, RealPhoto)
{
    std::cout << "RealPhoto test started!" << std::endl;
    
    // Path to a real image in the repo
    std::string img_path = std::string("../WID-small.jpg");  // Fixed path for build directory
    std::cout << "Trying to load image from: " << img_path << std::endl;
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR | cv::IMREAD_ANYDEPTH);
    std::cout << "Image loaded, empty=" << image.empty() << std::endl;
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path;
    
    std::cout << "Initial image type=" << image.type() << " depth=" << image.depth() << " channels=" << image.channels() << std::endl;
    
    // Always convert to 8-bit 3-channel to ensure compatibility
    if (image.depth() != CV_8U) {
        std::cout << "Converting depth from " << image.depth() << " to CV_8U" << std::endl;
        image.convertTo(image, CV_8U, 1.0 / 256.0);   // down-convert from 16-bit
    }
    if (image.channels() != 3) {
        std::cout << "Converting channels from " << image.channels() << " to 3" << std::endl;
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);  // ensure 3 channels
    }
    
    std::cout << "Final image type=" << image.type() << " (CV_8UC3=" << CV_8UC3 << ")" << std::endl;
    CV_Assert(image.type() == CV_8UC3);
    
    std::cout << "About to call CPU grabCut..." << std::endl;

    // Define rectangle roughly around the central subject
    cv::Rect rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2);

    // DETAILED INTERMEDIATE COMPARISON
    printf("\n=== INTERMEDIATE COMPARISON PHASE ===\n");
    
    // Prepare CPU buffers
    cv::Mat mask_cpu(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_cpu, fgd_cpu;
    
    // Step 1: Initial mask setup - compare initial masks after rectangle initialization
    printf("\n--- STEP 1: INITIAL MASK SETUP ---\n");
    cv::grabCutWithSharedKMeans(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 0, cv::GC_INIT_WITH_RECT, 42); // 0 iterations for initial setup only, fixed seed=42
    
    // Print CPU initial mask distribution
    int cpu_counts[4] = {0,0,0,0};
    for(int y=0; y<mask_cpu.rows; ++y) {
        for(int x=0; x<mask_cpu.cols; ++x) {
            cpu_counts[mask_cpu.at<uchar>(y,x)]++;
        }
    }
    printf("CPU initial mask: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           cpu_counts[0], cpu_counts[1], cpu_counts[2], cpu_counts[3]);
    
    // Prepare Metal buffers
    cv::Mat mask_metal(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_metal, fgd_metal;
    
    // Initialize Metal mask with same rect (0 iterations) - USE DETERMINISTIC VERSION
    cv::metal::Stream stream1;
    cv::metal::grabCutWithSharedKMeans(image, mask_metal, rect, bgd_metal, fgd_metal, 0, cv::GC_INIT_WITH_RECT, 42, stream1); // Same seed=42
    stream1.commitAndWait();
    
    // Print Metal initial mask distribution  
    int metal_counts[4] = {0,0,0,0};
    for(int y=0; y<mask_metal.rows; ++y) {
        for(int x=0; x<mask_metal.cols; ++x) {
            metal_counts[mask_metal.at<uchar>(y,x)]++;
        }
    }
    printf("Metal initial mask: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           metal_counts[0], metal_counts[1], metal_counts[2], metal_counts[3]);
    
    // Step 2: Compare beta values
    printf("\n--- STEP 2: BETA CALCULATION ---\n");
    // Re-run CPU with 1 iteration to get intermediate values - USE DETERMINISTIC VERSION
    mask_cpu.setTo(cv::Scalar(cv::GC_BGD));
    cv::grabCutWithSharedKMeans(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 1, cv::GC_INIT_WITH_RECT, 42); // Fixed seed=42
    
    // Re-run Metal with 1 iteration - USE DETERMINISTIC VERSION  
    mask_metal.setTo(cv::Scalar(cv::GC_BGD));
    cv::metal::Stream stream2;
    cv::metal::grabCutWithSharedKMeans(image, mask_metal, rect, bgd_metal, fgd_metal, 1, cv::GC_INIT_WITH_RECT, 42, stream2); // Same seed=42
    stream2.commitAndWait();
    
    // Step 3: Compare final masks after 1 iteration
    printf("\n--- STEP 3: AFTER 1 ITERATION ---\n");
    cpu_counts[0] = cpu_counts[1] = cpu_counts[2] = cpu_counts[3] = 0;
    for(int y=0; y<mask_cpu.rows; ++y) {
        for(int x=0; x<mask_cpu.cols; ++x) {
            cpu_counts[mask_cpu.at<uchar>(y,x)]++;
        }
    }
    printf("CPU after 1 iter: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           cpu_counts[0], cpu_counts[1], cpu_counts[2], cpu_counts[3]);
    
    metal_counts[0] = metal_counts[1] = metal_counts[2] = metal_counts[3] = 0;
    for(int y=0; y<mask_metal.rows; ++y) {
        for(int x=0; x<mask_metal.cols; ++x) {
            metal_counts[mask_metal.at<uchar>(y,x)]++;
        }
    }
    printf("Metal after 1 iter: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           metal_counts[0], metal_counts[1], metal_counts[2], metal_counts[3]);
    
    // Step 4: Sample pixel comparisons
    printf("\n--- STEP 4: PIXEL-BY-PIXEL COMPARISON ---\n");
    int sampleCount = 0;
    for(int y=image.rows/4; y<3*image.rows/4 && sampleCount<10; y+=50) {
        for(int x=image.cols/4; x<3*image.cols/4 && sampleCount<10; x+=50) {
            uchar cpuVal = mask_cpu.at<uchar>(y,x);
            uchar metalVal = mask_metal.at<uchar>(y,x);
            printf("Sample pixel %d: CPU=%d (%s) Metal=%d (%s) %s\n", 
                   sampleCount++, cpuVal, 
                   (cpuVal==0)?"BGD":(cpuVal==1)?"FGD":(cpuVal==2)?"PR_BGD":"PR_FGD",
                   metalVal,
                   (metalVal==0)?"BGD":(metalVal==1)?"FGD":(metalVal==2)?"PR_BGD":"PR_FGD",
                   (cpuVal==metalVal)?"MATCH":"DIFFER");
        }
    }

    // Continue with full test
    std::cout << "About to call CPU grabCut..." << std::endl;
    cv::grabCutWithSharedKMeans(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 5, cv::GC_INIT_WITH_RECT, 42); // Fixed seed=42
    std::cout << "About to call Metal grabCut..." << std::endl;
    cv::metal::Stream stream3;
    cv::metal::grabCutWithSharedKMeans(image, mask_metal, rect, bgd_metal, fgd_metal, 5, cv::GC_INIT_WITH_RECT, 42, stream3); // Same seed=42
    stream3.commitAndWait();
    std::cout << "Metal grabCut completed successfully!" << std::endl;

    // Create final segmented images by applying masks
    cv::Mat result_cpu, result_metal;
    image.copyTo(result_cpu);
    image.copyTo(result_metal);
    
    // Apply CPU mask: set background pixels to black
    for (int y = 0; y < mask_cpu.rows; y++) {
        for (int x = 0; x < mask_cpu.cols; x++) {
            uchar mask_val = mask_cpu.at<uchar>(y, x);
            if (mask_val == cv::GC_BGD || mask_val == cv::GC_PR_BGD) {
                result_cpu.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // Black background
            }
        }
    }
    
    // Apply Metal mask: set background pixels to black  
    for (int y = 0; y < mask_metal.rows; y++) {
        for (int x = 0; x < mask_metal.cols; x++) {
            uchar mask_val = mask_metal.at<uchar>(y, x);
            if (mask_val == cv::GC_BGD || mask_val == cv::GC_PR_BGD) {
                result_metal.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // Black background
            }
        }
    }

    // Save all results as .jpg files
    std::string out_dir = "./outputs/";
    
    // Save original image for reference
    cv::imwrite(out_dir + "original_image.jpg", image);
    std::cout << "Saved: " << out_dir + "original_image.jpg" << std::endl;
    
    // Save masks (convert to 8-bit grayscale)
    cv::Mat mask_cpu_vis = mask_cpu * 85; // Scale 0,1,2,3 to 0,85,170,255
    cv::Mat mask_metal_vis = mask_metal * 85;
    cv::imwrite(out_dir + "cpu_mask.jpg", mask_cpu_vis);
    cv::imwrite(out_dir + "metal_mask.jpg", mask_metal_vis);
    std::cout << "Saved: " << out_dir + "cpu_mask.jpg" << std::endl;
    std::cout << "Saved: " << out_dir + "metal_mask.jpg" << std::endl;
    
    // Save segmented results
    cv::imwrite(out_dir + "cpu_result.jpg", result_cpu);
    cv::imwrite(out_dir + "metal_result.jpg", result_metal);
    std::cout << "Saved: " << out_dir + "cpu_result.jpg" << std::endl;
    std::cout << "Saved: " << out_dir + "metal_result.jpg" << std::endl;

    // Save results for manual inspection (debug images)
    saveMaskDebugImage(image, mask_cpu, out_dir + "real_cpu_result");
    saveMaskDebugImage(image, mask_metal, out_dir + "real_metal_result");

    // Compare classification counts
    cv::Mat fg_cpu = (mask_cpu == cv::GC_FGD) | (mask_cpu == cv::GC_PR_FGD);
    cv::Mat fg_metal = (mask_metal == cv::GC_FGD) | (mask_metal == cv::GC_PR_FGD);

    double cpu_fg = cv::sum(fg_cpu)[0] / 255.0;
    double metal_fg = cv::sum(fg_metal)[0] / 255.0;

    std::cout << "Real photo FG pixels CPU=" << cpu_fg << " Metal=" << metal_fg << std::endl;

    // Ensure Metal finds at least half as many foreground pixels as CPU (sanity check)
    EXPECT_GT(metal_fg, cpu_fg * 0.5);
}




// Helper function to calculate Metal GMM probabilities for a pixel
std::vector<double> calculateMetalGMMProbabilities(const cv::Vec3b& pixel_bgr, 
                                                   const cv::Mat& gmmModel, 
                                                   bool /* isForeground */ = false) {
    std::vector<double> probs(5, 0.0);
    
    // Convert pixel to normalized [0,1] range and then to [0,255] like Metal kernel
    // Metal kernel: float3 pixel_255 = float3(pixel.b, pixel.g, pixel.r) * 255.0f;
    double pixel_b = pixel_bgr[0]; // B
    double pixel_g = pixel_bgr[1]; // G  
    double pixel_r = pixel_bgr[2]; // R
    
    for (int c = 0; c < 5; c++) {
        double weight = gmmModel.ptr<float>(0)[c];
        if (weight > 0.0) {
            // Get means (stored as BGR in CPU model)
            double mean_r = gmmModel.ptr<float>(0)[5 + c*3 + 0]; // R
            double mean_g = gmmModel.ptr<float>(0)[5 + c*3 + 1]; // G
            double mean_b = gmmModel.ptr<float>(0)[5 + c*3 + 2]; // B
            
            // Calculate differences (match Metal kernel BGR ordering)
            double diff_r = pixel_r - mean_r;
            double diff_g = pixel_g - mean_g;
            double diff_b = pixel_b - mean_b;
            
            // Extract full covariance matrix elements
            double cov_00 = gmmModel.ptr<float>(0)[20 + c*9 + 0]; // rr
            double cov_01 = gmmModel.ptr<float>(0)[20 + c*9 + 1]; // rg
            double cov_02 = gmmModel.ptr<float>(0)[20 + c*9 + 2]; // rb
            double cov_11 = gmmModel.ptr<float>(0)[20 + c*9 + 4]; // gg
            double cov_12 = gmmModel.ptr<float>(0)[20 + c*9 + 5]; // gb
            double cov_22 = gmmModel.ptr<float>(0)[20 + c*9 + 8]; // bb
            
            // Calculate Mahalanobis distance using full covariance matrix
            // Match Metal kernel calculation exactly:
            // float xxa = v.x * v.x * gmm_bg[i].cov_inv_00;
            // float yyd = v.y * v.y * gmm_bg[i].cov_inv_11;
            // float zzf = v.z * v.z * gmm_bg[i].cov_inv_22;
            // float yxb = v.x * v.y * gmm_bg[i].cov_inv_01;
            // float zxc = v.z * v.x * gmm_bg[i].cov_inv_02;
            // float zye = v.z * v.y * gmm_bg[i].cov_inv_12;
            // float mahal_dist = xxa + yyd + zzf + 2.0f * (yxb + zxc + zye);
            
            // Note: Metal kernel uses v.x=diff_b, v.y=diff_g, v.z=diff_r (BGR order)
            double xxa = diff_b * diff_b * cov_00; // Actually cov_inv for bb
            double yyd = diff_g * diff_g * cov_11; // Actually cov_inv for gg  
            double zzf = diff_r * diff_r * cov_22; // Actually cov_inv for rr
            double yxb = diff_b * diff_g * cov_01; // Actually cov_inv for bg
            double zxc = diff_r * diff_b * cov_02; // Actually cov_inv for rb
            double zye = diff_r * diff_g * cov_12; // Actually cov_inv for rg
            
            double mahal_dist = xxa + yyd + zzf + 2.0 * (yxb + zxc + zye);
            
            // Clamp to prevent numerical issues (match Metal kernel)
            mahal_dist = std::min(mahal_dist, 50.0);
            
            // Calculate probability using inv_sqrt_det stored in the model
            // Note: CPU model stores regular covariance, need to compute determinant and invert
            double det = cov_00 * cov_11 * cov_22 + 2.0 * cov_01 * cov_02 * cov_12 
                        - cov_00 * cov_12 * cov_12 - cov_11 * cov_02 * cov_02 - cov_22 * cov_01 * cov_01;
            
            if (det > 0) {
                double inv_sqrt_det = 1.0 / sqrt(det);
                // Match Metal kernel: prob = gmm_bg[i].inv_sqrt_det * exp(-0.5f * mahal_dist);
                probs[c] = inv_sqrt_det * exp(-0.5 * mahal_dist);
            }
        }
    }
    
    return probs;
}

// Enhanced test with real Metal probability calculations
TEST(MetalImgproc_GrabCut, DetailedProbabilityComparison)
{
    std::cout << "\n=== DETAILED PROBABILITY COMPARISON TEST ===\n" << std::endl;
    
    // Load test image
    std::string img_path = std::string("../WID-small.jpg");  // Fixed path for build directory
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path;
    
    if (image.type() != CV_8UC3) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    }
    
    cv::Rect rect(image.cols/3, image.rows/3, image.cols/3, image.rows/3);
    const uint64_t fixed_seed = 54321;
    
    // Run both implementations for 2 iterations
    cv::Mat mask_cpu(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat mask_metal(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_cpu, fgd_cpu, bgd_metal, fgd_metal;
    
    cv::grabCutWithSharedKMeans(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 2, cv::GC_INIT_WITH_RECT, fixed_seed);
    
    cv::metal::Stream stream;
    cv::metal::grabCutWithSharedKMeans(image, mask_metal, rect, bgd_metal, fgd_metal, 2, cv::GC_INIT_WITH_RECT, fixed_seed, stream);
    stream.commitAndWait();
    
    // Select test pixels with different characteristics
    std::vector<cv::Point> test_pixels = {
        cv::Point(image.cols/2, image.rows/2),        // Center
        cv::Point(rect.x + rect.width/4, rect.y + rect.height/4),    // Inside rect
        cv::Point(rect.x + 3*rect.width/4, rect.y + 3*rect.height/4), // Inside rect
        cv::Point(50, 50),                            // Outside rect (background)
        cv::Point(image.cols-50, image.rows-50),      // Outside rect (background)
        cv::Point(rect.x + rect.width/2, rect.y + rect.height/2),    // Rect center
    };
    
    printf("\n=== PIXEL-BY-PIXEL PROBABILITY ANALYSIS ===\n");
    printf("Pixel     | Color(BGR)   | CPU_Mask | Metal_Mask | CPU_BG_Best | Metal_BG_Best | Prob_Diff | Status\n");
    printf("----------|--------------|----------|------------|-------------|---------------|-----------|--------\n");
    
    double total_prob_diff = 0.0;
    int pixels_compared = 0;
    
    for (const cv::Point& p : test_pixels) {
        if (p.x < 0 || p.x >= image.cols || p.y < 0 || p.y >= image.rows) continue;
        
        cv::Vec3b pixel_bgr = image.at<cv::Vec3b>(p);
        uchar cpu_mask = mask_cpu.at<uchar>(p);
        uchar metal_mask = mask_metal.at<uchar>(p);
        
        // Calculate background probabilities for both CPU and Metal models
        std::vector<double> cpu_bg_probs = calculateMetalGMMProbabilities(pixel_bgr, bgd_cpu, false);
        std::vector<double> metal_bg_probs = calculateMetalGMMProbabilities(pixel_bgr, bgd_metal, false);
        
        // Find best components
        int cpu_best = std::max_element(cpu_bg_probs.begin(), cpu_bg_probs.end()) - cpu_bg_probs.begin();
        int metal_best = std::max_element(metal_bg_probs.begin(), metal_bg_probs.end()) - metal_bg_probs.begin();
        
        // Calculate probability difference
        double prob_diff = 0.0;
        for (int c = 0; c < 5; c++) {
            prob_diff += std::abs(cpu_bg_probs[c] - metal_bg_probs[c]);
        }
        
        total_prob_diff += prob_diff;
        pixels_compared++;
        
        std::string status = (cpu_best == metal_best) ? "MATCH" : "DIFFER";
        if (cpu_mask != metal_mask) status += "/MASK_DIFF";
        
        printf("(%3d,%3d) | (%3d,%3d,%3d) |    %d     |     %d      |      %d      |       %d       | %8.6f  | %s\n",
               p.x, p.y, pixel_bgr[0], pixel_bgr[1], pixel_bgr[2], 
               cpu_mask, metal_mask, cpu_best, metal_best, prob_diff, status.c_str());
        
        // Detailed probability breakdown for first few pixels
        if (pixels_compared <= 3) {
            printf("  Detailed probabilities:\n");
            printf("    Component | CPU_Prob   | Metal_Prob | Diff\n");
            printf("    ----------|------------|------------|----------\n");
            for (int c = 0; c < 5; c++) {
                double diff = std::abs(cpu_bg_probs[c] - metal_bg_probs[c]);
                printf("        %d     | %9.6f  | %9.6f  | %8.6f\n", 
                       c, cpu_bg_probs[c], metal_bg_probs[c], diff);
            }
            printf("\n");
        }
    }
    
    double avg_prob_diff = total_prob_diff / pixels_compared;
    printf("\nSummary:\n");
    printf("  Total pixels compared: %d\n", pixels_compared);
    printf("  Average probability difference: %.8f\n", avg_prob_diff);
    
    // Verify that probability differences are reasonable
    EXPECT_LT(avg_prob_diff, 0.1) << "Average probability difference should be small";
    
    // Test foreground probabilities as well
    printf("\n=== FOREGROUND PROBABILITY ANALYSIS ===\n");
    
    double fg_prob_diff = 0.0;
    int fg_pixels = 0;
    
    for (const cv::Point& p : test_pixels) {
        if (p.x < 0 || p.x >= image.cols || p.y < 0 || p.y >= image.rows) continue;
        
        cv::Vec3b pixel_bgr = image.at<cv::Vec3b>(p);
        
        // Calculate foreground probabilities
        std::vector<double> cpu_fg_probs = calculateMetalGMMProbabilities(pixel_bgr, fgd_cpu, true);
        std::vector<double> metal_fg_probs = calculateMetalGMMProbabilities(pixel_bgr, fgd_metal, true);
        
        // Calculate total difference
        double pixel_fg_diff = 0.0;
        for (int c = 0; c < 5; c++) {
            pixel_fg_diff += std::abs(cpu_fg_probs[c] - metal_fg_probs[c]);
        }
        
        fg_prob_diff += pixel_fg_diff;
        fg_pixels++;
    }
    
    double avg_fg_prob_diff = fg_prob_diff / fg_pixels;
    printf("Average foreground probability difference: %.8f\n", avg_fg_prob_diff);
    
    EXPECT_LT(avg_fg_prob_diff, 0.1) << "Average foreground probability difference should be small";
    
    std::cout << "\n=== DETAILED PROBABILITY COMPARISON TEST COMPLETED ===\n" << std::endl;
}

// COMPREHENSIVE TEST: Compare Metal GrabCut iteration 1 vs Original CPU GrabCut
TEST(MetalImgproc_GrabCut, OriginalCPU_vs_Metal_Iteration1_Comparison)
{
    std::cout << "\n=== ORIGINAL CPU vs METAL GRABCUT ITERATION 1 COMPARISON ===\n" << std::endl;
    
    // Load test image
    std::string img_path = std::string("../WID-small.jpg");
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path;
    
    // Ensure proper format
    if (image.type() != CV_8UC3) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    }
    
    // Define test rectangle
    cv::Rect rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2);
    
    std::cout << "Image size: " << image.size() << ", rect: " << rect << std::endl;
    
    // ========== SHARED KMEANS CPU GRABCUT ==========
    printf("\n=== RUNNING CPU GRABCUT WITH SHARED KMEANS ===\n");
    
    // Use DETERMINISTIC initialization with fixed seed
    const uint64_t FIXED_SEED = 42;
    
    // Prepare CPU variables
    cv::Mat mask_cpu_shared(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_cpu_shared, fgd_cpu_shared;
    
    // STEP 1: Initialize mask from rectangle
    cv::Mat mask_after_rect = mask_cpu_shared.clone();
    mask_after_rect.setTo(cv::GC_BGD);
    mask_after_rect(rect).setTo(cv::GC_PR_FGD);
    
    // Count pixels after rectangle initialization
    int rect_counts[4] = {0,0,0,0};
    for(int y = 0; y < mask_after_rect.rows; ++y) {
        for(int x = 0; x < mask_after_rect.cols; ++x) {
            rect_counts[mask_after_rect.at<uchar>(y,x)]++;
        }
    }
    printf("CPU after rect init: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           rect_counts[0], rect_counts[1], rect_counts[2], rect_counts[3]);
    
    // Run CPU GrabCut with shared K-means for exactly 1 iteration
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cv::grabCutWithSharedKMeans(image, mask_cpu_shared, rect, bgd_cpu_shared, fgd_cpu_shared, 1, cv::GC_INIT_WITH_RECT, FIXED_SEED);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    
    printf("CPU GrabCut with shared K-means completed in %lld ms\n", cpu_time.count());
    
    // DETAILED ANALYSIS: Check final mask distribution and sample pixels
    printf("\n=== CPU ITERATION 1 DETAILED ANALYSIS ===\n");
    int cpu_final_counts[4] = {0,0,0,0};
    for(int y = 0; y < mask_cpu_shared.rows; ++y) {
        for(int x = 0; x < mask_cpu_shared.cols; ++x) {
            cpu_final_counts[mask_cpu_shared.at<uchar>(y,x)]++;
        }
    }
    printf("CPU final mask: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           cpu_final_counts[0], cpu_final_counts[1], cpu_final_counts[2], cpu_final_counts[3]);
    
    // Sample some pixels to see how they changed
    printf("CPU mask transitions (sample pixels):\n");
    for(int i = 0; i < 5; i++) {
        int y = rect.y + i * rect.height / 10;
        int x = rect.x + i * rect.width / 10;
        if(y < image.rows && x < image.cols) {
            uchar initial = (rect.contains(cv::Point(x,y))) ? cv::GC_PR_FGD : cv::GC_BGD;
            uchar final = mask_cpu_shared.at<uchar>(y,x);
            cv::Vec3b color = image.at<cv::Vec3b>(y,x);
            printf("  [%d,%d]: color(%d,%d,%d) %d→%d\n", x, y, color[0], color[1], color[2], initial, final);
        }
    }
    
    // ========== METAL GRABCUT WITH SHARED KMEANS ==========
    printf("\n=== RUNNING METAL GRABCUT WITH SHARED KMEANS ===\n");
    
    // Prepare Metal variables
    cv::Mat mask_metal(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    cv::Mat bgd_metal, fgd_metal;
    
    // STEP 1: Check Metal mask initialization  
    cv::Mat mask_metal_after_rect = mask_metal.clone();
    mask_metal_after_rect.setTo(cv::GC_BGD);
    mask_metal_after_rect(rect).setTo(cv::GC_PR_FGD);
    
    // Count pixels after rectangle initialization
    int metal_rect_counts[4] = {0,0,0,0};
    for(int y = 0; y < mask_metal_after_rect.rows; ++y) {
        for(int x = 0; x < mask_metal_after_rect.cols; ++x) {
            metal_rect_counts[mask_metal_after_rect.at<uchar>(y,x)]++;
        }
    }
    printf("Metal after rect init: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           metal_rect_counts[0], metal_rect_counts[1], metal_rect_counts[2], metal_rect_counts[3]);
    
    // Verify they match
    bool rect_init_matches = true;
    for(int i = 0; i < 4; i++) {
        if(rect_counts[i] != metal_rect_counts[i]) {
            printf("❌ Rectangle initialization DIFFERS at mask value %d: CPU=%d Metal=%d\n", 
                   i, rect_counts[i], metal_rect_counts[i]);
            rect_init_matches = false;
        }
    }
    if(rect_init_matches) {
        printf("✅ Rectangle initialization MATCHES between CPU and Metal\n");
    }
    
    // Run Metal GrabCut with shared K-means for exactly 1 iteration (SAME SEED!)
    cv::metal::Stream stream;
    auto start_metal = std::chrono::high_resolution_clock::now();
    cv::metal::grabCutWithSharedKMeans(image, mask_metal, rect, bgd_metal, fgd_metal, 1, cv::GC_INIT_WITH_RECT, FIXED_SEED, stream);
    stream.commitAndWait();
    auto end_metal = std::chrono::high_resolution_clock::now();
    auto metal_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_metal - start_metal);
    
    printf("Metal GrabCut with shared K-means completed in %lld ms\n", metal_time.count());
    
    // DETAILED ANALYSIS: Check final mask distribution and sample pixels
    printf("\n=== METAL ITERATION 1 DETAILED ANALYSIS ===\n");
    int metal_final_counts[4] = {0,0,0,0};
    for(int y = 0; y < mask_metal.rows; ++y) {
        for(int x = 0; x < mask_metal.cols; ++x) {
            metal_final_counts[mask_metal.at<uchar>(y,x)]++;
        }
    }
    printf("Metal final mask: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           metal_final_counts[0], metal_final_counts[1], metal_final_counts[2], metal_final_counts[3]);
    
    // Sample same pixels to compare transitions
    printf("Metal mask transitions (same sample pixels):\n");
    for(int i = 0; i < 5; i++) {
        int y = rect.y + i * rect.height / 10;
        int x = rect.x + i * rect.width / 10;
        if(y < image.rows && x < image.cols) {
            uchar initial = (rect.contains(cv::Point(x,y))) ? cv::GC_PR_FGD : cv::GC_BGD;
            uchar final = mask_metal.at<uchar>(y,x);
            cv::Vec3b color = image.at<cv::Vec3b>(y,x);
            printf("  [%d,%d]: color(%d,%d,%d) %d→%d\n", x, y, color[0], color[1], color[2], initial, final);
        }
    }
    
    // CRITICAL COMPARISON: Direct pixel-by-pixel comparison
    printf("\n=== PIXEL-BY-PIXEL MASK COMPARISON ===\n");
    int disagreement_counts[4][4] = {0}; // [cpu_value][metal_value]
    int total_disagreements = 0;
    
    for(int y = 0; y < mask_cpu_shared.rows; ++y) {
        for(int x = 0; x < mask_cpu_shared.cols; ++x) {
            uchar cpu_val = mask_cpu_shared.at<uchar>(y,x);
            uchar metal_val = mask_metal.at<uchar>(y,x);
            disagreement_counts[cpu_val][metal_val]++;
            if(cpu_val != metal_val) {
                total_disagreements++;
            }
        }
    }
    
    printf("Disagreement matrix (CPU→Metal):\n");
    printf("     CPU\\Metal  |    0   |    1   |    2   |    3   \n");
    printf("    -----------|--------|--------|--------|--------\n");
    for(int cpu = 0; cpu < 4; cpu++) {
        printf("    %d         |", cpu);
        for(int metal = 0; metal < 4; metal++) {
            printf(" %6d |", disagreement_counts[cpu][metal]);
        }
        printf("\n");
    }
    printf("Total disagreements: %d / %d pixels (%.2f%%)\n", 
           total_disagreements, mask_cpu_shared.total(), 
           100.0 * total_disagreements / mask_cpu_shared.total());
    
    // ========== ALGORITHMIC PARAMETER COMPARISON ==========
    printf("\n=== ALGORITHMIC PARAMETER COMPARISON ===\n");
    
    // Compare beta values (from debug output)
    // Note: Beta values are printed in the algorithm debug output, let's analyze them
    printf("Beta comparison: Check debug output above for CPU vs Metal beta values\n");
    
    // Compare some specific pixel probabilities using learned GMM models
    printf("\n=== PROBABILITY CALCULATION VERIFICATION ===\n");
    
    // Select a few pixels that disagreed and analyze their probabilities
    std::vector<cv::Point> disagreement_samples;
    int samples_found = 0;
    for(int y = rect.y; y < rect.y + rect.height && samples_found < 5; y += 50) {
        for(int x = rect.x; x < rect.x + rect.width && samples_found < 5; x += 50) {
            if(mask_cpu_shared.at<uchar>(y,x) != mask_metal.at<uchar>(y,x)) {
                disagreement_samples.push_back(cv::Point(x,y));
                samples_found++;
            }
        }
    }
    
    printf("Analyzing %zu disagreement pixels:\n", disagreement_samples.size());
    for(size_t i = 0; i < disagreement_samples.size(); i++) {
        cv::Point p = disagreement_samples[i];
        cv::Vec3b pixel = image.at<cv::Vec3b>(p);
        uchar cpu_mask = mask_cpu_shared.at<uchar>(p);
        uchar metal_mask = mask_metal.at<uchar>(p);
        
        printf("Pixel[%d,%d]: color(%d,%d,%d) CPU_mask=%d Metal_mask=%d\n", 
               p.x, p.y, pixel[0], pixel[1], pixel[2], cpu_mask, metal_mask);
        
        // Calculate simplified probability for this pixel with both models
        // Background probabilities
        double cpu_bg_max_prob = 0.0, metal_bg_max_prob = 0.0;
        for(int c = 0; c < 5; c++) {
            // CPU BG component
            double cpu_weight = static_cast<double>(bgd_cpu_shared.ptr<float>(0)[c]);
            cv::Vec3d cpu_mean(static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 0]), 
                              static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 1]), 
                              static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 2]));
            double cpu_dist = cv::norm(cv::Vec3d(pixel) - cpu_mean);
            double cpu_prob = cpu_weight / (1.0 + cpu_dist);
            cpu_bg_max_prob = std::max(cpu_bg_max_prob, cpu_prob);
            
            // Metal BG component
            double metal_weight = static_cast<double>(bgd_metal.ptr<float>(0)[c]);
            cv::Vec3d metal_mean(static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 0]), 
                                static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 1]), 
                                static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 2]));
            double metal_dist = cv::norm(cv::Vec3d(pixel) - metal_mean);
            double metal_prob = metal_weight / (1.0 + metal_dist);
            metal_bg_max_prob = std::max(metal_bg_max_prob, metal_prob);
        }
        
        // Foreground probabilities
        double cpu_fg_max_prob = 0.0, metal_fg_max_prob = 0.0;
        for(int c = 0; c < 5; c++) {
            // CPU FG component
            double cpu_weight = static_cast<double>(fgd_cpu_shared.ptr<float>(0)[c]);
            cv::Vec3d cpu_mean(static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 0]), 
                              static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 1]), 
                              static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 2]));
            double cpu_dist = cv::norm(cv::Vec3d(pixel) - cpu_mean);
            double cpu_prob = cpu_weight / (1.0 + cpu_dist);
            cpu_fg_max_prob = std::max(cpu_fg_max_prob, cpu_prob);
            
            // Metal FG component
            double metal_weight = static_cast<double>(fgd_metal.ptr<float>(0)[c]);
            cv::Vec3d metal_mean(static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 0]), 
                                static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 1]), 
                                static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 2]));
            double metal_dist = cv::norm(cv::Vec3d(pixel) - metal_mean);
            double metal_prob = metal_weight / (1.0 + metal_dist);
            metal_fg_max_prob = std::max(metal_fg_max_prob, metal_prob);
        }
        
        printf("  CPU: BG_prob=%.6f FG_prob=%.6f → %s\n", 
               cpu_bg_max_prob, cpu_fg_max_prob, 
               (cpu_bg_max_prob > cpu_fg_max_prob) ? "BACKGROUND" : "FOREGROUND");
        printf("  Metal: BG_prob=%.6f FG_prob=%.6f → %s\n", 
               metal_bg_max_prob, metal_fg_max_prob,
               (metal_bg_max_prob > metal_fg_max_prob) ? "BACKGROUND" : "FOREGROUND");
        printf("  Prob differences: BG=%.6f FG=%.6f\n\n", 
               std::abs(cpu_bg_max_prob - metal_bg_max_prob),
               std::abs(cpu_fg_max_prob - metal_fg_max_prob));
    }
    
    // ========== GMM PARAMETER ANALYSIS ==========
    printf("\n=== GMM PARAMETER STRUCTURE ANALYSIS ===\n");
    
    // Verify model dimensions and types
    ASSERT_EQ(bgd_cpu_shared.type(), CV_32FC1) << "CPU shared background model should be CV_32FC1";
    ASSERT_EQ(fgd_cpu_shared.type(), CV_32FC1) << "CPU shared foreground model should be CV_32FC1";
    ASSERT_EQ(bgd_metal.type(), CV_32FC1) << "Metal background model should be CV_32FC1";
    ASSERT_EQ(fgd_metal.type(), CV_32FC1) << "Metal foreground model should be CV_32FC1";
    
    ASSERT_EQ(bgd_cpu_shared.total(), 65u) << "CPU background model should have 65 elements";
    ASSERT_EQ(fgd_cpu_shared.total(), 65u) << "CPU foreground model should have 65 elements";
    ASSERT_EQ(bgd_metal.total(), 65u) << "Metal background model should have 65 elements";
    ASSERT_EQ(fgd_metal.total(), 65u) << "Metal foreground model should have 65 elements";
    
    printf("Model dimensions verified - both have 65 elements (5 weights + 15 means + 45 covariances)\n");
    
    // ========== WEIGHT COMPARISON ==========
    printf("\n=== WEIGHT COMPARISON ===\n");
    printf("Component | CPU_BG     | Metal_BG   | Abs_Diff   | CPU_FG     | Metal_FG   | Abs_Diff\n");
    printf("----------|------------|------------|------------|------------|------------|-----------\n");
    
    double max_weight_diff_bg = 0.0, max_weight_diff_fg = 0.0;
    double total_weight_cpu_bg = 0.0, total_weight_cpu_fg = 0.0;
    double total_weight_metal_bg = 0.0, total_weight_metal_fg = 0.0;
    
    for (int c = 0; c < 5; c++) {
        double cpu_bg_weight = static_cast<double>(bgd_cpu_shared.ptr<float>(0)[c]);
        double metal_bg_weight = static_cast<double>(bgd_metal.ptr<float>(0)[c]);  // Metal is CV_32FC1
        double cpu_fg_weight = static_cast<double>(fgd_cpu_shared.ptr<float>(0)[c]);
        double metal_fg_weight = static_cast<double>(fgd_metal.ptr<float>(0)[c]);  // Metal is CV_32FC1
        
        double bg_diff = std::abs(cpu_bg_weight - metal_bg_weight);
        double fg_diff = std::abs(cpu_fg_weight - metal_fg_weight);
        
        max_weight_diff_bg = std::max(max_weight_diff_bg, bg_diff);
        max_weight_diff_fg = std::max(max_weight_diff_fg, fg_diff);
        
        total_weight_cpu_bg += cpu_bg_weight;
        total_weight_cpu_fg += cpu_fg_weight;
        total_weight_metal_bg += metal_bg_weight;
        total_weight_metal_fg += metal_fg_weight;
        
        printf("    %d     | %9.6f  | %9.6f  | %9.6f  | %9.6f  | %9.6f  | %9.6f\n",
               c, cpu_bg_weight, metal_bg_weight, bg_diff, 
               cpu_fg_weight, metal_fg_weight, fg_diff);
        
        // Check that weights are positive and reasonable
        EXPECT_GT(cpu_bg_weight, 0.0) << "CPU background weight " << c << " should be positive";
        EXPECT_GT(metal_bg_weight, 0.0) << "Metal background weight " << c << " should be positive";
        EXPECT_GT(cpu_fg_weight, 0.0) << "CPU foreground weight " << c << " should be positive";
        EXPECT_GT(metal_fg_weight, 0.0) << "Metal foreground weight " << c << " should be positive";
        
        EXPECT_LT(cpu_bg_weight, 1.0) << "CPU background weight " << c << " should be < 1.0";
        EXPECT_LT(metal_bg_weight, 1.0) << "Metal background weight " << c << " should be < 1.0";
        EXPECT_LT(cpu_fg_weight, 1.0) << "CPU foreground weight " << c << " should be < 1.0";
        EXPECT_LT(metal_fg_weight, 1.0) << "Metal foreground weight " << c << " should be < 1.0";
    }
    
    printf("\nWeight Summary:\n");
    printf("  Total CPU BG weights: %.6f (should ≈ 1.0)\n", total_weight_cpu_bg);
    printf("  Total Metal BG weights: %.6f (should ≈ 1.0)\n", total_weight_metal_bg);
    printf("  Total CPU FG weights: %.6f (should ≈ 1.0)\n", total_weight_cpu_fg);
    printf("  Total Metal FG weights: %.6f (should ≈ 1.0)\n", total_weight_metal_fg);
    printf("  Max BG weight difference: %.6f\n", max_weight_diff_bg);
    printf("  Max FG weight difference: %.6f\n", max_weight_diff_fg);
    
    // Weights should sum to approximately 1.0
    EXPECT_NEAR(total_weight_cpu_bg, 1.0, 0.01) << "CPU background weights should sum to ~1.0";
    EXPECT_NEAR(total_weight_metal_bg, 1.0, 0.01) << "Metal background weights should sum to ~1.0";
    EXPECT_NEAR(total_weight_cpu_fg, 1.0, 0.01) << "CPU foreground weights should sum to ~1.0";
    EXPECT_NEAR(total_weight_metal_fg, 1.0, 0.01) << "Metal foreground weights should sum to ~1.0";
    
    // ========== MEAN COMPARISON ==========
    printf("\n=== MEAN COMPARISON ===\n");
    printf("Comp | CPU_BG_Mean        | Metal_BG_Mean      | Euclidean_Diff | CPU_FG_Mean        | Metal_FG_Mean      | Euclidean_Diff\n");
    printf("-----|--------------------|--------------------|----------------|--------------------|--------------------|---------------\n");
    
    double max_mean_diff_bg = 0.0, max_mean_diff_fg = 0.0;
    
    for (int c = 0; c < 5; c++) {
        cv::Vec3d cpu_bg_mean(static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 0]), 
                              static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 1]), 
                              static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 2]));
        cv::Vec3d metal_bg_mean(static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 0]), 
                                static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 1]), 
                                static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 2]));
        cv::Vec3d cpu_fg_mean(static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 0]), 
                              static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 1]), 
                              static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 2]));
        cv::Vec3d metal_fg_mean(static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 0]), 
                                static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 1]), 
                                static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 2]));
        
        double bg_mean_diff = cv::norm(cpu_bg_mean - metal_bg_mean);
        double fg_mean_diff = cv::norm(cpu_fg_mean - metal_fg_mean);
        
        max_mean_diff_bg = std::max(max_mean_diff_bg, bg_mean_diff);
        max_mean_diff_fg = std::max(max_mean_diff_fg, fg_mean_diff);
        
        printf("  %d  | (%5.1f,%5.1f,%5.1f) | (%5.1f,%5.1f,%5.1f) | %13.4f  | (%5.1f,%5.1f,%5.1f) | (%5.1f,%5.1f,%5.1f) | %13.4f\n",
               c, cpu_bg_mean[0], cpu_bg_mean[1], cpu_bg_mean[2], 
               metal_bg_mean[0], metal_bg_mean[1], metal_bg_mean[2], bg_mean_diff,
               cpu_fg_mean[0], cpu_fg_mean[1], cpu_fg_mean[2],
               metal_fg_mean[0], metal_fg_mean[1], metal_fg_mean[2], fg_mean_diff);
        
        // Check that means are in valid color range [0, 255]
        for (int ch = 0; ch < 3; ch++) {
            EXPECT_GE(cpu_bg_mean[ch], 0.0) << "CPU BG mean should be >= 0";
            EXPECT_LE(cpu_bg_mean[ch], 255.0) << "CPU BG mean should be <= 255";
            EXPECT_GE(metal_bg_mean[ch], 0.0) << "Metal BG mean should be >= 0";
            EXPECT_LE(metal_bg_mean[ch], 255.0) << "Metal BG mean should be <= 255";
            EXPECT_GE(cpu_fg_mean[ch], 0.0) << "CPU FG mean should be >= 0";
            EXPECT_LE(cpu_fg_mean[ch], 255.0) << "CPU FG mean should be <= 255";
            EXPECT_GE(metal_fg_mean[ch], 0.0) << "Metal FG mean should be >= 0";
            EXPECT_LE(metal_fg_mean[ch], 255.0) << "Metal FG mean should be <= 255";
        }
    }
    
    printf("\nMean Summary:\n");
    printf("  Max BG mean difference: %.4f\n", max_mean_diff_bg);
    printf("  Max FG mean difference: %.4f\n", max_mean_diff_fg);
    
    // ========== COVARIANCE ANALYSIS ==========
    printf("\n=== COVARIANCE MATRIX ANALYSIS ===\n");
    printf("Analyzing covariance matrix properties (determinant, trace, condition number)\n");
    printf("Comp | CPU_BG_Det  | Metal_BG_Det | CPU_BG_Trace | Metal_BG_Trace | CPU_FG_Det  | Metal_FG_Det | CPU_FG_Trace | Metal_FG_Trace\n");
    printf("-----|-------------|--------------|--------------|----------------|-------------|--------------|--------------|---------------\n");
    
    for (int c = 0; c < 5; c++) {
        // Extract covariance matrices (3x3 each)
        cv::Matx33d cpu_bg_cov, cpu_fg_cov;
        cv::Matx33d metal_bg_cov, metal_fg_cov;  // Both are CV_32FC1 now
        
        // CPU covariances (CV_32FC1)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cpu_bg_cov(i,j) = static_cast<double>(bgd_cpu_shared.ptr<float>(0)[20 + c*9 + i*3 + j]);
                cpu_fg_cov(i,j) = static_cast<double>(fgd_cpu_shared.ptr<float>(0)[20 + c*9 + i*3 + j]);
            }
        }
        
        // Metal covariances (CV_32FC1)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                metal_bg_cov(i,j) = static_cast<double>(bgd_metal.ptr<float>(0)[20 + c*9 + i*3 + j]);
                metal_fg_cov(i,j) = static_cast<double>(fgd_metal.ptr<float>(0)[20 + c*9 + i*3 + j]);
            }
        }
        
        // Calculate determinants and traces
        double cpu_bg_det = cv::determinant(cpu_bg_cov);
        double metal_bg_det = cv::determinant(metal_bg_cov);
        double cpu_fg_det = cv::determinant(cpu_fg_cov);
        double metal_fg_det = cv::determinant(metal_fg_cov);
        
        // Fix: For Matx types, cv::trace returns a double directly
        double cpu_bg_trace = cv::trace(cpu_bg_cov);
        double metal_bg_trace = cv::trace(metal_bg_cov);
        double cpu_fg_trace = cv::trace(cpu_fg_cov);
        double metal_fg_trace = cv::trace(metal_fg_cov);
        
        printf("  %d  | %10.2e  | %10.2e   | %11.2f  | %13.2f   | %10.2e  | %10.2e   | %11.2f  | %13.2f\n",
               c, cpu_bg_det, metal_bg_det, cpu_bg_trace, metal_bg_trace,
               cpu_fg_det, metal_fg_det, cpu_fg_trace, metal_fg_trace);
        
        // Check that covariance matrices are positive definite (determinant > 0)
        EXPECT_GT(cpu_bg_det, 0.0) << "CPU background covariance determinant should be positive";
        EXPECT_GT(metal_bg_det, 0.0) << "Metal background covariance determinant should be positive";
        EXPECT_GT(cpu_fg_det, 0.0) << "CPU foreground covariance determinant should be positive";
        EXPECT_GT(metal_fg_det, 0.0) << "Metal foreground covariance determinant should be positive";
        
        // Traces should be positive (sum of eigenvalues)
        EXPECT_GT(cpu_bg_trace, 0.0) << "CPU background covariance trace should be positive";
        EXPECT_GT(metal_bg_trace, 0.0) << "Metal background covariance trace should be positive";
        EXPECT_GT(cpu_fg_trace, 0.0) << "CPU foreground covariance trace should be positive";
        EXPECT_GT(metal_fg_trace, 0.0) << "Metal foreground covariance trace should be positive";
    }
    
    // ========== PIXEL PROBABILITY COMPARISON ==========
    printf("\n=== PIXEL PROBABILITY COMPARISON ===\n");
    
    // Select representative pixels for detailed probability analysis
    std::vector<cv::Point> test_pixels = {
        cv::Point(image.cols/2, image.rows/2),        // Image center
        cv::Point(rect.x + rect.width/4, rect.y + rect.height/4),      // Inside rect
        cv::Point(rect.x + 3*rect.width/4, rect.y + 3*rect.height/4),  // Inside rect
        cv::Point(50, 50),                            // Outside rect (background)
        cv::Point(image.cols-50, image.rows-50),      // Outside rect (background)
        cv::Point(rect.x + rect.width/2, rect.y + rect.height/2),      // Rect center
    };
    
    printf("Comparing probability calculations for %zu representative pixels\n", test_pixels.size());
    printf("Pixel     | Color(BGR)   | CPU_Mask | Metal_Mask | CPU_BG_MaxProb | Metal_BG_MaxProb | CPU_FG_MaxProb | Metal_FG_MaxProb\n");
    printf("----------|--------------|----------|------------|----------------|------------------|----------------|------------------\n");
    
    double total_bg_prob_diff = 0.0, total_fg_prob_diff = 0.0;
    int prob_comparisons = 0;
    
    for (const cv::Point& p : test_pixels) {
        if (p.x < 0 || p.x >= image.cols || p.y < 0 || p.y >= image.rows) continue;
        
        cv::Vec3b pixel_bgr = image.at<cv::Vec3b>(p);
        uchar cpu_mask = mask_cpu_shared.at<uchar>(p);
        uchar metal_mask = mask_metal.at<uchar>(p);
        
        // Calculate probabilities manually for both implementations
        // Note: This is a simplified probability calculation for demonstration
        double cpu_bg_max_prob = 0.0, metal_bg_max_prob = 0.0;
        double cpu_fg_max_prob = 0.0, metal_fg_max_prob = 0.0;
        
        // Find the component with highest probability for each implementation
        for (int c = 0; c < 5; c++) {
            // Simplified probability calculation (not full Gaussian, just for comparison)
            cv::Vec3d cpu_bg_mean(static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 0]), 
                                  static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 1]), 
                                  static_cast<double>(bgd_cpu_shared.ptr<float>(0)[5 + c*3 + 2]));
            cv::Vec3d metal_bg_mean(static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 0]), 
                                    static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 1]), 
                                    static_cast<double>(bgd_metal.ptr<float>(0)[5 + c*3 + 2]));
            cv::Vec3d cpu_fg_mean(static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 0]), 
                                  static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 1]), 
                                  static_cast<double>(fgd_cpu_shared.ptr<float>(0)[5 + c*3 + 2]));
            cv::Vec3d metal_fg_mean(static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 0]), 
                                    static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 1]), 
                                    static_cast<double>(fgd_metal.ptr<float>(0)[5 + c*3 + 2]));
            
            double cpu_bg_dist = cv::norm(cv::Vec3d(pixel_bgr) - cpu_bg_mean);
            double metal_bg_dist = cv::norm(cv::Vec3d(pixel_bgr) - metal_bg_mean);
            
            double cpu_bg_weight = static_cast<double>(bgd_cpu_shared.ptr<float>(0)[c]);
            double metal_bg_weight = static_cast<double>(bgd_metal.ptr<float>(0)[c]);
            
            // Simplified probability (weight / distance)
            double cpu_bg_prob = cpu_bg_weight / (1.0 + cpu_bg_dist);
            double metal_bg_prob = metal_bg_weight / (1.0 + metal_bg_dist);
            
            cpu_bg_max_prob = std::max(cpu_bg_max_prob, cpu_bg_prob);
            metal_bg_max_prob = std::max(metal_bg_max_prob, metal_bg_prob);
            
            double cpu_fg_dist = cv::norm(cv::Vec3d(pixel_bgr) - cpu_fg_mean);
            double metal_fg_dist = cv::norm(cv::Vec3d(pixel_bgr) - metal_fg_mean);
            
            double cpu_fg_weight = static_cast<double>(fgd_cpu_shared.ptr<float>(0)[c]);
            double metal_fg_weight = static_cast<double>(fgd_metal.ptr<float>(0)[c]);
            
            double cpu_fg_prob = cpu_fg_weight / (1.0 + cpu_fg_dist);
            double metal_fg_prob = metal_fg_weight / (1.0 + metal_fg_dist);
            
            cpu_fg_max_prob = std::max(cpu_fg_max_prob, cpu_fg_prob);
            metal_fg_max_prob = std::max(metal_fg_max_prob, metal_fg_prob);
        }
        
        total_bg_prob_diff += std::abs(cpu_bg_max_prob - metal_bg_max_prob);
        total_fg_prob_diff += std::abs(cpu_fg_max_prob - metal_fg_max_prob);
        prob_comparisons++;
        
        printf("(%3d,%3d) | (%3d,%3d,%3d) |    %d     |     %d      | %13.6f  | %15.6f  | %13.6f  | %15.6f\n",
               p.x, p.y, pixel_bgr[0], pixel_bgr[1], pixel_bgr[2], 
               cpu_mask, metal_mask, cpu_bg_max_prob, metal_bg_max_prob, 
               cpu_fg_max_prob, metal_fg_max_prob);
    }
    
    double avg_bg_prob_diff = total_bg_prob_diff / prob_comparisons;
    double avg_fg_prob_diff = total_fg_prob_diff / prob_comparisons;
    
    printf("\nProbability Summary:\n");
    printf("  Average BG probability difference: %.8f\n", avg_bg_prob_diff);
    printf("  Average FG probability difference: %.8f\n", avg_fg_prob_diff);
    
    // ========== MASK DISTRIBUTION COMPARISON ==========
    printf("\n=== MASK DISTRIBUTION COMPARISON ===\n");
    
    int cpu_counts[4] = {0,0,0,0};
    int metal_counts[4] = {0,0,0,0};
    
    for(int y = 0; y < mask_cpu_shared.rows; ++y) {
        for(int x = 0; x < mask_cpu_shared.cols; ++x) {
            cpu_counts[mask_cpu_shared.at<uchar>(y,x)]++;
            metal_counts[mask_metal.at<uchar>(y,x)]++;
        }
    }
    
    printf("Final mask distribution after 1 iteration:\n");
    printf("        | BGD(0)   | FGD(1)   | PR_BGD(2) | PR_FGD(3) | Total\n");
    printf("--------|----------|----------|-----------|-----------|--------\n");
    printf("CPU     | %8d | %8d | %9d | %9d | %8d\n", 
           cpu_counts[0], cpu_counts[1], cpu_counts[2], cpu_counts[3], 
           cpu_counts[0] + cpu_counts[1] + cpu_counts[2] + cpu_counts[3]);
    printf("Metal   | %8d | %8d | %9d | %9d | %8d\n", 
           metal_counts[0], metal_counts[1], metal_counts[2], metal_counts[3],
           metal_counts[0] + metal_counts[1] + metal_counts[2] + metal_counts[3]);
    
    // Calculate percentage differences
    int total_pixels = image.total();
    for (int i = 0; i < 4; i++) {
        int diff = std::abs(cpu_counts[i] - metal_counts[i]);
        double diff_percent = 100.0 * diff / total_pixels;
        printf("Mask value %d: difference = %d pixels (%.2f%%)\n", i, diff, diff_percent);
        
        // For different K-means initialization, expect larger differences but they should still be reasonable
        EXPECT_LT(diff_percent, 30.0) << "Mask distribution difference should be < 30% for value " << i;
    }
    
    // ========== PERFORMANCE COMPARISON ==========
    printf("\n=== PERFORMANCE COMPARISON ===\n");
    printf("Original CPU GrabCut time: %lld ms\n", cpu_time.count());
    printf("Metal GrabCut time: %lld ms\n", metal_time.count());
    
    if (metal_time.count() > 0) {
        double speedup = static_cast<double>(cpu_time.count()) / metal_time.count();
        printf("Metal speedup: %.2fx\n", speedup);
        
        // Metal should be reasonably fast (not necessarily faster due to small image size and overhead)
        EXPECT_LT(metal_time.count(), 10000) << "Metal GrabCut should complete within 10 seconds";
    }
    
    // ========== ALGORITHM CONVERGENCE PROPERTIES ==========
    printf("\n=== ALGORITHM CONVERGENCE PROPERTIES ===\n");
    
    // Both algorithms should produce reasonable segmentations
    int cpu_fg_pixels = cpu_counts[GC_FGD] + cpu_counts[GC_PR_FGD];
    int metal_fg_pixels = metal_counts[GC_FGD] + metal_counts[GC_PR_FGD];
    
    printf("Foreground pixels: CPU=%d (%.1f%%), Metal=%d (%.1f%%)\n", 
           cpu_fg_pixels, 100.0 * cpu_fg_pixels / total_pixels,
           metal_fg_pixels, 100.0 * metal_fg_pixels / total_pixels);
    
    // Both should find significant foreground regions
    EXPECT_GT(cpu_fg_pixels, total_pixels * 0.05) << "CPU should find at least 5% foreground";
    EXPECT_GT(metal_fg_pixels, total_pixels * 0.05) << "Metal should find at least 5% foreground";
    EXPECT_LT(cpu_fg_pixels, total_pixels * 0.95) << "CPU should not classify > 95% as foreground";
    EXPECT_LT(metal_fg_pixels, total_pixels * 0.95) << "Metal should not classify > 95% as foreground";
    
    std::cout << "\n=== ORIGINAL CPU vs METAL GRABCUT ITERATION 1 COMPARISON COMPLETED ===\n" << std::endl;

    // ========== SAVE DEBUG VISUALIZATIONS ==========
    {
        std::string out_dir = "./outputs/";

        // Save original for reference in this comparison run
        cv::imwrite(out_dir + "comparison_original.jpg", image);

        // Re-use the common helper for overlay + raw mask saving
        saveMaskDebugImage(image, mask_cpu_shared, out_dir + "comparison_cpu_iter1");
        saveMaskDebugImage(image, mask_metal, out_dir + "comparison_metal_iter1");

        printf("Saved debug visualizations to '%s'.\n", out_dir.c_str());
    }
}

#ifdef USE_METAL_GRAPHCUT

TEST(MetalImgproc_GrabCut_FullGPU, SmallSynthetic)
{
    Size sz(128,128);
    Mat image(sz, CV_8UC3, Scalar(120,120,120));
    circle(image, Point(64,64), 30, Scalar(0,0,200), -1);

    Mat mask = Mat::zeros(sz, CV_8UC1);
    Rect rect(32,32,64,64);

    Mat bgdM, fgdM;
    cv::metal::grabCut(image, mask, rect, bgdM, fgdM, 5, GC_INIT_WITH_RECT);

    // Ensure some foreground detected
    int fgPixels = countNonZero((mask == GC_FGD) | (mask == GC_PR_FGD));
    EXPECT_GT(fgPixels, 1000);
}

#endif // USE_METAL_GRAPHCUT

}} // namespace opencv_test

#endif // HAVE_METAL 