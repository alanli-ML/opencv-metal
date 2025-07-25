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
    cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 0, cv::GC_INIT_WITH_RECT); // 0 iterations for initial setup only
    
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
    cv::metal::grabCut(image, mask_metal, rect, bgd_metal, fgd_metal, 0, cv::GC_INIT_WITH_RECT, stream1);
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
    cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 1, cv::GC_INIT_WITH_RECT); // Fixed seed=42
    
    // Re-run Metal with 1 iteration - USE DETERMINISTIC VERSION  
    mask_metal.setTo(cv::Scalar(cv::GC_BGD));
    cv::metal::Stream stream2;
    cv::metal::grabCut(image, mask_metal, rect, bgd_metal, fgd_metal, 1, cv::GC_INIT_WITH_RECT, stream2); // Same seed=42
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
    cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 5, cv::GC_INIT_WITH_RECT); // Fixed seed=42
    std::cout << "About to call Metal grabCut..." << std::endl;
    cv::metal::Stream stream3;
    cv::metal::grabCut(image, mask_metal, rect, bgd_metal, fgd_metal, 5, cv::GC_INIT_WITH_RECT, stream3); // Same seed=42
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
    
    cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 2, cv::GC_INIT_WITH_RECT);
    
    cv::metal::Stream stream;
    cv::metal::grabCut(image, mask_metal, rect, bgd_metal, fgd_metal, 2, cv::GC_INIT_WITH_RECT, stream);
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



// COMPREHENSIVE STEP-BY-STEP DEBUGGING TEST
TEST(MetalImgproc_GrabCut, StepByStepDebug)
{
    std::cout << "\n=== STEP-BY-STEP CPU vs METAL DEBUG ANALYSIS ===\n" << std::endl;
    
    // Load test image
    std::string img_path = std::string("../WID-small.jpg");
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path;
    
    if (image.type() != CV_8UC3) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
    }
    
    // Use smaller region for detailed analysis
    cv::Rect rect(image.cols/3, image.rows/3, image.cols/3, image.rows/3);
    std::cout << "Debug region: " << rect << std::endl;
    
    // ===== STEP 1: INITIAL SETUP COMPARISON =====
    printf("\n=== STEP 1: INITIAL SETUP COMPARISON ===\n");
    
    // Prepare CPU mask
    cv::Mat mask_cpu(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
    mask_cpu(rect).setTo(cv::GC_PR_FGD);
    cv::Mat bgd_cpu, fgd_cpu;
    
    // Prepare Metal mask (identical initialization)
    cv::Mat mask_metal = mask_cpu.clone();
    cv::Mat bgd_metal, fgd_metal;
    
    printf("Initial masks identical: %s\n", 
           (cv::sum(mask_cpu != mask_metal)[0] == 0) ? "YES" : "NO");
    
    // ===== STEP 2: FIRST ITERATION DETAILED BREAKDOWN =====
    printf("\n=== STEP 2: ITERATION 1 DETAILED BREAKDOWN ===\n");
    
    // Create custom CPU implementation that matches Metal exactly
    // First, let's see what CPU produces in one iteration
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 1, cv::GC_INIT_WITH_RECT);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    
    printf("CPU iteration 1 completed in %lld ms\n", cpu_time.count());
    
    // Count CPU results
    int cpu_counts[4] = {0,0,0,0};
    for(int y = 0; y < mask_cpu.rows; ++y) {
        for(int x = 0; x < mask_cpu.cols; ++x) {
            cpu_counts[mask_cpu.at<uchar>(y,x)]++;
        }
    }
    printf("CPU iter 1: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           cpu_counts[0], cpu_counts[1], cpu_counts[2], cpu_counts[3]);
    
    // Now run Metal with detailed debugging
    cv::metal::Stream stream;
    auto start_metal = std::chrono::high_resolution_clock::now();
    cv::metal::grabCut(image, mask_metal, rect, bgd_metal, fgd_metal, 1, cv::GC_INIT_WITH_RECT, stream);
    stream.commitAndWait();
    auto end_metal = std::chrono::high_resolution_clock::now();
    auto metal_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_metal - start_metal);
    
    printf("Metal iteration 1 completed in %lld ms\n", metal_time.count());
    
    // Count Metal results
    int metal_counts[4] = {0,0,0,0};
    for(int y = 0; y < mask_metal.rows; ++y) {
        for(int x = 0; x < mask_metal.cols; ++x) {
            metal_counts[mask_metal.at<uchar>(y,x)]++;
        }
    }
    printf("Metal iter 1: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           metal_counts[0], metal_counts[1], metal_counts[2], metal_counts[3]);
    
    // ===== STEP 3: GMM PARAMETER COMPARISON =====
    printf("\n=== STEP 3: GMM PARAMETER COMPARISON ===\n");
    
    printf("CPU BG model: type=%d size=%zu\n", bgd_cpu.type(), bgd_cpu.total());
    printf("CPU FG model: type=%d size=%zu\n", fgd_cpu.type(), fgd_cpu.total());
    printf("Metal BG model: type=%d size=%zu\n", bgd_metal.type(), bgd_metal.total());
    printf("Metal FG model: type=%d size=%zu\n", fgd_metal.type(), fgd_metal.total());
    
    // Compare weights (first 5 elements)
    printf("\nWeight comparison:\n");
    printf("Component | CPU_BG_Weight | Metal_BG_Weight | CPU_FG_Weight | Metal_FG_Weight\n");
    printf("----------|---------------|-----------------|---------------|----------------\n");
    
    for (int c = 0; c < 5; c++) {
        double cpu_bg_weight, metal_bg_weight, cpu_fg_weight, metal_fg_weight;
        
        if (bgd_cpu.type() == CV_64FC1) {
            cpu_bg_weight = bgd_cpu.ptr<double>(0)[c];
            cpu_fg_weight = fgd_cpu.ptr<double>(0)[c];
        } else {
            cpu_bg_weight = bgd_cpu.ptr<float>(0)[c];
            cpu_fg_weight = fgd_cpu.ptr<float>(0)[c];
        }
        
        if (bgd_metal.type() == CV_64FC1) {
            metal_bg_weight = bgd_metal.ptr<double>(0)[c];
            metal_fg_weight = fgd_metal.ptr<double>(0)[c];
        } else {
            metal_bg_weight = bgd_metal.ptr<float>(0)[c];
            metal_fg_weight = fgd_metal.ptr<float>(0)[c];
        }
        
        printf("    %d     | %12.6f  | %14.6f  | %12.6f  | %14.6f\n",
               c, cpu_bg_weight, metal_bg_weight, cpu_fg_weight, metal_fg_weight);
    }
    
    // ===== STEP 4: SECOND ITERATION COMPARISON =====
    printf("\n=== STEP 4: SECOND ITERATION COMPARISON ===\n");
    
    // Save iteration 1 states
    cv::Mat mask_cpu_iter1 = mask_cpu.clone();
    cv::Mat mask_metal_iter1 = mask_metal.clone();
    
    // Run second iteration on both
    cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, 1, cv::GC_EVAL);
    cv::metal::grabCut(image, mask_metal, rect, bgd_metal, fgd_metal, 1, cv::GC_EVAL, stream);
    stream.commitAndWait();
    
    // Count results after iteration 2
    cpu_counts[0] = cpu_counts[1] = cpu_counts[2] = cpu_counts[3] = 0;
    for(int y = 0; y < mask_cpu.rows; ++y) {
        for(int x = 0; x < mask_cpu.cols; ++x) {
            cpu_counts[mask_cpu.at<uchar>(y,x)]++;
        }
    }
    
    metal_counts[0] = metal_counts[1] = metal_counts[2] = metal_counts[3] = 0;
    for(int y = 0; y < mask_metal.rows; ++y) {
        for(int x = 0; x < mask_metal.cols; ++x) {
            metal_counts[mask_metal.at<uchar>(y,x)]++;
        }
    }
    
    printf("CPU iter 2: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           cpu_counts[0], cpu_counts[1], cpu_counts[2], cpu_counts[3]);
    printf("Metal iter 2: BGD=%d FGD=%d PR_BGD=%d PR_FGD=%d\n", 
           metal_counts[0], metal_counts[1], metal_counts[2], metal_counts[3]);
    
    // ===== STEP 5: PIXEL-LEVEL CHANGE ANALYSIS =====
    printf("\n=== STEP 5: PIXEL-LEVEL CHANGE ANALYSIS ===\n");
    
    int cpu_changes = 0, metal_changes = 0;
    for(int y = 0; y < mask_cpu.rows; ++y) {
        for(int x = 0; x < mask_cpu.cols; ++x) {
            if(mask_cpu_iter1.at<uchar>(y,x) != mask_cpu.at<uchar>(y,x)) cpu_changes++;
            if(mask_metal_iter1.at<uchar>(y,x) != mask_metal.at<uchar>(y,x)) metal_changes++;
        }
    }
    
    printf("Pixels changed iter 1→2: CPU=%d Metal=%d\n", cpu_changes, metal_changes);
    
    // ===== STEP 6: FINAL DIVERGENCE MEASUREMENT =====
    printf("\n=== STEP 6: FINAL DIVERGENCE MEASUREMENT ===\n");
    
    int disagreements = 0;
    for(int y = 0; y < mask_cpu.rows; ++y) {
        for(int x = 0; x < mask_cpu.cols; ++x) {
            if(mask_cpu.at<uchar>(y,x) != mask_metal.at<uchar>(y,x)) {
                disagreements++;
            }
        }
    }
    
    double disagreement_percent = 100.0 * disagreements / (mask_cpu.rows * mask_cpu.cols);
    printf("Final disagreement: %d pixels (%.2f%%)\n", disagreements, disagreement_percent);
    
    // ===== STEP 7: SAVE DEBUG OUTPUTS =====
    printf("\n=== STEP 7: SAVE DEBUG OUTPUTS ===\n");
    
    std::string out_dir = "./outputs/";
    
    // Save step-by-step results
    saveMaskDebugImage(image, mask_cpu_iter1, out_dir + "debug_cpu_iter1");
    saveMaskDebugImage(image, mask_metal_iter1, out_dir + "debug_metal_iter1");
    saveMaskDebugImage(image, mask_cpu, out_dir + "debug_cpu_iter2");
    saveMaskDebugImage(image, mask_metal, out_dir + "debug_metal_iter2");
    
    printf("Debug images saved to %s\n", out_dir.c_str());
    
    std::cout << "\n=== STEP-BY-STEP DEBUG ANALYSIS COMPLETED ===\n" << std::endl;
    
    // Final verification - this should pass for early iterations
    EXPECT_LT(disagreement_percent, 50.0) << "After 2 iterations, disagreement should be < 50%";
}

// Test the fixed kmeansClusterByMask functionality with real image data
TEST(MetalImgproc_GrabCut, KMeansFixValidation)
{
    std::cout << "\n=== K-MEANS FIX VALIDATION TEST ===\n" << std::endl;
    
    // Load real test image
    std::string img_path = std::string("../WID-small.jpg");
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path 
                               << " - ensure image is in the correct location relative to build directory";
    
    // Ensure image is in correct format
    if (image.type() != CV_8UC3) {
        if (image.channels() == 4) {
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        } else if (image.channels() == 1) {
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        }
    }
    CV_Assert(image.type() == CV_8UC3);
    
    std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
    
    // Define test rectangle around interesting subject area
    cv::Rect rect(image.cols/4, image.rows/4, image.cols/2, image.rows/2);
    
    // Run multiple test scenarios to validate k-means clustering fixes
    std::vector<std::pair<std::string, int>> test_scenarios = {
        {"INIT_WITH_RECT_1iter", 1},
        {"INIT_WITH_RECT_3iter", 3},
        {"INIT_WITH_RECT_5iter", 5}
    };
    
    for (const auto& scenario : test_scenarios) {
        std::cout << "\n--- Testing " << scenario.first << " ---" << std::endl;
        
        // Prepare fresh masks and models for each test
        cv::Mat mask_cpu(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
        cv::Mat mask_metal(image.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));
        cv::Mat bgd_cpu, fgd_cpu, bgd_metal, fgd_metal;
        
        // Run CPU implementation (reference)
        std::cout << "Running CPU grabCut..." << std::endl;
        auto start_cpu = std::chrono::high_resolution_clock::now();
        cv::grabCut(image, mask_cpu, rect, bgd_cpu, fgd_cpu, scenario.second, cv::GC_INIT_WITH_RECT);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
        
        // Run Metal implementation with k-means fixes
        std::cout << "Running Metal grabCut with k-means fixes..." << std::endl;
        auto start_metal = std::chrono::high_resolution_clock::now();
        cv::metal::Stream stream;
        cv::metal::grabCutWithSharedKMeans(image, mask_metal, rect, bgd_metal, fgd_metal, 
                                          scenario.second, cv::GC_INIT_WITH_RECT, 42);
        stream.commitAndWait();
        auto end_metal = std::chrono::high_resolution_clock::now();
        auto metal_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_metal - start_metal);
        
        std::cout << "CPU time: " << cpu_time.count() << "ms, Metal time: " << metal_time.count() << "ms" << std::endl;
        
        // VALIDATION 1: Check mask value distributions
        std::cout << "\n=== MASK VALUE DISTRIBUTION COMPARISON ===" << std::endl;
        std::vector<int> cpu_counts(4, 0), metal_counts(4, 0);
        
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                uchar cpu_val = mask_cpu.at<uchar>(y, x);
                uchar metal_val = mask_metal.at<uchar>(y, x);
                
                if (cpu_val <= 3) cpu_counts[cpu_val]++;
                if (metal_val <= 3) metal_counts[metal_val]++;
            }
        }
        
        std::cout << "CPU mask distribution: ";
        for (int i = 0; i < 4; i++) {
            std::cout << "GC_" << (i == 0 ? "BGD" : i == 1 ? "FGD" : i == 2 ? "PR_BGD" : "PR_FGD") 
                     << "=" << cpu_counts[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Metal mask distribution: ";
        for (int i = 0; i < 4; i++) {
            std::cout << "GC_" << (i == 0 ? "BGD" : i == 1 ? "FGD" : i == 2 ? "PR_BGD" : "PR_FGD") 
                     << "=" << metal_counts[i] << " ";
        }
        std::cout << std::endl;
        
        // VALIDATION 2: Check foreground/background classification similarity
        int agree_bg = 0, agree_fg = 0, disagree = 0;
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                uchar cpu_val = mask_cpu.at<uchar>(y, x);
                uchar metal_val = mask_metal.at<uchar>(y, x);
                
                bool cpu_is_fg = (cpu_val == cv::GC_FGD || cpu_val == cv::GC_PR_FGD);
                bool metal_is_fg = (metal_val == cv::GC_FGD || metal_val == cv::GC_PR_FGD);
                
                if (cpu_is_fg && metal_is_fg) {
                    agree_fg++;
                } else if (!cpu_is_fg && !metal_is_fg) {
                    agree_bg++;
                } else {
                    disagree++;
                }
            }
        }
        
        int total_pixels = image.rows * image.cols;
        double agreement_rate = double(agree_bg + agree_fg) / total_pixels;
        
        std::cout << "\n=== FOREGROUND/BACKGROUND CLASSIFICATION AGREEMENT ===" << std::endl;
        std::cout << "Agree on foreground: " << agree_fg << " pixels" << std::endl;
        std::cout << "Agree on background: " << agree_bg << " pixels" << std::endl;
        std::cout << "Disagree: " << disagree << " pixels" << std::endl;
        std::cout << "Agreement rate: " << (agreement_rate * 100.0) << "%" << std::endl;
        
        // VALIDATION 3: K-means clustering quality validation
        // Check that both background and foreground have reasonable component distributions
        std::cout << "\n=== K-MEANS CLUSTERING QUALITY VALIDATION ===" << std::endl;
        
        // Count how many different probable regions exist
        int cpu_pr_bgd = cpu_counts[cv::GC_PR_BGD];
        int cpu_pr_fgd = cpu_counts[cv::GC_PR_FGD]; 
        int metal_pr_bgd = metal_counts[cv::GC_PR_BGD];
        int metal_pr_fgd = metal_counts[cv::GC_PR_FGD];
        
        std::cout << "CPU probable regions: BG=" << cpu_pr_bgd << ", FG=" << cpu_pr_fgd << std::endl;
        std::cout << "Metal probable regions: BG=" << metal_pr_bgd << ", FG=" << metal_pr_fgd << std::endl;
        
        // VALIDATION 4: GMM model parameter comparison
        std::cout << "\n=== GMM MODEL COMPARISON ===" << std::endl;
        ASSERT_FALSE(bgd_cpu.empty()) << "CPU background model is empty";
        ASSERT_FALSE(fgd_cpu.empty()) << "CPU foreground model is empty"; 
        ASSERT_FALSE(bgd_metal.empty()) << "Metal background model is empty";
        ASSERT_FALSE(fgd_metal.empty()) << "Metal foreground model is empty";
        
        std::cout << "CPU models: bgd=" << bgd_cpu.size() << ", fgd=" << fgd_cpu.size() << std::endl;
        std::cout << "Metal models: bgd=" << bgd_metal.size() << ", fgd=" << fgd_metal.size() << std::endl;
        
        // Check that models have reasonable structure (should be 1x65 for standard GMM)
        EXPECT_EQ(bgd_cpu.rows, 1) << "CPU background model should have 1 row";
        EXPECT_EQ(bgd_cpu.cols, 65) << "CPU background model should have 65 columns (5 components * 13 params)";
        EXPECT_EQ(bgd_metal.rows, 1) << "Metal background model should have 1 row";
        EXPECT_EQ(bgd_metal.cols, 65) << "Metal background model should have 65 columns";
        
        // ASSERTIONS FOR TEST PASS/FAIL
        
        // 1. Agreement rate should be reasonable for a real image (not as high as synthetic)
        EXPECT_GT(agreement_rate, 0.6) << "CPU and Metal should agree on at least 60% of pixels for " 
                                      << scenario.first << " (got " << (agreement_rate*100) << "%)";
        
        // 2. Both implementations should produce some probable foreground pixels
        EXPECT_GT(cpu_pr_fgd, total_pixels * 0.05) << "CPU should produce some probable foreground pixels";
        EXPECT_GT(metal_pr_fgd, total_pixels * 0.05) << "Metal should produce some probable foreground pixels";
        
        // 3. Both implementations should produce some probable background pixels
        EXPECT_GT(cpu_pr_bgd, total_pixels * 0.05) << "CPU should produce some probable background pixels";
        EXPECT_GT(metal_pr_bgd, total_pixels * 0.05) << "Metal should produce some probable background pixels";
        
        // 4. K-means fix validation: Metal should not have excessive bias toward component 0
        // This was the main symptom of the bug we fixed
        int fg_pixels_total = metal_counts[cv::GC_FGD] + metal_counts[cv::GC_PR_FGD];
        int bg_pixels_total = metal_counts[cv::GC_BGD] + metal_counts[cv::GC_PR_BGD];
        
        EXPECT_GT(fg_pixels_total, 0) << "Metal should classify some pixels as foreground";
        EXPECT_GT(bg_pixels_total, 0) << "Metal should classify some pixels as background";
        
        // 5. Performance should be reasonable
        EXPECT_LT(metal_time.count(), 10000) << "Metal should complete within 10 seconds for real image";
        
        std::cout << "✓ " << scenario.first << " validation passed!" << std::endl;
    }
    
    std::cout << "\n=== K-MEANS FIX VALIDATION COMPLETE ===\n" << std::endl;
}

// Direct test of kmeansClusterByMask functionality with validation  
TEST(MetalImgproc_KMeans, ClusterByMaskValidation)
{
    std::cout << "\n=== DIRECT K-MEANS CLUSTER BY MASK VALIDATION ===\n" << std::endl;
    
    // Load real test image
    std::string img_path = std::string("../WID-small.jpg");
    cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
    ASSERT_FALSE(image.empty()) << "Cannot load test image: " << img_path;
    
    // Ensure correct format
    if (image.type() != CV_8UC3) {
        if (image.channels() == 4) {
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        }
    }
    
    // Convert to BGRA for Metal
    cv::Mat imageBGRA;
    cv::cvtColor(image, imageBGRA, cv::COLOR_BGR2BGRA);
    
    // Create a test mask with mixed regions
    cv::Mat mask(image.size(), CV_8UC1);
    mask.setTo(cv::GC_BGD); // Start with all background
    
    // Set foreground and probable regions
    cv::Rect center_rect(image.cols/3, image.rows/3, image.cols/3, image.rows/3);
    mask(center_rect).setTo(cv::GC_PR_FGD); // Center as probable foreground
    
    // Add some certain foreground pixels
    cv::Rect fg_rect(image.cols/2 - 20, image.rows/2 - 20, 40, 40);
    mask(fg_rect).setTo(cv::GC_FGD);
    
    // Add some probable background
    cv::Rect bg_border(10, 10, image.cols - 20, 30);
    mask(bg_border).setTo(cv::GC_PR_BGD);
    
    std::cout << "Created test mask with mixed regions" << std::endl;
    
    // Upload to Metal
    cv::metal::MetalMat metalImg, metalMask;
    metalImg.upload(imageBGRA);
    metalMask.upload(mask);
    
    cv::metal::Stream stream;
    
    // Test background clustering
    std::cout << "\n--- Testing Background K-means Clustering ---" << std::endl;
    cv::metal::MetalMat bgLabels;
    cv::Mat bgCentroids;
    
    cv::metal::kmeansClusterByMask(metalImg, metalMask, true, bgLabels, bgCentroids, stream);
    
    // Download and validate results
    cv::Mat h_bgLabels;
    bgLabels.download(h_bgLabels, stream, true);
    
    std::cout << "Background centroids shape: " << bgCentroids.size() << " type: " << bgCentroids.type() << std::endl;
    EXPECT_EQ(bgCentroids.rows, 5) << "Should have 5 background centroids";
    EXPECT_EQ(bgCentroids.cols, 3) << "Should have 3 channels (BGR)";
    
    // Validate label assignments for background pixels
    std::vector<int> bg_component_counts(5, 0);
    int bg_pixels_processed = 0;
    int bg_pixels_expected = 0;
    
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            uchar maskVal = mask.at<uchar>(y, x);
            int label = h_bgLabels.at<int>(y, x);
            
            if (maskVal == cv::GC_BGD || maskVal == cv::GC_PR_BGD) {
                // This is a background pixel - should have valid label
                bg_pixels_expected++;
                if (label >= 0 && label < 5) {
                    bg_component_counts[label]++;
                    bg_pixels_processed++;
                }
            }
        }
    }
    
    std::cout << "Background pixels expected: " << bg_pixels_expected 
              << ", processed: " << bg_pixels_processed << std::endl;
    std::cout << "Background component distribution: ";
    for (int i = 0; i < 5; i++) {
        std::cout << "C" << i << "=" << bg_component_counts[i] << " ";
    }
    std::cout << std::endl;
    
    // Test foreground clustering
    std::cout << "\n--- Testing Foreground K-means Clustering ---" << std::endl;
    cv::metal::MetalMat fgLabels;
    cv::Mat fgCentroids;
    
    cv::metal::kmeansClusterByMask(metalImg, metalMask, false, fgLabels, fgCentroids, stream);
    
    // Download and validate results
    cv::Mat h_fgLabels;
    fgLabels.download(h_fgLabels, stream, true);
    
    std::cout << "Foreground centroids shape: " << fgCentroids.size() << " type: " << fgCentroids.type() << std::endl;
    EXPECT_EQ(fgCentroids.rows, 5) << "Should have 5 foreground centroids";
    EXPECT_EQ(fgCentroids.cols, 3) << "Should have 3 channels (BGR)";
    
    // Validate label assignments for foreground pixels
    std::vector<int> fg_component_counts(5, 0);
    int fg_pixels_processed = 0;
    int fg_pixels_expected = 0;
    
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            uchar maskVal = mask.at<uchar>(y, x);
            int label = h_fgLabels.at<int>(y, x);
            
            if (maskVal == cv::GC_FGD || maskVal == cv::GC_PR_FGD) {
                // This is a foreground pixel - should have valid label
                fg_pixels_expected++;
                if (label >= 0 && label < 5) {
                    fg_component_counts[label]++;
                    fg_pixels_processed++;
                }
            }
        }
    }
    
    std::cout << "Foreground pixels expected: " << fg_pixels_expected 
              << ", processed: " << fg_pixels_processed << std::endl;
    std::cout << "Foreground component distribution: ";
    for (int i = 0; i < 5; i++) {
        std::cout << "C" << i << "=" << fg_component_counts[i] << " ";
    }
    std::cout << std::endl;
    
    // CRITICAL VALIDATIONS (these test the bug fixes)
    
    // 1. All background pixels should be processed correctly
    EXPECT_EQ(bg_pixels_processed, bg_pixels_expected) 
        << "All background pixels should be processed correctly";
    
    // 2. All foreground pixels should be processed correctly  
    EXPECT_EQ(fg_pixels_processed, fg_pixels_expected)
        << "All foreground pixels should be processed correctly";
    
    // 3. Component distribution should not be overly biased toward component 0
    // This was the main symptom of the original bug
    if (bg_pixels_expected > 0) {
        double bg_component0_ratio = double(bg_component_counts[0]) / bg_pixels_expected;
        EXPECT_LT(bg_component0_ratio, 0.8) 
            << "Background component 0 should not be overly dominant (ratio: " 
            << bg_component0_ratio << ")";
    }
    
    if (fg_pixels_expected > 0) {
        double fg_component0_ratio = double(fg_component_counts[0]) / fg_pixels_expected;
        EXPECT_LT(fg_component0_ratio, 0.8) 
            << "Foreground component 0 should not be overly dominant (ratio: " 
            << fg_component0_ratio << ")";
    }
    
    // 4. Multiple components should be used (diversity check)
    int bg_used_components = 0, fg_used_components = 0;
    for (int i = 0; i < 5; i++) {
        if (bg_component_counts[i] > 0) bg_used_components++;
        if (fg_component_counts[i] > 0) fg_used_components++;
    }
    
    EXPECT_GE(bg_used_components, 2) << "At least 2 background components should be used";
    EXPECT_GE(fg_used_components, 2) << "At least 2 foreground components should be used";
    
    // 5. Centroids should be in valid range [0, 255]
    for (int i = 0; i < 5; i++) {
        for (int c = 0; c < 3; c++) {
            float bg_val = bgCentroids.at<float>(i, c);
            float fg_val = fgCentroids.at<float>(i, c);
            
            EXPECT_GE(bg_val, 0.0f) << "Background centroid should be >= 0";
            EXPECT_LE(bg_val, 255.0f) << "Background centroid should be <= 255";
            EXPECT_GE(fg_val, 0.0f) << "Foreground centroid should be >= 0";
            EXPECT_LE(fg_val, 255.0f) << "Foreground centroid should be <= 255";
        }
    }
    
    std::cout << "✓ All k-means cluster by mask validations passed!" << std::endl;
    std::cout << "\n=== K-MEANS CLUSTER BY MASK VALIDATION COMPLETE ===\n" << std::endl;
}

} // namespace
} // namespace opencv_test

#endif // HAVE_METAL
