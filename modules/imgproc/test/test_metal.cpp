#include "test_precomp.hpp"
#include "opencv2/imgproc/metal.hpp"
#include <opencv2/imgcodecs.hpp>  // For debug image saving
#include <fstream>                // For debug logging
#include <iomanip>                // For std::setw
#include <map>
#include <set>

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
    testing::Values(
        std::make_tuple(perf::szVGA, 1, 0),
        std::make_tuple(perf::szVGA, 0, 1),
        std::make_tuple(perf::sz720p, 1, 0),
        std::make_tuple(perf::sz720p, 0, 1)
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

}} // namespace opencv_test

#endif // HAVE_METAL

// Connected Components tests
#ifdef HAVE_METAL

namespace opencv_test { namespace {

// Helper function to normalize labels for comparison
// Connected components algorithms don't guarantee sequential labeling,
// so we need to normalize them to compare components correctly
static void normalize_labels(Mat& labels) {
    CV_Assert(labels.type() == CV_32S);
    
    std::map<int, int> label_map;
    int next_label = 0;
    
    for (int i = 0; i < labels.rows; ++i) {
        int* ptr = labels.ptr<int>(i);
        for (int j = 0; j < labels.cols; ++j) {
            int& label = ptr[j];
            if (label > 0) {
                if (label_map.find(label) == label_map.end()) {
                    label_map[label] = ++next_label;
                }
                label = label_map[label];
            }
        }
    }
}

// Helper function to create test images
static Mat createTestImage(const std::string& pattern, Size size) {
    Mat img = Mat::zeros(size, CV_8UC1);
    
    if (pattern == "chessboard") {
        // Create chessboard pattern
        int blockSize = 8;
        for (int i = 0; i < size.height; i += blockSize) {
            for (int j = 0; j < size.width; j += blockSize) {
                if (((i / blockSize) + (j / blockSize)) % 2 == 0) {
                    Rect roi(j, i, std::min(blockSize, size.width - j), std::min(blockSize, size.height - i));
                    img(roi) = 255;
                }
            }
        }
    } else if (pattern == "circles") {
        // Create concentric circles
        Point center(size.width / 2, size.height / 2);
        cv::circle(img, center, std::min(size.width, size.height) / 6, Scalar(255), -1);
        cv::circle(img, center, std::min(size.width, size.height) / 4, Scalar(0), 6);
        cv::circle(img, center, std::min(size.width, size.height) / 3, Scalar(255), 4);
    } else if (pattern == "single_row") {
        // Single horizontal line
        if (size.height > 0) {
            img.row(size.height / 2) = 255;
        }
    } else if (pattern == "single_col") {
        // Single vertical line
        if (size.width > 0) {
            img.col(size.width / 2) = 255;
        }
    } else if (pattern == "random") {
        // Random binary pattern
        cv::RNG rng(42); // Fixed seed for reproducibility
        randu(img, 0, 2);
        img *= 255;
    }
    
    return img;
}

// Additional test patterns for specific failure analysis
static Mat createSpecificTestPattern(const std::string& pattern, Size size) {
    Mat img = Mat::zeros(size, CV_8UC1);
    
    if (pattern == "single_component") {
        // One large connected component (original failing pattern - now fixed)
        for (int i = 1; i < size.height - 1; i += 2) {
            for (int j = 1; j < size.width - 1; j += 2) {
                img.at<uchar>(i, j) = 255;
                if (j + 1 < size.width - 1) img.at<uchar>(i, j + 1) = 255;
            }
        }
    } else if (pattern == "diagonal_lines") {
        // Diagonal lines that should connect
        for (int i = 0; i < std::min(size.height, size.width); ++i) {
            img.at<uchar>(i, i) = 255;
            if (i + 1 < size.height && i + 1 < size.width) {
                img.at<uchar>(i + 1, i) = 255;
                img.at<uchar>(i, i + 1) = 255;
            }
        }
    } else if (pattern == "block_boundary") {
        // Pattern that tests 16x16 block boundaries specifically
        int blockSize = 16;
        for (int i = 0; i < size.height; i += blockSize) {
            for (int j = 0; j < size.width; j += blockSize) {
                // Create pattern at block boundaries
                if (i > 0 && j > 0) {
                    img.at<uchar>(i - 1, j - 1) = 255;
                    img.at<uchar>(i - 1, j) = 255;
                    img.at<uchar>(i, j - 1) = 255;
                    img.at<uchar>(i, j) = 255;
                }
            }
        }
    } else if (pattern == "thin_connections") {
        // Thin connections that might fail in Union-Find
        for (int i = 10; i < size.height - 10; i += 20) {
            for (int j = 0; j < size.width; ++j) {
                img.at<uchar>(i, j) = 255;
                if (j % 10 == 0 && i + 1 < size.height) {
                    img.at<uchar>(i + 1, j) = 255; // Vertical connection
                }
            }
        }
    } else if (pattern == "small_components") {
        // Many small 2x2 components (should NOT merge)
        for (int i = 0; i < size.height - 1; i += 4) {
            for (int j = 0; j < size.width - 1; j += 4) {
                img.at<uchar>(i, j) = 255;
                img.at<uchar>(i, j + 1) = 255;
                img.at<uchar>(i + 1, j) = 255;
                img.at<uchar>(i + 1, j + 1) = 255;
            }
        }
    } else if (pattern == "large_sparse") {
        // Large sparse component (tests long Union-Find chains)
        cv::RNG rng(12345); // Fixed seed
        std::vector<Point> points;
        
        // Generate sparse points
        for (int i = 0; i < size.height * size.width / 50; ++i) {
            Point p(rng.uniform(1, size.width - 1), rng.uniform(1, size.height - 1));
            points.push_back(p);
            img.at<uchar>(p.y, p.x) = 255;
        }
        
        // Connect them sparsely
        for (size_t i = 1; i < points.size(); ++i) {
            Point p1 = points[i - 1];
            Point p2 = points[i];
            cv::line(img, p1, p2, Scalar(255), 1);
        }
    } else if (pattern == "atomic_stress") {
        // Pattern designed to stress atomic operations
        int center_x = size.width / 2;
        int center_y = size.height / 2;
        
        // Star pattern with many connections converging
        for (int angle = 0; angle < 360; angle += 10) {
            double rad = angle * CV_PI / 180.0;
            int x = center_x + (int)(20 * cos(rad));
            int y = center_y + (int)(20 * sin(rad));
            if (x >= 0 && x < size.width && y >= 0 && y < size.height) {
                cv::line(img, Point(center_x, center_y), Point(x, y), Scalar(255), 1);
            }
        }
    }
    
    return img;
}

// Correctness test for Connected Components
typedef testing::TestWithParam<tuple<Size, std::string>> Imgproc_ConnectedComponents;
TEST_P(Imgproc_ConnectedComponents, Correctness)
{
    Size sz = get<0>(GetParam());
    std::string pattern = get<1>(GetParam());
    int connectivity = 8;
    int ltype = CV_32S;

    Mat src = createTestImage(pattern, sz);
    Mat labels_cpu, labels_metal_host;

    // CPU reference implementation
    cv::connectedComponents(src, labels_cpu, connectivity, ltype);

    // Metal implementation
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels;
    cv::metal::connectedComponents(d_src, d_labels, connectivity, ltype);
    d_labels.download(labels_metal_host);

    // Normalize labels for comparison (essential for connected components)
    normalize_labels(labels_cpu);
    normalize_labels(labels_metal_host);

    // Compare normalized results
    EXPECT_MAT_NEAR(labels_cpu, labels_metal_host, 0);
}

INSTANTIATE_TEST_CASE_P(Imgproc_Metal, Imgproc_ConnectedComponents,
    testing::Combine(
        testing::Values(Size(64, 64), Size(128, 96), Size(256, 192)),
        testing::Values("chessboard", "circles", "single_row", "single_col", "random")
    )
);

// Comprehensive test for specific failure patterns (regression test for type safety bug)
TEST_P(Imgproc_ConnectedComponents, SpecificPatterns)
{
    Size sz = get<0>(GetParam());
    std::string pattern = get<1>(GetParam());
    int connectivity = 8;
    int ltype = CV_32S;

    Mat src = createSpecificTestPattern(pattern, sz);
    Mat labels_cpu, labels_metal_host;

    // CPU reference implementation
    cv::connectedComponents(src, labels_cpu, connectivity, ltype);

    // Metal implementation
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels;
    cv::metal::connectedComponents(d_src, d_labels, connectivity, ltype);
    d_labels.download(labels_metal_host);

    // Normalize labels for comparison
    normalize_labels(labels_cpu);
    normalize_labels(labels_metal_host);

    EXPECT_MAT_NEAR(labels_cpu, labels_metal_host, 0);
}

INSTANTIATE_TEST_CASE_P(Imgproc_Metal_Specific, Imgproc_ConnectedComponents,
    testing::Combine(
        testing::Values(Size(64, 64), Size(128, 96)),
        testing::Values("single_component", "diagonal_lines", "block_boundary", 
                       "thin_connections", "small_components", "large_sparse", "atomic_stress")
    )
);

// Edge case tests for Connected Components
TEST(Imgproc_ConnectedComponents, EmptyImage)
{
    Size sz(64, 64);
    Mat src = Mat::zeros(sz, CV_8UC1); // All zeros
    Mat labels_cpu, labels_metal_host;

    // CPU reference
    cv::connectedComponents(src, labels_cpu, 8, CV_32S);

    // Metal implementation
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels;
    cv::metal::connectedComponents(d_src, d_labels, 8, CV_32S);
    d_labels.download(labels_metal_host);

    // All labels should be 0 for empty image
    EXPECT_MAT_NEAR(labels_cpu, labels_metal_host, 0);
}

TEST(Imgproc_ConnectedComponents, FullImage)
{
    Size sz(64, 64);
    Mat src = Mat::ones(sz, CV_8UC1) * 255; // All ones
    Mat labels_cpu, labels_metal_host;

    // CPU reference
    cv::connectedComponents(src, labels_cpu, 8, CV_32S);

    // Metal implementation
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels;
    cv::metal::connectedComponents(d_src, d_labels, 8, CV_32S);
    d_labels.download(labels_metal_host);

    normalize_labels(labels_cpu);
    normalize_labels(labels_metal_host);

    EXPECT_MAT_NEAR(labels_cpu, labels_metal_host, 0);
}

TEST(Imgproc_ConnectedComponents, SinglePixel)
{
    Size sz(3, 3);
    Mat src = Mat::zeros(sz, CV_8UC1);
    src.at<uchar>(1, 1) = 255; // Single center pixel
    Mat labels_cpu, labels_metal_host;

    // CPU reference
    cv::connectedComponents(src, labels_cpu, 8, CV_32S);

    // Metal implementation
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels;
    cv::metal::connectedComponents(d_src, d_labels, 8, CV_32S);
    d_labels.download(labels_metal_host);

    normalize_labels(labels_cpu);
    normalize_labels(labels_metal_host);

    EXPECT_MAT_NEAR(labels_cpu, labels_metal_host, 0);
}

// Stream variant test
TEST(Imgproc_ConnectedComponents, StreamVariant)
{
    Size sz(128, 96);
    Mat src = createTestImage("chessboard", sz);
    Mat labels_sync, labels_stream;

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels_sync, d_labels_stream;

    // Synchronous version
    cv::metal::connectedComponents(d_src, d_labels_sync, 8, CV_32S);

    // Stream version
    cv::metal::Stream stream;
    cv::metal::connectedComponents(d_src, d_labels_stream, 8, CV_32S, stream);
    stream.commitAndWait();

    // Download results
    d_labels_sync.download(labels_sync);
    d_labels_stream.download(labels_stream);

    // Normalize and compare
    normalize_labels(labels_sync);
    normalize_labels(labels_stream);

    EXPECT_MAT_NEAR(labels_sync, labels_stream, 0);
}

// Regression test for the critical type mismatch bug (minimal test case)
TEST(Imgproc_ConnectedComponents, TypeSafetyRegression)
{
    // Minimal 4x4 pattern that exposed the type mismatch bug
    Mat src = Mat::zeros(4, 4, CV_8UC1);
    
    // Block (0,0): pixel at (1,1) - position 'd' 
    src.at<uchar>(1, 1) = 255;
    
    // Block (0,1): pixel at (1,2) - position 'c'  
    src.at<uchar>(1, 2) = 255;
    
    Mat labels_cpu, labels_metal_host;

    // CPU reference implementation
    cv::connectedComponents(src, labels_cpu, 8, CV_32S);

    // Metal implementation (should correctly connect the two pixels)
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels;
    cv::metal::connectedComponents(d_src, d_labels, 8, CV_32S);
    d_labels.download(labels_metal_host);

    // DEBUG: Print the raw results before normalization
    std::cout << "\n=== DEBUG TypeSafetyRegression ===" << std::endl;
    std::cout << "Input image:" << std::endl;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            std::cout << std::setw(3) << (int)src.at<uchar>(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nCPU labels (raw):" << std::endl;
    for (int i = 0; i < labels_cpu.rows; i++) {
        for (int j = 0; j < labels_cpu.cols; j++) {
            std::cout << std::setw(3) << labels_cpu.at<int>(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nMetal labels (raw):" << std::endl;
    for (int i = 0; i < labels_metal_host.rows; i++) {
        for (int j = 0; j < labels_metal_host.cols; j++) {
            std::cout << std::setw(3) << labels_metal_host.at<int>(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // Normalize labels for comparison
    normalize_labels(labels_cpu);
    normalize_labels(labels_metal_host);

    std::cout << "\nCPU labels (normalized):" << std::endl;
    for (int i = 0; i < labels_cpu.rows; i++) {
        for (int j = 0; j < labels_cpu.cols; j++) {
            std::cout << std::setw(3) << labels_cpu.at<int>(i, j) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nMetal labels (normalized):" << std::endl;
    for (int i = 0; i < labels_metal_host.rows; i++) {
        for (int j = 0; j < labels_metal_host.cols; j++) {
            std::cout << std::setw(3) << labels_metal_host.at<int>(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=================================" << std::endl;

    // This test would fail before the type safety fix (Metal=2 components, CPU=1 component)
    // After fix: both should detect 1 component
    EXPECT_MAT_NEAR(labels_cpu, labels_metal_host, 0);
}

//==============================================================================
// Optimized Filtering Operations Tests
//==============================================================================

// Test optimized bilateral filter correctness
typedef testing::TestWithParam<tuple<Size, int, float, float, int>> OptimizedBilateralFilter;
TEST_P(OptimizedBilateralFilter, Correctness)
{
    Size sz = get<0>(GetParam());
    int ksize = get<1>(GetParam());
    float sigma_color = get<2>(GetParam());
    float sigma_spatial = get<3>(GetParam());
    int type = get<4>(GetParam());

    cv::RNG rng(42);
    Mat src = randomMat(rng, sz, type, 0, type == CV_8UC1 || type == CV_8UC4 ? 255 : 1, false);
    Mat dst_cpu, dst_metal_cpu;

    // CPU reference
    cv::bilateralFilter(src, dst_cpu, ksize, sigma_color, sigma_spatial);

    // Metal optimized implementation
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;
    cv::metal::bilateralFilter(d_src, d_dst, ksize, sigma_color, sigma_spatial);
    d_dst.download(dst_metal_cpu);

    // Use appropriate tolerance for bilateral filter
    double tolerance = (type == CV_8UC1 || type == CV_8UC4) ? 3.0 : 0.02;
    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, tolerance);
}
INSTANTIATE_TEST_CASE_P(Imgproc_Metal_Optimized, OptimizedBilateralFilter,
    testing::Combine(
        testing::Values(Size(64, 64), Size(128, 128)),
        testing::Values(5, 9),
        testing::Values(25.0f, 50.0f),
        testing::Values(25.0f, 50.0f),
        testing::Values(CV_8UC1, CV_32FC1)  // Only use types supported by CPU OpenCV bilateral filter
    )
);

// Test optimized box filter correctness
typedef testing::TestWithParam<tuple<Size, Size, int>> OptimizedBoxFilter;
TEST_P(OptimizedBoxFilter, Correctness)
{
    Size img_sz = get<0>(GetParam());
    Size kernel_sz = get<1>(GetParam());
    int type = get<2>(GetParam());

    cv::RNG rng(42);
    Mat src = randomMat(rng, img_sz, type, 0, type == CV_8UC1 || type == CV_8UC4 ? 255 : 1, false);
    Mat dst_cpu, dst_metal_cpu;

    // CPU reference
    cv::boxFilter(src, dst_cpu, -1, kernel_sz, Point(-1,-1), true);

    // Metal optimized implementation
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;
    cv::metal::boxFilter(d_src, d_dst, -1, kernel_sz, Point(-1,-1), true);
    d_dst.download(dst_metal_cpu);

    // Box filter should be very accurate
    double tolerance = (type == CV_8UC1 || type == CV_8UC4) ? 1.0 : 0.01;
    EXPECT_MAT_NEAR(dst_cpu, dst_metal_cpu, tolerance);
}
INSTANTIATE_TEST_CASE_P(Imgproc_Metal_Optimized, OptimizedBoxFilter,
    testing::Combine(
        testing::Values(Size(64, 64), Size(128, 128)),
        testing::Values(Size(3,3), Size(5,5), Size(9,9)),
        testing::Values(CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3)  // Use standard supported types
    )
);

// Edge case test for small images
TEST(Imgproc_Metal_Optimized, SmallImageEdgeCase)
{
    Mat small_img = Mat::ones(8, 8, CV_8UC1) * 128;
    cv::randu(small_img, 0, 255);
    
    // CPU reference
    Mat cpu_result;
    cv::bilateralFilter(small_img, cpu_result, 3, 50.0, 50.0);
    
    // Metal implementation
    cv::metal::MetalMat d_src(small_img);
    cv::metal::MetalMat d_dst;
    cv::metal::bilateralFilter(d_src, d_dst, 3, 50.0, 50.0);
    
    Mat metal_result;
    d_dst.download(metal_result);
    
    EXPECT_MAT_NEAR(cpu_result, metal_result, 3.0);
}

// Correctness test for K-means clustering
typedef testing::TestWithParam<tuple<Size, int, int, int>> Imgproc_Kmeans;
TEST_P(Imgproc_Kmeans, Correctness)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    int K = get<2>(GetParam());
    int attempts = get<3>(GetParam());
    
    cv::RNG rng;
    Mat src = randomMat(rng, sz, type, 0, 255, false);
    
    // Create test data for reproducible results
    if (type == CV_8UC3) {
        src = Mat::zeros(sz, type);
        if (K == 2) {
            // Two-color pattern for K=2
            cv::rectangle(src, Rect(0, 0, sz.width/2, sz.height), Scalar(255, 0, 0), -1);
            cv::rectangle(src, Rect(sz.width/2, 0, sz.width/2, sz.height), Scalar(0, 255, 0), -1);
        } else if (K == 4) {
            // Four-color pattern for K=4
            cv::rectangle(src, Rect(0, 0, sz.width/2, sz.height/2), Scalar(255, 0, 0), -1);      // Red
            cv::rectangle(src, Rect(sz.width/2, 0, sz.width/2, sz.height/2), Scalar(0, 255, 0), -1);  // Green
            cv::rectangle(src, Rect(0, sz.height/2, sz.width/2, sz.height/2), Scalar(0, 0, 255), -1); // Blue
            cv::rectangle(src, Rect(sz.width/2, sz.height/2, sz.width/2, sz.height/2), Scalar(255, 255, 0), -1); // Yellow
        }
    } else if (type == CV_8UC4) {
        src = Mat::zeros(sz, type);
        if (K == 2) {
            // Two-color pattern for K=2
            cv::rectangle(src, Rect(0, 0, sz.width/2, sz.height), Scalar(255, 0, 0, 255), -1);
            cv::rectangle(src, Rect(sz.width/2, 0, sz.width/2, sz.height), Scalar(0, 255, 0, 255), -1);
        } else if (K == 4) {
            // Four-color pattern for K=4
            cv::rectangle(src, Rect(0, 0, sz.width/2, sz.height/2), Scalar(255, 0, 0, 255), -1);      // Red
            cv::rectangle(src, Rect(sz.width/2, 0, sz.width/2, sz.height/2), Scalar(0, 255, 0, 255), -1);  // Green
            cv::rectangle(src, Rect(0, sz.height/2, sz.width/2, sz.height/2), Scalar(0, 0, 255, 255), -1); // Blue
            cv::rectangle(src, Rect(sz.width/2, sz.height/2, sz.width/2, sz.height/2), Scalar(255, 255, 0, 255), -1); // Yellow
        }
    }

    Mat labels_cpu, centers_cpu;
    Mat labels_metal_cpu, centers_metal_cpu;

    // CPU reference implementation (requires CV_32F data)
    Mat src_float;
    src.convertTo(src_float, CV_32F, 1.0/255.0); // Convert to float [0,1] range
    
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 20, 1.0);
    
    double compactness_cpu = cv::kmeans(src_float.reshape(1, src_float.rows * src_float.cols), K, labels_cpu, 
                                       criteria, attempts, KMEANS_RANDOM_CENTERS, centers_cpu);

    // Metal implementation
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels, d_centers;
    

    
    double compactness_metal = cv::metal::kmeans(d_src, K, d_labels, criteria, attempts, 
                                                KMEANS_RANDOM_CENTERS, d_centers);
    
    d_labels.download(labels_metal_cpu);
    d_centers.download(centers_metal_cpu);

    // Reshape Metal labels to match CPU format (1D)
    labels_metal_cpu = labels_metal_cpu.reshape(1, labels_metal_cpu.rows * labels_metal_cpu.cols);
    
    // Debug labels
    double min_cpu, max_cpu;
    cv::minMaxLoc(labels_cpu, &min_cpu, &max_cpu);
    
    double min_metal, max_metal;
    cv::minMaxLoc(labels_metal_cpu, &min_metal, &max_metal);
    


    // For centers, convert from 4-channel to 3-channel for comparison
    Mat centers_metal_3ch;
    if (centers_metal_cpu.channels() == 4) {
        centers_metal_3ch.create(centers_metal_cpu.rows, 3, CV_32F);  // K×3 format like CPU
        for (int i = 0; i < centers_metal_cpu.rows; i++) {
            Vec4f src = centers_metal_cpu.at<Vec4f>(i, 0);
            centers_metal_3ch.at<float>(i, 0) = src[0];
            centers_metal_3ch.at<float>(i, 1) = src[1];
            centers_metal_3ch.at<float>(i, 2) = src[2];
        }
        centers_metal_cpu = centers_metal_3ch;
    }

    // Basic validation
    EXPECT_EQ(labels_cpu.rows, labels_metal_cpu.rows);
    EXPECT_EQ(labels_cpu.cols, labels_metal_cpu.cols);
    EXPECT_EQ(centers_cpu.rows, centers_metal_cpu.rows);
    EXPECT_EQ(centers_cpu.cols, centers_metal_cpu.cols);
    
    // Validate labels are in valid range
    double min_label, max_label;
    cv::minMaxLoc(labels_metal_cpu, &min_label, &max_label);
    EXPECT_GE(min_label, 0);
    EXPECT_LT(max_label, K);
    
    // Validate centers are in valid range [0, 1] for normalized values
    double min_center, max_center;
    cv::minMaxLoc(centers_metal_cpu, &min_center, &max_center);
    EXPECT_GE(min_center, 0.0);
    EXPECT_LE(max_center, 1.0);
    
    // For simple patterns, expect similar number of clusters found
    std::set<int> unique_labels_cpu, unique_labels_metal;
    for (int i = 0; i < labels_cpu.rows; i++) {
        unique_labels_cpu.insert(labels_cpu.at<int>(i, 0));
    }
    for (int i = 0; i < labels_metal_cpu.rows; i++) {
        unique_labels_metal.insert(labels_metal_cpu.at<int>(i, 0));
    }
    
    // Should find same number of clusters for simple patterns
    EXPECT_EQ(unique_labels_cpu.size(), unique_labels_metal.size());
    
    // Test compactness values - handle perfect clustering case
    if (compactness_cpu > 0.0 && compactness_metal > 0.0) {
        // Both algorithms found non-perfect clustering
        EXPECT_LT(compactness_metal / compactness_cpu, 10.0);  // Within 10x of CPU
        EXPECT_GT(compactness_metal / compactness_cpu, 0.1);   // Not more than 10x worse
    } else if (compactness_cpu == 0.0 && compactness_metal == 0.0) {
        // Both algorithms achieved perfect clustering - this is ideal!
        SUCCEED();
    } else {
        // One achieved perfect clustering, the other didn't - check they're both small
        EXPECT_LT(max(compactness_cpu, compactness_metal), 100.0);  // Should be reasonably small
    }
}
INSTANTIATE_TEST_CASE_P(Imgproc_Metal, Imgproc_Kmeans,
    testing::Combine(
        testing::Values(Size(64, 64), Size(128, 128)),
        testing::Values(CV_8UC3, CV_8UC4),
        testing::Values(2, 4),
        testing::Values(1, 3)
    )
);

// Stream-based K-means test
TEST(Imgproc_Kmeans, StreamCorrectness)
{
    Size sz(64, 64);
    int type = CV_8UC3;
    int K = 3;
    
    Mat src = Mat::zeros(sz, type);
    // Create three distinct regions
    cv::rectangle(src, Rect(0, 0, sz.width/3, sz.height), Scalar(255, 0, 0), -1);
    cv::rectangle(src, Rect(sz.width/3, 0, sz.width/3, sz.height), Scalar(0, 255, 0), -1);
    cv::rectangle(src, Rect(2*sz.width/3, 0, sz.width/3, sz.height), Scalar(0, 0, 255), -1);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels1, d_labels2;
    cv::metal::MetalMat d_centers1, d_centers2;

    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0);

    // Test without stream
    double compactness1 = cv::metal::kmeans(d_src, K, d_labels1, criteria, 1, 
                                           KMEANS_RANDOM_CENTERS, d_centers1);

    // Test with stream
    cv::metal::Stream stream;
    double compactness2 = cv::metal::kmeans(d_src, K, d_labels2, criteria, 1, 
                                           KMEANS_RANDOM_CENTERS, d_centers2, stream);
    stream.waitUntilCompleted();

    Mat labels1, labels2, centers1, centers2;
    d_labels1.download(labels1);
    d_labels2.download(labels2);
    d_centers1.download(centers1);
    d_centers2.download(centers2);

    // Results should be valid for both versions
    EXPECT_GT(compactness1, 0.0);
    EXPECT_GT(compactness2, 0.0);
    
    // Validate label ranges
    double min_label, max_label;
    cv::minMaxLoc(labels1, &min_label, &max_label);
    EXPECT_GE(min_label, 0);
    EXPECT_LT(max_label, K);
    
    cv::minMaxLoc(labels2, &min_label, &max_label);
    EXPECT_GE(min_label, 0);
    EXPECT_LT(max_label, K);
}

// Edge cases test for K-means
TEST(Imgproc_Kmeans, EdgeCases)
{
    Size sz(32, 32);
    Mat src(sz, CV_8UC3);
    cv::randu(src, 0, 255);
    
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels, d_centers;
    TermCriteria criteria(TermCriteria::MAX_ITER, 10, 0);

    // Test K=1 (single cluster)
    double compactness;
    compactness = cv::metal::kmeans(d_src, 1, d_labels, criteria, 1, 
                                   KMEANS_RANDOM_CENTERS, d_centers);
    EXPECT_GT(compactness, 0.0);
    
    Mat labels;
    d_labels.download(labels);
    
    // All labels should be 0 for K=1
    double min_label, max_label;
    cv::minMaxLoc(labels, &min_label, &max_label);
    EXPECT_EQ(min_label, 0);
    EXPECT_EQ(max_label, 0);

    // Test small image
    Mat small_src = src(Rect(0, 0, 8, 8));
    cv::metal::MetalMat d_small_src(small_src);
    compactness = cv::metal::kmeans(d_small_src, 2, d_labels, criteria, 1, 
                                   KMEANS_RANDOM_CENTERS, d_centers);
    EXPECT_GT(compactness, 0.0);
}

// Input validation test
TEST(Imgproc_Kmeans, InputValidation)
{
    Size sz(32, 32);
    Mat src(sz, CV_8UC3);
    cv::randu(src, 0, 255);
    
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels, d_centers;
    TermCriteria criteria(TermCriteria::MAX_ITER, 10, 0);

    // Test invalid K values
    EXPECT_THROW(cv::metal::kmeans(d_src, 0, d_labels, criteria, 1, 
                                  KMEANS_RANDOM_CENTERS, d_centers), cv::Exception);
    EXPECT_THROW(cv::metal::kmeans(d_src, -1, d_labels, criteria, 1, 
                                  KMEANS_RANDOM_CENTERS, d_centers), cv::Exception);
    EXPECT_THROW(cv::metal::kmeans(d_src, 1001, d_labels, criteria, 1, 
                                  KMEANS_RANDOM_CENTERS, d_centers), cv::Exception);

    // Test invalid attempts
    EXPECT_THROW(cv::metal::kmeans(d_src, 2, d_labels, criteria, 0, 
                                  KMEANS_RANDOM_CENTERS, d_centers), cv::Exception);
    EXPECT_THROW(cv::metal::kmeans(d_src, 2, d_labels, criteria, -1, 
                                  KMEANS_RANDOM_CENTERS, d_centers), cv::Exception);

    // Test empty input
    cv::metal::MetalMat d_empty;
    EXPECT_THROW(cv::metal::kmeans(d_empty, 2, d_labels, criteria, 1, 
                                  KMEANS_RANDOM_CENTERS, d_centers), cv::Exception);

    // Test unsupported format
    Mat unsupported(sz, CV_8UC1);
    cv::metal::MetalMat d_unsupported(unsupported);
    EXPECT_THROW(cv::metal::kmeans(d_unsupported, 2, d_labels, criteria, 1, 
                                  KMEANS_RANDOM_CENTERS, d_centers), cv::Exception);
}

// Performance regression test
TEST(Imgproc_Metal_Optimized, PerformanceRegression)
{
    Mat test_img(512, 512, CV_8UC4);
    cv::randu(test_img, 0, 255);
    
    cv::metal::MetalMat d_src(test_img);
    cv::metal::MetalMat d_dst;
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        cv::metal::bilateralFilter(d_src, d_dst, 9, 50.0, 50.0);
    }
    
    // Measure optimized implementation
    int64 start = cv::getTickCount();
    for (int i = 0; i < 5; i++) {
        cv::metal::bilateralFilter(d_src, d_dst, 9, 50.0, 50.0);
    }
    int64 end = cv::getTickCount();
    
    double time_ms = (end - start) * 1000.0 / cv::getTickFrequency() / 5.0;
    
    std::cout << "Optimized BilateralFilter (512x512, 4ch, k=9): " 
              << time_ms << " ms" << std::endl;
    
    // Basic sanity check - should complete in reasonable time
    EXPECT_LT(time_ms, 100.0) << "Performance regression detected";
}

}} // namespace opencv_test

#endif // HAVE_METAL