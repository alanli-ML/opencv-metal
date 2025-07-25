#include "perf_precomp.hpp"

#ifdef HAVE_METAL

namespace opencv_test {

// Performance test parameters: Size, iterations
typedef perf::TestBaseWithParam<tuple<Size, int>> MetalImgproc_GrabCut_Perf;

PERF_TEST_P(MetalImgproc_GrabCut_Perf, CPU_Baseline,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p),
        testing::Values(3, 5, 10)
    )
)
{
    Size sz = get<0>(GetParam());
    int iterations = get<1>(GetParam());
    
    // Create test image
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    // Add some structure to make segmentation meaningful
    circle(image, Point(sz.width/2, sz.height/2), min(sz.width, sz.height)/4, 
           Scalar(100, 200, 50), -1);
    
    Mat mask = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel, fgdModel;
    Rect rect(sz.width/4, sz.height/4, sz.width/2, sz.height/2);
    
    // Performance test for CPU baseline
    TEST_CYCLE()
    {
        Mat mask_copy = mask.clone();
        cv::grabCut(image, mask_copy, rect, bgdModel, fgdModel, iterations, GC_INIT_WITH_RECT);
    }
    
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(MetalImgproc_GrabCut_Perf, Metal_Individual,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p),
        testing::Values(3, 5, 10)
    )
)
{
    Size sz = get<0>(GetParam());
    int iterations = get<1>(GetParam());
    
    // Create test image
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    // Add some structure to make segmentation meaningful
    circle(image, Point(sz.width/2, sz.height/2), min(sz.width, sz.height)/4, 
           Scalar(100, 200, 50), -1);
    
    Mat mask = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel, fgdModel;
    Rect rect(sz.width/4, sz.height/4, sz.width/2, sz.height/2);
    
    // Performance test for Metal implementation (individual operations)
    TEST_CYCLE()
    {
        Mat mask_copy = mask.clone();
        cv::metal::grabCut(image, mask_copy, rect, bgdModel, fgdModel, iterations, GC_INIT_WITH_RECT);
    }
    
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(MetalImgproc_GrabCut_Perf, Metal_Stream,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p, perf::sz1080p),
        testing::Values(3, 5, 10)
    )
)
{
    Size sz = get<0>(GetParam());
    int iterations = get<1>(GetParam());
    
    // Create test image
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    // Add some structure to make segmentation meaningful
    circle(image, Point(sz.width/2, sz.height/2), min(sz.width, sz.height)/4, 
           Scalar(100, 200, 50), -1);
    
    Mat mask = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel, fgdModel;
    Rect rect(sz.width/4, sz.height/4, sz.width/2, sz.height/2);
    
    // Performance test for Metal implementation (with stream)
    TEST_CYCLE()
    {
        cv::metal::Stream stream;
        Mat mask_copy = mask.clone();
        cv::metal::grabCut(image, mask_copy, rect, bgdModel, fgdModel, iterations, GC_INIT_WITH_RECT, stream);
        stream.commitAndWait();
    }
    
    SANITY_CHECK_NOTHING();
}

// Test different GrabCut modes performance
typedef perf::TestBaseWithParam<tuple<Size, int, int>> MetalImgproc_GrabCut_Modes;

PERF_TEST_P(MetalImgproc_GrabCut_Modes, Metal_Modes,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p),
        testing::Values(GC_INIT_WITH_RECT, GC_INIT_WITH_MASK, GC_EVAL),
        testing::Values(3, 5)
    )
)
{
    Size sz = get<0>(GetParam());
    int mode = get<1>(GetParam());
    int iterations = get<2>(GetParam());
    
    // Create test image
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    // Add some structure
    circle(image, Point(sz.width/2, sz.height/2), min(sz.width, sz.height)/4, 
           Scalar(100, 200, 50), -1);
    
    Mat mask = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel, fgdModel;
    Rect rect(sz.width/4, sz.height/4, sz.width/2, sz.height/2);
    
    // Initialize mask for non-rect modes
    if (mode != GC_INIT_WITH_RECT) {
        // Pre-initialize with GC_INIT_WITH_RECT
        cv::metal::grabCut(image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
    }
    
    // Performance test for specific mode
    TEST_CYCLE()
    {
        Mat mask_copy = mask.clone();
        Mat bgdModel_copy = bgdModel.clone();
        Mat fgdModel_copy = fgdModel.clone();
        cv::metal::grabCut(image, mask_copy, rect, bgdModel_copy, fgdModel_copy, iterations, mode);
    }
    
    SANITY_CHECK_NOTHING();
}

// Chained operations performance test (demonstrates Metal backend advantage)
PERF_TEST(MetalImgproc_GrabCut_Perf, ChainedOperations_Individual)
{
    Size sz = perf::sz720p;
    
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    circle(image, Point(sz.width/2, sz.height/2), min(sz.width, sz.height)/4, 
           Scalar(100, 200, 50), -1);
    
    Rect rect(sz.width/4, sz.height/4, sz.width/2, sz.height/2);
    
    // Test multiple independent GrabCut operations (simulating batch processing)
    TEST_CYCLE()
    {
        for (int i = 0; i < 3; i++) {
            Mat mask = Mat::zeros(sz, CV_8UC1);
            Mat bgdModel, fgdModel;
            cv::metal::grabCut(image, mask, rect, bgdModel, fgdModel, 2, GC_INIT_WITH_RECT);
        }
    }
    
    SANITY_CHECK_NOTHING();
}

PERF_TEST(MetalImgproc_GrabCut_Perf, ChainedOperations_Stream)
{
    Size sz = perf::sz720p;
    
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    circle(image, Point(sz.width/2, sz.height/2), min(sz.width, sz.height)/4, 
           Scalar(100, 200, 50), -1);
    
    Rect rect(sz.width/4, sz.height/4, sz.width/2, sz.height/2);
    
    // Test multiple operations using the same stream (better GPU utilization)
    TEST_CYCLE()
    {
        cv::metal::Stream stream;
        for (int i = 0; i < 3; i++) {
            Mat mask = Mat::zeros(sz, CV_8UC1);
            Mat bgdModel, fgdModel;
            cv::metal::grabCut(image, mask, rect, bgdModel, fgdModel, 2, GC_INIT_WITH_RECT, stream);
        }
        stream.commitAndWait();
    }
    
    SANITY_CHECK_NOTHING();
}

// Component-level performance tests
PERF_TEST(MetalImgproc_GrabCut_Perf, Beta_Calculation)
{
    Size sz = perf::sz1080p;
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    cv::metal::MetalMat d_image(image);
    cv::metal::Stream stream;
    
    TEST_CYCLE()
    {
        double beta = cv::metal::calcBeta(d_image);
        (void)beta; // Suppress unused variable warning
    }
    
    SANITY_CHECK_NOTHING();
}

PERF_TEST(MetalImgproc_GrabCut_Perf, PairwiseWeights_Calculation)
{
    Size sz = perf::sz1080p;
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    cv::metal::MetalMat d_image(image);
    cv::metal::MetalMat leftW, topleftW, topW, toprightW;
    cv::metal::Stream stream;
    
    double beta = 0.1; // Typical value
    double gamma = 50.0;
    
    TEST_CYCLE()
    {
        cv::metal::calcNWeights(d_image, leftW, topleftW, topW, toprightW, beta, gamma, stream);
        stream.commitAndWait();
    }
    
    SANITY_CHECK_NOTHING();
}

// Memory transfer performance
PERF_TEST(MetalImgproc_GrabCut_Perf, Memory_Transfer)
{
    Size sz = perf::sz1080p;
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    TEST_CYCLE()
    {
        // Test upload/download performance
        cv::metal::MetalMat d_image(image);
        Mat downloaded_image;
        d_image.download(downloaded_image);
    }
    
    SANITY_CHECK_NOTHING();
}

// Convergence analysis performance test
PERF_TEST(MetalImgproc_GrabCut_Perf, Convergence_Analysis)
{
    Size sz = perf::szVGA;
    Mat image(sz, CV_8UC3);
    randu(image, 0, 255);
    
    // Create image with clear structure for faster convergence
    image.setTo(Scalar(50, 50, 50));
    circle(image, Point(sz.width/2, sz.height/2), sz.width/4, Scalar(200, 200, 200), -1);
    
    Mat mask = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel, fgdModel;
    Rect rect(sz.width/4, sz.height/4, sz.width/2, sz.height/2);
    
    // Test with many iterations to see convergence behavior
    TEST_CYCLE()
    {
        Mat mask_copy = mask.clone();
        cv::metal::grabCut(image, mask_copy, rect, bgdModel, fgdModel, 15, GC_INIT_WITH_RECT);
    }
    
    SANITY_CHECK_NOTHING();
}

// Real-world scenario test with different image characteristics
typedef perf::TestBaseWithParam<tuple<Size, int>> MetalImgproc_GrabCut_Scenarios;

PERF_TEST_P(MetalImgproc_GrabCut_Scenarios, Natural_Images,
    testing::Combine(
        testing::Values(perf::szVGA, perf::sz720p),
        testing::Values(1, 2, 3) // Different image types
    )
)
{
    Size sz = get<0>(GetParam());
    int image_type = get<1>(GetParam());
    
    Mat image(sz, CV_8UC3);
    
    // Create different types of test images
    switch (image_type) {
        case 1: // High contrast image
            image.setTo(Scalar(0, 0, 0));
            circle(image, Point(sz.width/2, sz.height/2), sz.width/3, Scalar(255, 255, 255), -1);
            break;
        case 2: // Textured image
            randu(image, 0, 255);
            for (int i = 0; i < 20; i++) {
                circle(image, Point(rand() % sz.width, rand() % sz.height), 
                       rand() % 50 + 10, Scalar(rand() % 255, rand() % 255, rand() % 255), -1);
            }
            break;
        case 3: // Gradient image
            for (int y = 0; y < sz.height; y++) {
                for (int x = 0; x < sz.width; x++) {
                    float intensity = (float)x / sz.width * 255;
                    image.at<Vec3b>(y, x) = Vec3b((uchar)intensity, (uchar)intensity, (uchar)intensity);
                }
            }
            break;
    }
    
    Mat mask = Mat::zeros(sz, CV_8UC1);
    Mat bgdModel, fgdModel;
    Rect rect(sz.width/4, sz.height/4, sz.width/2, sz.height/2);
    
    TEST_CYCLE()
    {
        Mat mask_copy = mask.clone();
        cv::metal::grabCut(image, mask_copy, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);
    }
    
    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test

#endif // HAVE_METAL 