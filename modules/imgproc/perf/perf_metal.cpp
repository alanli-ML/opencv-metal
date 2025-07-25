#include "perf_precomp.hpp"
#include "opencv2/imgproc/metal.hpp"

#ifdef HAVE_METAL

namespace opencv_test { namespace {

// Performance test for GaussianBlur
typedef perf::TestBaseWithParam<tuple<Size, int>> Imgproc_GaussianBlur;
PERF_TEST_P(Imgproc_GaussianBlur, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(3, 7)
    )
)
{
    Size sz = get<0>(GetParam());
    int ksize_val = get<1>(GetParam());
    Size ksize(ksize_val, ksize_val);
    int type = CV_32FC1;
    double sigma = 1.2;

    Mat src(sz, type);
    randu(src, 0, 1);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::GaussianBlur(d_src, d_dst, ksize, sigma);
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for Sobel
typedef perf::TestBaseWithParam<tuple<Size, int, int>> Imgproc_Sobel;
PERF_TEST_P(Imgproc_Sobel, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(1, 0),
        testing::Values(0, 1)
    )
)
{
    Size sz = get<0>(GetParam());
    int dx = get<1>(GetParam());
    int dy = get<2>(GetParam());
    int type = CV_32FC1;
    int ksize = 3;

    Mat src(sz, type);
    randu(src, 0, 1);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::Sobel(d_src, d_dst, -1, dx, dy, ksize);
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for resize
typedef perf::TestBaseWithParam<tuple<Size, int, double>> Imgproc_Resize;
PERF_TEST_P(Imgproc_Resize, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values((int)INTER_NEAREST, (int)INTER_LINEAR),
        testing::Values(0.5, 2.0)
    )
)
{
    Size sz = get<0>(GetParam());
    int interpolation = get<1>(GetParam());
    double scale = get<2>(GetParam());
    int type = CV_8UC4;
    Size dsize(cvRound(sz.width * scale), cvRound(sz.height * scale));

    Mat src(sz, type);
    randu(src, 0, 255);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::resize(d_src, d_dst, dsize, 0, 0, interpolation);
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for chained operations
typedef perf::TestBaseWithParam<Size> Imgproc_ChainedOps;

PERF_TEST_P(Imgproc_ChainedOps, Sync,
    testing::Values(szVGA, sz720p, sz1080p)
)
{
    Size sz = GetParam();
    int type = CV_32FC1;
    Size dsize(sz.width / 2, sz.height / 2);
    Size ksize(5, 5);
    double sigma = 1.2;

    Mat src(sz, type);
    randu(src, 0, 1);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_resized, d_blurred, d_sobel;

    TEST_CYCLE()
    {
        cv::metal::resize(d_src, d_resized, dsize, 0, 0, INTER_LINEAR);
        cv::metal::GaussianBlur(d_resized, d_blurred, ksize, sigma);
        cv::metal::Sobel(d_blurred, d_sobel, -1, 1, 0, 3);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Imgproc_ChainedOps, AsyncStream,
    testing::Values(szVGA, sz720p, sz1080p)
)
{
    Size sz = GetParam();
    int type = CV_32FC1;
    Size dsize(sz.width / 2, sz.height / 2);
    Size ksize(5, 5);
    double sigma = 1.2;

    Mat src(sz, type);
    randu(src, 0, 1);

    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_resized, d_blurred, d_sobel;
    cv::metal::Stream stream;

    TEST_CYCLE()
    {
        cv::metal::resize(d_src, d_resized, dsize, 0, 0, INTER_LINEAR, stream);
        cv::metal::GaussianBlur(d_resized, d_blurred, ksize, sigma, stream);
        cv::metal::Sobel(d_blurred, d_sobel, -1, 1, 0, 3, stream);
        stream.commitAndWait();
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for bilateralFilter
typedef perf::TestBaseWithParam<tuple<Size, int>> Imgproc_BilateralFilter;
PERF_TEST_P(Imgproc_BilateralFilter, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p),
        testing::Values(5, 9)
    )
)
{
    Size sz = get<0>(GetParam());
    int ksize = get<1>(GetParam());
    int type = CV_8UC4;
    float sigma_color = 15;
    float sigma_spatial = 15;

    Mat src(sz, type);
    randu(src, 0, 255);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_dst;

    TEST_CYCLE()
    {
        cv::metal::bilateralFilter(d_src, d_dst, ksize, sigma_color, sigma_spatial);
    }

    SANITY_CHECK_NOTHING();
}


}} // namespace
#endif // HAVE_METAL

// Connected Components performance tests
#ifdef HAVE_METAL

namespace opencv_test { namespace {

// Helper function to create test patterns for performance testing
static Mat createPerfTestImage(const std::string& pattern, Size size) {
    Mat img = Mat::zeros(size, CV_8UC1);
    cv::RNG rng(42); // Fixed seed for reproducibility
    
    if (pattern == "chessboard") {
        int blockSize = 8;
        for (int i = 0; i < size.height; i += blockSize) {
            for (int j = 0; j < size.width; j += blockSize) {
                if (((i / blockSize) + (j / blockSize)) % 2 == 0) {
                    Rect roi(j, i, std::min(blockSize, size.width - j), std::min(blockSize, size.height - i));
                    img(roi) = 255;
                }
            }
        }
    } else if (pattern == "sparse") {
        // Sparse random components for realistic workload
        for (int i = 0; i < size.height; i += 16) {
            for (int j = 0; j < size.width; j += 16) {
                if (rng.uniform(0, 4) == 0) { // 25% density
                    Rect roi(j, i, std::min(8, size.width - j), std::min(8, size.height - i));
                    img(roi) = 255;
                }
            }
        }
    } else if (pattern == "dense") {
        // Dense pattern with many small components
        for (int i = 0; i < size.height; i += 4) {
            for (int j = 0; j < size.width; j += 4) {
                if (rng.uniform(0, 2) == 0) { // 50% density
                    cv::circle(img, Point(j + 2, i + 2), 1, Scalar(255), -1);
                }
            }
        }
    }
    
    return img;
}

// Performance test for Connected Components
typedef perf::TestBaseWithParam<tuple<Size, std::string>> Imgproc_ConnectedComponents;
PERF_TEST_P(Imgproc_ConnectedComponents, Metal,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values("chessboard", "sparse", "dense")
    )
)
{
    Size sz = get<0>(GetParam());
    std::string pattern = get<1>(GetParam());

    Mat src = createPerfTestImage(pattern, sz);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels;

    TEST_CYCLE()
    {
        cv::metal::connectedComponents(d_src, d_labels, 8, CV_32S);
    }

    SANITY_CHECK_NOTHING();
}

// CPU baseline for comparison
PERF_TEST_P(Imgproc_ConnectedComponents, CPU_Baseline,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values("chessboard", "sparse", "dense")
    )
)
{
    Size sz = get<0>(GetParam());
    std::string pattern = get<1>(GetParam());

    Mat src = createPerfTestImage(pattern, sz);
    Mat labels;

    TEST_CYCLE()
    {
        cv::connectedComponents(src, labels, 8, CV_32S);
    }

    SANITY_CHECK_NOTHING();
}

// Stream vs Sync performance comparison (key Metal backend advantage)
typedef perf::TestBaseWithParam<Size> Imgproc_ConnectedComponents_StreamComparison;
PERF_TEST_P(Imgproc_ConnectedComponents_StreamComparison, Individual_Operations,
    testing::Values(szVGA, sz720p, sz1080p)
)
{
    Size sz = GetParam();
    Mat src1 = createPerfTestImage("sparse", sz);
    Mat src2 = createPerfTestImage("dense", sz);
    
    cv::metal::MetalMat d_src1(src1), d_src2(src2);
    cv::metal::MetalMat d_labels1, d_labels2;

    TEST_CYCLE()
    {
        // Individual operations (no stream)
        cv::metal::connectedComponents(d_src1, d_labels1, 8, CV_32S);
        cv::metal::connectedComponents(d_src2, d_labels2, 8, CV_32S);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Imgproc_ConnectedComponents_StreamComparison, Stream_Based_Operations,
    testing::Values(szVGA, sz720p, sz1080p)
)
{
    Size sz = GetParam();
    Mat src1 = createPerfTestImage("sparse", sz);
    Mat src2 = createPerfTestImage("dense", sz);
    
    cv::metal::MetalMat d_src1(src1), d_src2(src2);
    cv::metal::MetalMat d_labels1, d_labels2;

    TEST_CYCLE()
    {
        // Stream-based operations for improved pipeline efficiency
        cv::metal::Stream stream;
        cv::metal::connectedComponents(d_src1, d_labels1, 8, CV_32S, stream);
        cv::metal::connectedComponents(d_src2, d_labels2, 8, CV_32S, stream);
        stream.commitAndWait();
    }

    SANITY_CHECK_NOTHING();
}

// Performance test for K-means clustering
typedef perf::TestBaseWithParam<tuple<Size, int, int, int>> Imgproc_Kmeans;
PERF_TEST_P(Imgproc_Kmeans, Perf,
    testing::Combine(
        testing::Values(szVGA, sz720p, sz1080p),
        testing::Values(CV_8UC3, CV_8UC4),
        testing::Values(2, 4, 8),
        testing::Values(1, 3)
    )
)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    int K = get<2>(GetParam());
    int attempts = get<3>(GetParam());

    Mat src(sz, type);
    randu(src, 0, 255);
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels, d_centers;
    
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0);

    TEST_CYCLE()
    {
        cv::metal::kmeans(d_src, K, d_labels, criteria, attempts, KMEANS_RANDOM_CENTERS, d_centers);
    }

    SANITY_CHECK_NOTHING();
}

// Stream-based K-means performance test (critical for pipeline efficiency)
typedef perf::TestBaseWithParam<tuple<Size, int>> Imgproc_KmeansStream;
PERF_TEST_P(Imgproc_KmeansStream, StreamVsSync,
    testing::Combine(
        testing::Values(szVGA, sz720p),
        testing::Values(CV_8UC3, CV_8UC4)
    )
)
{
    Size sz = get<0>(GetParam());
    int type = get<1>(GetParam());
    
    Mat src1(sz, type), src2(sz, type);
    randu(src1, 0, 255);
    randu(src2, 0, 255);
    
    cv::metal::MetalMat d_src1(src1), d_src2(src2);
    cv::metal::MetalMat d_labels1, d_labels2;
    cv::metal::MetalMat d_centers1, d_centers2;
    
    TermCriteria criteria(TermCriteria::MAX_ITER, 5, 0); // Fast convergence for perf test
    int K = 4;
    int attempts = 1;

    TEST_CYCLE()
    {
        // Stream-based operations for improved pipeline efficiency
        cv::metal::Stream stream;
        cv::metal::kmeans(d_src1, K, d_labels1, criteria, attempts, KMEANS_RANDOM_CENTERS, d_centers1, stream);
        cv::metal::kmeans(d_src2, K, d_labels2, criteria, attempts, KMEANS_RANDOM_CENTERS, d_centers2, stream);
        stream.commitAndWait();
    }

    SANITY_CHECK_NOTHING();
}

// CPU vs Metal K-means comparison
PERF_TEST(Imgproc_Kmeans, CPUComparison)
{
    Size sz = szVGA;
    int type = CV_8UC3;
    int K = 4;
    int attempts = 1;
    
    Mat src(sz, type);
    randu(src, 0, 255);
    
    TermCriteria criteria(TermCriteria::MAX_ITER, 10, 0);
    
    // CPU performance baseline
    Mat data_cpu = src.reshape(1, src.rows * src.cols);
    Mat labels_cpu, centers_cpu;
    
    // CPU performance comparison (for baseline)
    cv::kmeans(data_cpu, K, labels_cpu, criteria, attempts, KMEANS_RANDOM_CENTERS, centers_cpu);
    
    // Metal performance
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels, d_centers;
    
    TEST_CYCLE()
    {
        cv::metal::kmeans(d_src, K, d_labels, criteria, attempts, KMEANS_RANDOM_CENTERS, d_centers);
    }
    
    SANITY_CHECK_NOTHING();
}

// Large scale K-means performance test
PERF_TEST(Imgproc_Kmeans, LargeScale)
{
    Size sz(1920, 1080); // Full HD resolution
    int type = CV_8UC3;
    int K = 16; // More clusters for complex segmentation
    int attempts = 1;
    
    Mat src(sz, type);
    randu(src, 0, 255);
    
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels, d_centers;
    
    TermCriteria criteria(TermCriteria::MAX_ITER, 15, 0);

    TEST_CYCLE()
    {
        cv::metal::kmeans(d_src, K, d_labels, criteria, attempts, KMEANS_RANDOM_CENTERS, d_centers);
    }

    SANITY_CHECK_NOTHING();
}

// Memory efficiency test for K-means
PERF_TEST(Imgproc_Kmeans, MemoryEfficiency)
{
    Size sz = sz720p;
    int type = CV_8UC4; // 4-channel for higher memory usage
    int K = 8;
    int attempts = 3; // Multiple attempts to test memory management

    Mat src(sz, type);
    randu(src, 0, 255);
    
    cv::metal::MetalMat d_src(src);
    cv::metal::MetalMat d_labels, d_centers;
    
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 8, 1.0);

    TEST_CYCLE()
    {
        cv::metal::kmeans(d_src, K, d_labels, criteria, attempts, KMEANS_RANDOM_CENTERS, d_centers);
    }

    SANITY_CHECK_NOTHING();
}

}} // namespace opencv_test

#endif // HAVE_METAL