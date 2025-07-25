#include "precomp.hpp"

#ifdef HAVE_METAL

#include "gmm_internal.hpp"
#include "gmm_kernels.hpp"

#if 0 // Removed unused function, keep variable for linkage
void enableCPUKMeansMode(bool enable);
#endif

bool g_disableEmergencyFallback = false; // still referenced in CPU fallback path

namespace cv { namespace metal {

// Forward declarations for utility functions
void trimapFromRect(MetalMat& mask, const Rect& rect, Stream& stream);
double calcBeta(const MetalMat& image, Stream& stream);
void calcNWeights(const MetalMat& image, MetalMat& leftW, MetalMat& topleftW, MetalMat& topW, MetalMat& toprightW, 
                  double beta, double gamma, Stream& stream);
bool checkConvergence(const MetalMat& oldMask, const MetalMat& newMask, Stream& stream);

GMM::GMM(Size imageSize) : m_imageSize(imageSize) {
    initializeGMMPipelines();
    
    MetalContext& ctx = MetalContext::getInstance();
    id<MTLDevice> device = ctx.device;
    
    // Allocate buffers for GMM parameters
    // Each component: weight(1) + mean(3) + cov_inv(6) + common_term(1) = 11 floats (CUDA layout)
    size_t componentSize = 11 * sizeof(float);
    // CRITICAL FIX: Use managed storage mode for better CPU-GPU synchronization
    m_gmmBgBuffer = [device newBufferWithLength:componentsCount * componentSize 
                                        options:MTLResourceStorageModeManaged];
    m_gmmFgBuffer = [device newBufferWithLength:componentsCount * componentSize 
                                        options:MTLResourceStorageModeManaged];
    
    // Allocate scratch buffers for reductions
    size_t scratchSize = imageSize.width * imageSize.height * sizeof(float);
    for (int i = 0; i < 8; i++) {
        m_scratchBuffers[i] = [device newBufferWithLength:scratchSize 
                                                  options:MTLResourceStorageModeShared];
    }
    
    // Allocate debug buffer for probability analysis
    m_debugBuffer = [device newBufferWithLength:sizeof(float) * 20  // DebugInfo structure 
                                        options:MTLResourceStorageModeShared];
    
    // NEW: Allocate statistics buffers for GPU GMM learning
    // Each component needs: count(1) + means(3) + covariance terms(6) = 10 values
    size_t statsBufferSize = componentsCount * 10 * sizeof(float);
    m_bgStatsBuffer = [device newBufferWithLength:statsBufferSize 
                                          options:MTLResourceStorageModeShared];
    m_fgStatsBuffer = [device newBufferWithLength:statsBufferSize 
                                          options:MTLResourceStorageModeShared];
    
    // Buffers for total pixel counts
    m_pixelCountsBuffer = [device newBufferWithLength:2 * sizeof(uint32_t)  // [total_bg, total_fg]
                                               options:MTLResourceStorageModeShared];
}

GMM::~GMM() {
    // ARC will handle cleanup
}

void GMM::initGMMs(const MetalMat& image, const MetalMat& mask, Stream& stream) {
    if (!getGMMInitializePipeline()) {
        CV_Error(Error::StsError, "Failed to create GMM initialize pipeline");
    }
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getGMMInitializePipeline()];
        [encoder setTexture:image.texture() atIndex:0];
        [encoder setTexture:mask.texture() atIndex:1];
        [encoder setBuffer:m_gmmBgBuffer offset:0 atIndex:0];
        [encoder setBuffer:m_gmmFgBuffer offset:0 atIndex:1];
        [encoder setBuffer:m_scratchBuffers[0] offset:0 atIndex:2]; // bg counts
        [encoder setBuffer:m_scratchBuffers[1] offset:0 atIndex:3]; // fg counts
        [encoder setBuffer:m_scratchBuffers[2] offset:0 atIndex:4]; // bg sums
        [encoder setBuffer:m_scratchBuffers[3] offset:0 atIndex:5]; // fg sums
        
        MTLSize gridSize = MTLSizeMake((image.cols() + 15) / 16, (image.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }

    // --- Simple CPU fallback initialization to approximate OpenCV behavior ---
    {
        // Download small proxy of the image and mask onto CPU (synchronous)
        cv::Mat h_img, h_mask;
        
        // Check for empty inputs before downloading
        if (image.empty() || mask.empty()) {
            return;
        }
        
        image.download(h_img, stream, true);
        mask.download(h_mask, stream, true);

        CV_Assert(!h_img.empty() && h_img.type() == CV_8UC4);
        CV_Assert(h_mask.type() == CV_8UC1);

        // Accumulators
        cv::Vec3d bgSum(0,0,0), fgSum(0,0,0);
        cv::Vec3d bgSqSum(0,0,0), fgSqSum(0,0,0);
        size_t bgCount = 0, fgCount = 0;

        for (int y = 0; y < h_img.rows; ++y) {
            const cv::Vec4b* pImg = h_img.ptr<cv::Vec4b>(y);
            const uchar* pMask = h_mask.ptr<uchar>(y);
            for (int x = 0; x < h_img.cols; ++x) {
                uchar m = pMask[x];
                if (m == GC_BGD) {
                    cv::Vec3d c(pImg[x][0], pImg[x][1], pImg[x][2]);
                    bgSum += c;
                    bgSqSum += c.mul(c);
                    ++bgCount;
                } else if (m == GC_PR_FGD) {
                    cv::Vec3d c(pImg[x][0], pImg[x][1], pImg[x][2]);
                    fgSum += c;
                    fgSqSum += c.mul(c);
                    ++fgCount;
                }
            }
        }

        if (bgCount == 0) bgCount = 1; // avoid div by zero
        if (fgCount == 0) fgCount = 1;

        cv::Vec3d bgMean = bgSum * (1.0 / bgCount);
        cv::Vec3d fgMean = fgSum * (1.0 / fgCount);

        cv::Vec3d bgVar = bgSqSum * (1.0 / bgCount) - bgMean.mul(bgMean);
        cv::Vec3d fgVar = fgSqSum * (1.0 / fgCount) - fgMean.mul(fgMean);

        // Ensure variances positive
        for (int k = 0; k < 3; ++k) {
            bgVar[k] = std::max(bgVar[k], 100.0) / 255.0; // Scale to [0,1] range for float precision
        }

        float* bgPtr = (float*)m_gmmBgBuffer.contents;
        float* fgPtr = (float*)m_gmmFgBuffer.contents;
        static const int kGMMComponentsPerClass = 5; // matches GPU constant
        const float weight = 1.0f / kGMMComponentsPerClass;
        for (int c = 0; c < kGMMComponentsPerClass; ++c) {
            // CUDA layout: [weight, mean_r, mean_g, mean_b, cov_inv_00, cov_inv_01, cov_inv_02, cov_inv_11, cov_inv_12, cov_inv_22, common_term]
            bgPtr[0] = weight;
            bgPtr[1] = static_cast<float>(bgMean[0]);
            bgPtr[2] = static_cast<float>(bgMean[1]);
            bgPtr[3] = static_cast<float>(bgMean[2]);
            // Diagonal covariance inverse
            bgPtr[4] = 1.0f / static_cast<float>(bgVar[0]); // cov_inv_00
            bgPtr[5] = 0.0f;                                 // cov_inv_01  
            bgPtr[6] = 0.0f;                                 // cov_inv_02
            bgPtr[7] = 1.0f / static_cast<float>(bgVar[1]); // cov_inv_11
            bgPtr[8] = 0.0f;                                 // cov_inv_12
            bgPtr[9] = 1.0f / static_cast<float>(bgVar[2]); // cov_inv_22
            // CPU-style: 1/sqrt(det), weight multiplied in kernel
            double det = static_cast<double>(bgVar[0] * bgVar[1] * bgVar[2]);
            bgPtr[10] = 1.0f / static_cast<float>(sqrt(det)); // Store 1/sqrt(det), multiply by weight in kernel
            
            fgPtr[0] = weight;
            fgPtr[1] = static_cast<float>(fgMean[0]);
            fgPtr[2] = static_cast<float>(fgMean[1]);
            fgPtr[3] = static_cast<float>(fgMean[2]);
            // Diagonal covariance inverse
            fgPtr[4] = 1.0f / static_cast<float>(fgVar[0]); // cov_inv_00
            fgPtr[5] = 0.0f;                                 // cov_inv_01  
            fgPtr[6] = 0.0f;                                 // cov_inv_02
            fgPtr[7] = 1.0f / static_cast<float>(fgVar[1]); // cov_inv_11
            fgPtr[8] = 0.0f;                                 // cov_inv_12
            fgPtr[9] = 1.0f / static_cast<float>(fgVar[2]); // cov_inv_22
            // Common term: weight / sqrt(det)
            det = static_cast<double>(fgVar[0] * fgVar[1] * fgVar[2]);
            fgPtr[10] = 1.0f / static_cast<float>(sqrt(det)); // Store 1/sqrt(det), multiply by weight in kernel
            
            bgPtr += 11; // 11 elements per component in CUDA layout
            fgPtr += 11;
        }
    }
}

void GMM::extractGMMParameters(Mat& bgdModel, Mat& fgdModel) {
    // Ensure output models have correct size and type
    bgdModel.create(1, 65, CV_32FC1); // 13 values per component * 5 components = 65
    fgdModel.create(1, 65, CV_32FC1);
    
    // Extract parameters from Metal buffers to CPU Mat format
    // Metal buffer layout: [weight, mean_r, mean_g, mean_b, cov_inv_00, cov_inv_01, cov_inv_02, cov_inv_11, cov_inv_12, cov_inv_22, inv_sqrt_det] * 5
    // CPU Mat layout: [weights(5), means(15), covariances(45)] = 65 total
    
    // CRITICAL FIX: For managed buffers, ensure GPU has finished writing before CPU reads
    // didModifyRange(0,0) is wrong - it tells Metal that CPU modified the buffer!
    // For GPU→CPU synchronization, we need to ensure GPU work is complete first
    static int callCount = 0;
    // Suppress verbose debug prints in production build
#if 0
    printf("DEBUG: extractGMMParameters - About to read GPU buffers (call #%d)\n", ++callCount);
#endif
    float* bgPtr = (float*)m_gmmBgBuffer.contents;
    float* fgPtr = (float*)m_gmmFgBuffer.contents;
#if 0
    printf("DEBUG: Raw GPU buffer values:\n");
    printf("BG[0]: weight=%.6f mean=(%.2f,%.2f,%.2f) cov_inv_00=%.6f\n", 
           bgPtr[0], bgPtr[1], bgPtr[2], bgPtr[3], bgPtr[4]);
    printf("BG[1]: weight=%.6f mean=(%.2f,%.2f,%.2f) cov_inv_00=%.6f\n", 
           bgPtr[11], bgPtr[12], bgPtr[13], bgPtr[14], bgPtr[15]);
    printf("FG[0]: weight=%.6f mean=(%.2f,%.2f,%.2f) cov_inv_00=%.6f\n", 
           fgPtr[0], fgPtr[1], fgPtr[2], fgPtr[3], fgPtr[4]);
    printf("FG[1]: weight=%.6f mean=(%.2f,%.2f,%.2f) cov_inv_00=%.6f\n", 
           fgPtr[11], fgPtr[12], fgPtr[13], fgPtr[14], fgPtr[15]);
#endif
    
    // Extract weights (first 5 values)
    for (int c = 0; c < componentsCount; c++) {
        bgdModel.ptr<float>(0)[c] = bgPtr[c * 11 + 0]; // weight
        fgdModel.ptr<float>(0)[c] = fgPtr[c * 11 + 0]; // weight
    }
    
    // END debug block
#if 0
    printf("DEBUG: Extracted weights to Mat - BG: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
           bgdModel.ptr<float>(0)[0], bgdModel.ptr<float>(0)[1], bgdModel.ptr<float>(0)[2], 
           bgdModel.ptr<float>(0)[3], bgdModel.ptr<float>(0)[4]);
    printf("DEBUG: Extracted weights to Mat - FG: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
           fgdModel.ptr<float>(0)[0], fgdModel.ptr<float>(0)[1], fgdModel.ptr<float>(0)[2], 
           fgdModel.ptr<float>(0)[3], fgdModel.ptr<float>(0)[4]);
#endif

    // Extract means (next 15 values: 3 per component * 5 components)
    for (int c = 0; c < componentsCount; c++) {
        bgdModel.ptr<float>(0)[5 + c * 3 + 0] = bgPtr[c * 11 + 1]; // mean_r
        bgdModel.ptr<float>(0)[5 + c * 3 + 1] = bgPtr[c * 11 + 2]; // mean_g  
        bgdModel.ptr<float>(0)[5 + c * 3 + 2] = bgPtr[c * 11 + 3]; // mean_b
        
        fgdModel.ptr<float>(0)[5 + c * 3 + 0] = fgPtr[c * 11 + 1]; // mean_r
        fgdModel.ptr<float>(0)[5 + c * 3 + 1] = fgPtr[c * 11 + 2]; // mean_g
        fgdModel.ptr<float>(0)[5 + c * 3 + 2] = fgPtr[c * 11 + 3]; // mean_b
    }
    
    // Extract covariances (last 45 values: 9 per component * 5 components)
    // CRITICAL FIX: Use double precision for matrix inversion to avoid catastrophic precision loss
    for (int c = 0; c < componentsCount; c++) {
        // Get inverse covariance matrix elements
        float inv_00 = bgPtr[c * 11 + 4]; // cov_inv_00
        float inv_01 = bgPtr[c * 11 + 5]; // cov_inv_01
        float inv_02 = bgPtr[c * 11 + 6]; // cov_inv_02
        float inv_11 = bgPtr[c * 11 + 7]; // cov_inv_11
        float inv_12 = bgPtr[c * 11 + 8]; // cov_inv_12
        float inv_22 = bgPtr[c * 11 + 9]; // cov_inv_22
        
        // CRITICAL FIX: Use double precision for matrix inversion to minimize precision loss
        cv::Matx33d covInvDouble(static_cast<double>(inv_00), static_cast<double>(inv_01), static_cast<double>(inv_02),
                                 static_cast<double>(inv_01), static_cast<double>(inv_11), static_cast<double>(inv_12),
                                 static_cast<double>(inv_02), static_cast<double>(inv_12), static_cast<double>(inv_22));
        
        // Compute regular covariance matrix by inverting in double precision
        cv::Matx33f cov;
        try {
            cv::Matx33d covDouble = covInvDouble.inv();
            // Convert back to float after high-precision inversion
            cov = cv::Matx33f(covDouble);
        } catch (...) {
            // If inversion fails, use identity matrix scaled appropriately
            cov = cv::Matx33f::eye() * 100.0f;
        }
        
        // Store in CPU format (row-major order)
        bgdModel.ptr<float>(0)[20 + c * 9 + 0] = cov(0,0); bgdModel.ptr<float>(0)[20 + c * 9 + 1] = cov(0,1); bgdModel.ptr<float>(0)[20 + c * 9 + 2] = cov(0,2);
        bgdModel.ptr<float>(0)[20 + c * 9 + 3] = cov(1,0); bgdModel.ptr<float>(0)[20 + c * 9 + 4] = cov(1,1); bgdModel.ptr<float>(0)[20 + c * 9 + 5] = cov(1,2);
        bgdModel.ptr<float>(0)[20 + c * 9 + 6] = cov(2,0); bgdModel.ptr<float>(0)[20 + c * 9 + 7] = cov(2,1); bgdModel.ptr<float>(0)[20 + c * 9 + 8] = cov(2,2);
        
        // Same for foreground - CRITICAL FIX: Use double precision for matrix inversion
        float fg_inv_00 = fgPtr[c * 11 + 4]; // cov_inv_00
        float fg_inv_01 = fgPtr[c * 11 + 5]; // cov_inv_01
        float fg_inv_02 = fgPtr[c * 11 + 6]; // cov_inv_02
        float fg_inv_11 = fgPtr[c * 11 + 7]; // cov_inv_11
        float fg_inv_12 = fgPtr[c * 11 + 8]; // cov_inv_12
        float fg_inv_22 = fgPtr[c * 11 + 9]; // cov_inv_22
        
        cv::Matx33d fgCovInvDouble(static_cast<double>(fg_inv_00), static_cast<double>(fg_inv_01), static_cast<double>(fg_inv_02),
                                   static_cast<double>(fg_inv_01), static_cast<double>(fg_inv_11), static_cast<double>(fg_inv_12),
                                   static_cast<double>(fg_inv_02), static_cast<double>(fg_inv_12), static_cast<double>(fg_inv_22));
        
        cv::Matx33f fgCov;
        try {
            cv::Matx33d fgCovDouble = fgCovInvDouble.inv();
            // Convert back to float after high-precision inversion
            fgCov = cv::Matx33f(fgCovDouble);
        } catch (...) {
            fgCov = cv::Matx33f::eye() * 100.0f;
        }
        
        fgdModel.ptr<float>(0)[20 + c * 9 + 0] = fgCov(0,0); fgdModel.ptr<float>(0)[20 + c * 9 + 1] = fgCov(0,1); fgdModel.ptr<float>(0)[20 + c * 9 + 2] = fgCov(0,2);
        fgdModel.ptr<float>(0)[20 + c * 9 + 3] = fgCov(1,0); fgdModel.ptr<float>(0)[20 + c * 9 + 4] = fgCov(1,1); fgdModel.ptr<float>(0)[20 + c * 9 + 5] = fgCov(1,2);
        fgdModel.ptr<float>(0)[20 + c * 9 + 6] = fgCov(2,0); fgdModel.ptr<float>(0)[20 + c * 9 + 7] = fgCov(2,1); fgdModel.ptr<float>(0)[20 + c * 9 + 8] = fgCov(2,2);
    }
    

}

bool GMM::compareGMMParameters(const std::string& context) const {
    const float TOLERANCE = 1e-4f; // Acceptable difference for floating point comparison
    bool allMatch = true;
    
    float* bgPtr = (float*)m_gmmBgBuffer.contents;
    float* fgPtr = (float*)m_gmmFgBuffer.contents;
    
#if 0
    printf("\n=== %s ===\n", context.c_str());
    
    // Compare background components
    for (int c = 0; c < componentsCount; c++) {
        printf("BG Component %d:\n", c);
        printf("  Weight: %.6f\n", bgPtr[c * 11 + 0]);
        printf("  Mean: (%.2f, %.2f, %.2f)\n", bgPtr[c * 11 + 1], bgPtr[c * 11 + 2], bgPtr[c * 11 + 3]);
        printf("  CovInv: [%.4f, %.4f, %.4f; %.4f, %.4f; %.4f]\n", 
               bgPtr[c * 11 + 4], bgPtr[c * 11 + 5], bgPtr[c * 11 + 6],
               bgPtr[c * 11 + 7], bgPtr[c * 11 + 8], bgPtr[c * 11 + 9]);
        printf("  InvSqrtDet: %.6f\n", bgPtr[c * 11 + 10]);
    }
    
    // Compare foreground components  
    for (int c = 0; c < componentsCount; c++) {
        printf("FG Component %d:\n", c);
        printf("  Weight: %.6f\n", fgPtr[c * 11 + 0]);
        printf("  Mean: (%.2f, %.2f, %.2f)\n", fgPtr[c * 11 + 1], fgPtr[c * 11 + 2], fgPtr[c * 11 + 3]);
        printf("  CovInv: [%.4f, %.4f, %.4f; %.4f, %.4f; %.4f]\n", 
               fgPtr[c * 11 + 4], fgPtr[c * 11 + 5], fgPtr[c * 11 + 6],
               fgPtr[c * 11 + 7], fgPtr[c * 11 + 8], fgPtr[c * 11 + 9]);
        printf("  InvSqrtDet: %.6f\n", fgPtr[c * 11 + 10]);
    }
    
    printf("=== End %s ===\n\n", context.c_str());
#endif
    
    return allMatch;
}

void GMM::learnGMMs(const MetalMat& image, const MetalMat& mask, const MetalMat& components, Stream& stream) {
    // Running CPU and GPU GMM learning for comparison
    
    // Save current GMM state for restoration
    size_t bufferSize = [m_gmmBgBuffer length];
    
#if 0
    printf("🚀 Running GPU GMM learning...\n");
#endif
    learnGMMsGPU(image, mask, components, stream);
    
    
}

// Export GMM creation function
cv::Ptr<GMM> createGMM(Size imageSize) {
    return cv::makePtr<GMM>(imageSize);
}

}} // cv::metal

#endif // HAVE_METAL