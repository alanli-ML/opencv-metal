#ifndef OPENCV_METALIMGPROC_GMM_INTERNAL_HPP
#define OPENCV_METALIMGPROC_GMM_INTERNAL_HPP

#include "precomp.hpp"

namespace cv { namespace metal {

// Forward declarations
class GMM;

// GMM creation function
cv::Ptr<GMM> createGMM(Size imageSize);

// GMM interface class
class GMM {
public:
    static const int componentsCount = 5;
    static const int totalComponents = 10; // 5 bg + 5 fg
    
    GMM(Size imageSize);
    ~GMM();
    
    void initGMMs(const MetalMat& image, const MetalMat& mask, Stream& stream);
    void assignGMMs(const MetalMat& image, const MetalMat& mask, MetalMat& components, Stream& stream);
    void learnGMMs(const MetalMat& image, const MetalMat& mask, const MetalMat& components, Stream& stream);
    void computeDataTerm(const MetalMat& image, const MetalMat& mask, MetalMat& bgTerm, MetalMat& fgTerm, Stream& stream);
    
    // NEW: GPU vs CPU comparison methods for testing
    void learnGMMsCPU(const MetalMat& image, const MetalMat& mask, const MetalMat& components, Stream& stream);
    void learnGMMsGPU(const MetalMat& image, const MetalMat& mask, const MetalMat& components, Stream& stream);
    bool compareGMMParameters(const std::string& context = "GMM comparison") const;
    
    // Extract learned GMM parameters to Mat objects for compatibility with CPU implementation
    void extractGMMParameters(Mat& bgdModel, Mat& fgdModel);
    
    // Access debug buffer for debugging probability calculations
    id<MTLBuffer> getDebugBuffer() const { return m_debugBuffer; }
    
    // Phase 4: Expose GMM buffers for GPU-resident parameters
    id<MTLBuffer> getBgBuffer() const { return m_gmmBgBuffer; }
    id<MTLBuffer> getFgBuffer() const { return m_gmmFgBuffer; }
    
    Size size() const { return m_imageSize; }
    
private:
    Size m_imageSize;
    id<MTLBuffer> m_gmmBgBuffer;
    id<MTLBuffer> m_gmmFgBuffer;
    id<MTLBuffer> m_scratchBuffers[8]; // Various scratch buffers for reductions
    id<MTLBuffer> m_debugBuffer; // Debug buffer for probability analysis
    
    // NEW: Buffers for GPU GMM learning
    id<MTLBuffer> m_bgStatsBuffer;     // Background statistics accumulation
    id<MTLBuffer> m_fgStatsBuffer;     // Foreground statistics accumulation  
    id<MTLBuffer> m_pixelCountsBuffer; // Total pixel counts [bg_total, fg_total]
    
    void updateGMMComponent(int componentId, bool isForeground, const MetalMat& image, 
                           const MetalMat& components, const MetalMat& mask, Stream& stream);
};

}} // cv::metal

#endif // OPENCV_METALIMGPROC_GMM_INTERNAL_HPP 