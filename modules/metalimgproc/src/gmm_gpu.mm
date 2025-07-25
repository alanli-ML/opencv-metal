#include "precomp.hpp"

#ifdef HAVE_METAL

#include "gmm_internal.hpp"
#include "gmm_kernels.hpp"

namespace cv { namespace metal {

void GMM::assignGMMs(const MetalMat& image, const MetalMat& mask, MetalMat& components, Stream& stream) {
    if (!getGMMAssignPipeline()) {
        CV_Error(Error::StsError, "Failed to create GMM assign pipeline");
    }
    

    
    // Force synchronization before kernel execution
    [m_gmmBgBuffer didModifyRange:NSMakeRange(0, m_gmmBgBuffer.length)];
    [m_gmmFgBuffer didModifyRange:NSMakeRange(0, m_gmmFgBuffer.length)];
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getGMMAssignPipeline()];
        [encoder setTexture:image.texture() atIndex:0];
        [encoder setTexture:mask.texture() atIndex:1];
        [encoder setTexture:components.texture() atIndex:2];
        [encoder setBuffer:m_gmmBgBuffer offset:0 atIndex:0];
        [encoder setBuffer:m_gmmFgBuffer offset:0 atIndex:1];
        [encoder setBuffer:m_debugBuffer offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake((image.cols() + 15) / 16, (image.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Note: Stream commit is now handled by caller to avoid double-commit errors
}

void GMM::learnGMMsGPU(const MetalMat& image, const MetalMat& mask, const MetalMat& components, Stream& stream) {
    if (!getGMMAccumulateStatsPipeline() || !getGMMCountPixelsPipeline() || !getGMMFinalizeParametersPipeline()) {
        CV_Error(Error::StsError, "Failed to create GMM learning pipelines");
    }
    
    // Clear statistics buffers
    memset([m_bgStatsBuffer contents], 0, [m_bgStatsBuffer length]);
    memset([m_fgStatsBuffer contents], 0, [m_fgStatsBuffer length]);
    
    // CRITICAL FIX: Channel ordering fixed to match CPU BGR order
    
    // PHASE 1: Accumulate statistics on GPU
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getGMMAccumulateStatsPipeline()];
        [encoder setTexture:image.texture() atIndex:0];
        [encoder setTexture:components.texture() atIndex:1];
        [encoder setTexture:mask.texture() atIndex:2];
        [encoder setBuffer:m_bgStatsBuffer offset:0 atIndex:0];
        [encoder setBuffer:m_fgStatsBuffer offset:0 atIndex:1];
        
        MTLSize gridSize = MTLSizeMake((image.cols() + 15) / 16, (image.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    
    // PHASE 2: Count total pixels on GPU
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getGMMCountPixelsPipeline()];
        [encoder setBuffer:m_bgStatsBuffer offset:0 atIndex:0];
        [encoder setBuffer:m_fgStatsBuffer offset:0 atIndex:1];
        [encoder setBuffer:m_pixelCountsBuffer offset:0 atIndex:2];
        
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
    }
    
    // Pixel counts computed on GPU - no need to sync for debug in production

    // Component sample counts and weights computed entirely on GPU for performance

    // PHASE 3: Finalize GMM parameters on GPU
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getGMMFinalizeParametersPipeline()];
        [encoder setBuffer:m_bgStatsBuffer offset:0 atIndex:0];
        [encoder setBuffer:m_fgStatsBuffer offset:0 atIndex:1];
        [encoder setBuffer:m_gmmBgBuffer offset:0 atIndex:2];
        [encoder setBuffer:m_gmmFgBuffer offset:0 atIndex:3];
        [encoder setBuffer:m_pixelCountsBuffer offset:0 atIndex:4];
        uint32_t kGmmComponents = 5; // Constant: always 5 components per class
        [encoder setBytes:&kGmmComponents length:sizeof(uint32_t) atIndex:5]; // Pass componentsCount as constant
        // Component totals calculated entirely on GPU - no CPU-side calculations needed
        
        MTLSize gridSize = MTLSizeMake(5, 1, 1); // Always 5 components
        MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Intermediate calculations performed on GPU - no need to sync for debug in production
    // GPU calculations complete - debug output removed for performance
    
    // NO commit, NO wait. The caller is responsible for synchronization.
}

void GMM::computeDataTerm(const MetalMat& image, const MetalMat& mask, MetalMat& bgTerm, MetalMat& fgTerm, Stream& stream) {
    if (!getGMMDataTermPipeline()) {
        CV_Error(Error::StsError, "Failed to create GMM data term pipeline");
    }
    
    bgTerm.create(image.size(), CV_32FC1);
    fgTerm.create(image.size(), CV_32FC1);
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getGMMDataTermPipeline()];
        [encoder setTexture:image.texture() atIndex:0];
        [encoder setTexture:mask.texture() atIndex:1];
        [encoder setTexture:bgTerm.texture() atIndex:2];
        [encoder setTexture:fgTerm.texture() atIndex:3];
        [encoder setBuffer:m_gmmBgBuffer offset:0 atIndex:0];
        [encoder setBuffer:m_gmmFgBuffer offset:0 atIndex:1];
        
        MTLSize gridSize = MTLSizeMake((image.cols() + 15) / 16, (image.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
}

void GMM::updateGMMComponent(int componentId, bool isForeground, const MetalMat& image, 
                            const MetalMat& components, const MetalMat& mask, Stream& stream) {
    if (!getGMMReductionPipeline()) {
        CV_Error(Error::StsError, "Failed to create GMM reduction pipeline");
    }
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getGMMReductionPipeline()];
        [encoder setTexture:image.texture() atIndex:0];
        [encoder setTexture:components.texture() atIndex:1];
        [encoder setTexture:mask.texture() atIndex:2];
        [encoder setBuffer:m_scratchBuffers[0] offset:0 atIndex:0]; // counts
        [encoder setBuffer:m_scratchBuffers[1] offset:0 atIndex:1]; // mean_sums_r
        [encoder setBuffer:m_scratchBuffers[2] offset:0 atIndex:2]; // mean_sums_g
        [encoder setBuffer:m_scratchBuffers[3] offset:0 atIndex:3]; // mean_sums_b
        [encoder setBytes:&componentId length:sizeof(int) atIndex:4];
        
        MTLSize gridSize = MTLSizeMake((image.cols() + 15) / 16, (image.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Note: Full implementation would finalize the GMM parameters on CPU
    // after reading back the reduction results
}

}} // cv::metal

#endif // HAVE_METAL 