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
    
    // DEBUG: Sync and examine statistics after accumulation
    stream.syncCPU();
    printf("\n=== DEBUG: GPU Statistics After Accumulation ===\n");
    float* bgStats = (float*)[m_bgStatsBuffer contents];
    float* fgStats = (float*)[m_fgStatsBuffer contents];
    
    for (int c = 0; c < 5; c++) {
        printf("BG Component %d: count=%.1f sum=(%.1f,%.1f,%.1f) sum_sq=(%.1f,%.1f,%.1f,%.1f,%.1f,%.1f)\n", 
               c, bgStats[c*10+0], bgStats[c*10+1], bgStats[c*10+2], bgStats[c*10+3],
               bgStats[c*10+4], bgStats[c*10+5], bgStats[c*10+6], bgStats[c*10+7], bgStats[c*10+8], bgStats[c*10+9]);
        printf("FG Component %d: count=%.1f sum=(%.1f,%.1f,%.1f) sum_sq=(%.1f,%.1f,%.1f,%.1f,%.1f,%.1f)\n", 
               c, fgStats[c*10+0], fgStats[c*10+1], fgStats[c*10+2], fgStats[c*10+3],
               fgStats[c*10+4], fgStats[c*10+5], fgStats[c*10+6], fgStats[c*10+7], fgStats[c*10+8], fgStats[c*10+9]);
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
    
    // DEBUG: Check pixel counts
    stream.syncCPU();
    uint32_t* pixelCounts = (uint32_t*)[m_pixelCountsBuffer contents];
    printf("GPU Pixel Counts: BG=%u FG=%u\n", pixelCounts[0], pixelCounts[1]);

    // CRITICAL FIX: Pre-calculate total component sample counts like CPU does
    // CPU uses totalSampleCount = sum of all component sample counts, NOT total image pixels
    
    float totalBgComponentSamples = 0;
    float totalFgComponentSamples = 0;
    for (int i = 0; i < 5; i++) {
        totalBgComponentSamples += bgStats[i * 10 + 0];  // Sum all BG component counts
        totalFgComponentSamples += fgStats[i * 10 + 0];  // Sum all FG component counts
    }
    
    printf("GPU Component Sample Totals: BG=%.1f FG=%.1f\n", totalBgComponentSamples, totalFgComponentSamples);
    
    // DEBUG: Print individual component counts and expected weights for comparison
    printf("GPU Component Details:\n");
    for (int i = 0; i < 5; i++) {
        float bgCount = bgStats[i * 10 + 0];
        float fgCount = fgStats[i * 10 + 0];
        float bgWeight = (totalBgComponentSamples > 0) ? bgCount / totalBgComponentSamples : 0.0f;
        float fgWeight = (totalFgComponentSamples > 0) ? fgCount / totalFgComponentSamples : 0.0f;
        printf("  Component %d: BG count=%.1f weight=%.6f, FG count=%.1f weight=%.6f\n", 
               i, bgCount, bgWeight, fgCount, fgWeight);
    }

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
        [encoder setBytes:&totalBgComponentSamples length:sizeof(float) atIndex:6]; // Pass pre-calculated BG total
        [encoder setBytes:&totalFgComponentSamples length:sizeof(float) atIndex:7]; // Pass pre-calculated FG total
        
        MTLSize gridSize = MTLSizeMake(5, 1, 1); // Always 5 components
        MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // DEBUG: Show intermediate calculations for each component
    stream.syncCPU();
    printf("\n=== DEBUG: GPU Intermediate Calculations ===\n");
    
    for (int c = 0; c < 5; c++) {
        if (bgStats[c*10+0] > 0 && pixelCounts[0] > 0) {
            float count = bgStats[c*10+0];
            float mean_r = bgStats[c*10+1]/count;
            float mean_g = bgStats[c*10+2]/count;
            float mean_b = bgStats[c*10+3]/count;
            
            float cov_rr = bgStats[c*10+4]/count - mean_r*mean_r;
            float cov_rg = bgStats[c*10+5]/count - mean_r*mean_g;
            float cov_rb = bgStats[c*10+6]/count - mean_r*mean_b;
            float cov_gg = bgStats[c*10+7]/count - mean_g*mean_g;
            float cov_gb = bgStats[c*10+8]/count - mean_g*mean_b;
            float cov_bb = bgStats[c*10+9]/count - mean_b*mean_b;
            
            float det = cov_rr * (cov_gg * cov_bb - cov_gb * cov_gb) -
                       cov_rg * (cov_rg * cov_bb - cov_gb * cov_rb) +
                       cov_rb * (cov_rg * cov_gb - cov_gg * cov_rb);
            
            printf("GPU BG[%d]: count=%.1f mean=(%.2f,%.2f,%.2f) cov=[%.6f,%.6f,%.6f;%.6f,%.6f;%.6f] det=%.9f\n",
                   c, count, mean_r, mean_g, mean_b, 
                   cov_rr, cov_rg, cov_rb, cov_gg, cov_gb, cov_bb, det);
        }
        
        if (fgStats[c*10+0] > 0 && pixelCounts[1] > 0) {
            float count = fgStats[c*10+0];
            float mean_r = fgStats[c*10+1]/count;
            float mean_g = fgStats[c*10+2]/count;
            float mean_b = fgStats[c*10+3]/count;
            
            float cov_rr = fgStats[c*10+4]/count - mean_r*mean_r;
            float cov_rg = fgStats[c*10+5]/count - mean_r*mean_g;
            float cov_rb = fgStats[c*10+6]/count - mean_r*mean_b;
            float cov_gg = fgStats[c*10+7]/count - mean_g*mean_g;
            float cov_gb = fgStats[c*10+8]/count - mean_g*mean_b;
            float cov_bb = fgStats[c*10+9]/count - mean_b*mean_b;
            
            float det = cov_rr * (cov_gg * cov_bb - cov_gb * cov_gb) -
                       cov_rg * (cov_rg * cov_bb - cov_gb * cov_rb) +
                       cov_rb * (cov_rg * cov_gb - cov_gg * cov_rb);
            
            printf("GPU FG[%d]: count=%.1f mean=(%.2f,%.2f,%.2f) cov=[%.6f,%.6f,%.6f;%.6f,%.6f;%.6f] det=%.9f\n",
                   c, count, mean_r, mean_g, mean_b,
                   cov_rr, cov_rg, cov_rb, cov_gg, cov_gb, cov_bb, det);
        }
    }
    
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