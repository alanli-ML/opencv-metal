#include "precomp.hpp"

#ifdef HAVE_METAL

namespace cv { namespace metal {

// MSL kernel source for utility functions
static const char* utilityKernelsSource = R"(
#include <metal_stdlib>
using namespace metal;

// Trimap from rect kernel
kernel void trimapFromRectKernel(texture2d<uint, access::write> mask [[texture(0)]],
                                constant uint2& rect_origin [[buffer(0)]],
                                constant uint2& rect_size [[buffer(1)]],
                                uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= mask.get_width() || gid.y >= mask.get_height()) return;
    
    uint value_int = 0; // GC_BGD (sure background)
    
    // Check if pixel is inside rect
    if (gid.x >= rect_origin.x && gid.x < rect_origin.x + rect_size.x &&
        gid.y >= rect_origin.y && gid.y < rect_origin.y + rect_size.y) {
        value_int = 3; // GC_PR_FGD (probable foreground)
    } 
    // Everything else is GC_BGD (sure background) - match CPU exactly
    
    // ALWAYS write to ensure all pixels are initialized (Metal textures may contain garbage)
    // For MTLPixelFormatR8Uint texture, we write integer values directly
    mask.write(value_int, gid);
}

// Edge weight calculation helper
float edgeWeight(float3 center, float3 neighbor, float gamma, float beta, float recipDistance) {
    float3 diff = center - neighbor;
    float distSq = dot(diff, diff);
    // Scale the distance squared by 255² to match the beta calculation scaling
    return recipDistance * gamma * exp(-beta * distSq * (255.0f * 255.0f));
}

// Pairwise weight calculation kernel
kernel void edgeCuesKernel(texture2d<float, access::sample> image [[texture(0)]],
                          texture2d<float, access::write> leftWeights [[texture(1)]],
                          texture2d<float, access::write> topLeftWeights [[texture(2)]],
                          texture2d<float, access::write> topWeights [[texture(3)]],
                          texture2d<float, access::write> topRightWeights [[texture(4)]],
                          constant float& gamma [[buffer(0)]],
                          device const float* betaBuffer [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    if (gid.x >= image.get_width() || gid.y >= image.get_height()) return;
    
    float beta = *betaBuffer; // Read beta from GPU buffer
    float4 centerPixel = image.sample(s, float2(gid) + 0.5f);
    float3 center = centerPixel.rgb; // Work with normalized [0,1] values
    
    // Left neighbor
    if (gid.x > 0) {
        float4 leftPixel = image.sample(s, float2(gid.x - 1, gid.y) + 0.5f);
        float weight = edgeWeight(center, leftPixel.rgb, gamma, beta, 1.0f);
        leftWeights.write(float4(weight,0,0,0), gid);
    } else {
        leftWeights.write(0.0f, gid);
    }
    
    // Top-left neighbor
    if (gid.x > 0 && gid.y > 0) {
        float4 topLeftPixel = image.sample(s, float2(gid.x - 1, gid.y - 1) + 0.5f);
        float weight = edgeWeight(center, topLeftPixel.rgb, gamma, beta, 1.0f / sqrt(2.0f));
        topLeftWeights.write(float4(weight,0,0,0), gid);
    } else {
        topLeftWeights.write(0.0f, gid);
    }
    
    // Top neighbor
    if (gid.y > 0) {
        float4 topPixel = image.sample(s, float2(gid.x, gid.y - 1) + 0.5f);
        float weight = edgeWeight(center, topPixel.rgb, gamma, beta, 1.0f);
        topWeights.write(float4(weight,0,0,0), gid);
    } else {
        topWeights.write(0.0f, gid);
    }
    
    // Top-right neighbor
    if (gid.x < image.get_width() - 1 && gid.y > 0) {
        float4 topRightPixel = image.sample(s, float2(gid.x + 1, gid.y - 1) + 0.5f);
        float weight = edgeWeight(center, topRightPixel.rgb, gamma, beta, 1.0f / sqrt(2.0f));
        topRightWeights.write(float4(weight,0,0,0), gid);
    } else {
        topRightWeights.write(0.0f, gid);
    }
}

// Beta calculation for edge weights
kernel void betaCalculationKernel(texture2d<float, access::sample> image [[texture(0)]],
                                 device atomic<float>* betaSum [[buffer(0)]],
                                 device atomic<uint>* pixelCount [[buffer(1)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    if (gid.x >= image.get_width() || gid.y >= image.get_height()) return;
    
    float4 centerPixel = image.sample(s, float2(gid) + 0.5f);
    float3 center = centerPixel.rgb; // Work with normalized [0,1] values directly
    float sum = 0.0f;
    uint count = 0;
    
    // Check 4-connected neighbors
    // Match CPU exactly: left, top-left, top, top-right (no right or bottom to avoid double-counting)
    if (gid.x > 0) { // left
        float4 leftPixel = image.sample(s, float2(gid.x - 1, gid.y) + 0.5f);
        float3 diff = center - leftPixel.rgb;
        // Scale by 255² to match CPU's integer arithmetic when needed for final beta calculation
        sum += dot(diff, diff) * (255.0f * 255.0f);
        count++;
    }
    
    if (gid.y > 0 && gid.x > 0) { // top-left  
        float4 topLeftPixel = image.sample(s, float2(gid.x - 1, gid.y - 1) + 0.5f);
        float3 diff = center - topLeftPixel.rgb;
        sum += dot(diff, diff) * (255.0f * 255.0f);
        count++;
    }
    
    if (gid.y > 0) { // top
        float4 topPixel = image.sample(s, float2(gid.x, gid.y - 1) + 0.5f);
        float3 diff = center - topPixel.rgb;
        sum += dot(diff, diff) * (255.0f * 255.0f);
        count++;
    }
    
    if (gid.y > 0 && gid.x < image.get_width() - 1) { // top-right
        float4 topRightPixel = image.sample(s, float2(gid.x + 1, gid.y - 1) + 0.5f);
        float3 diff = center - topRightPixel.rgb;
        sum += dot(diff, diff) * (255.0f * 255.0f);
        count++;
    }
    
    atomic_fetch_add_explicit(betaSum, sum, memory_order_relaxed);
    atomic_fetch_add_explicit(pixelCount, count, memory_order_relaxed);
}

// Apply matte kernel for visualization (optional)
kernel void applyMatteKernel(texture2d<float, access::sample> image [[texture(0)]],
                            texture2d<uint, access::read> mask [[texture(1)]],
                            texture2d<float, access::write> result [[texture(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    if (gid.x >= result.get_width() || gid.y >= result.get_height()) return;
    
    float4 imagePixel = image.sample(s, float2(gid) + 0.5f);
    uint maskValue = mask.read(gid).x;
    
    // Apply alpha based on mask value
    float alpha = (maskValue == 1 || maskValue == 3) ? 1.0f : 0.0f; // Foreground pixels
    result.write(float4(imagePixel.rgb, alpha), gid);
}

// Finalize beta calculation kernel (eliminates CPU synchronization)
kernel void finalizeBetaKernel(device const float* betaSum [[buffer(0)]],
                              device const uint* pixelCount [[buffer(1)]],
                              device float* outBeta [[buffer(2)]],
                              constant uint& width [[buffer(3)]],
                              constant uint& height [[buffer(4)]],
                              uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return; // Single thread kernel
    
    float totalSum = *betaSum;
    uint totalCount = *pixelCount;
    
    if (totalCount == 0 || totalSum <= 1e-10f) {
        *outBeta = 0.0f;
        return;
    }
    
    // CPU formula: beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2));
    float expectedCount = 4.0 * width * height - 3.0 * width - 3.0 * height + 2.0;
    *outBeta = 1.0f / (2.0f * totalSum / expectedCount);
}

// Convergence check kernel
kernel void convergenceCheckKernel(texture2d<uint, access::read> oldMask [[texture(0)]],
                                  texture2d<uint, access::read> newMask [[texture(1)]],
                                  device atomic<uint>* changedCount [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= oldMask.get_width() || gid.y >= oldMask.get_height()) return;
    
    uint oldValue = oldMask.read(gid).x;
    uint newValue = newMask.read(gid).x;
    
    if (oldValue != newValue) {
        atomic_fetch_add_explicit(changedCount, 1u, memory_order_relaxed);
    }
}
)";

// Static pipeline state cache
static id<MTLComputePipelineState> g_trimapFromRectPipeline = nil;
static id<MTLComputePipelineState> g_edgeCuesPipeline = nil;
static id<MTLComputePipelineState> g_betaCalculationPipeline = nil;
static id<MTLComputePipelineState> g_applyMattePipeline = nil;
static id<MTLComputePipelineState> g_finalizeBetaPipeline = nil;
static id<MTLComputePipelineState> g_convergenceCheckPipeline = nil;
static dispatch_once_t g_utilityPipelinesOnce = 0;

// Initialize utility pipelines
static void initializeUtilityPipelines() {
    dispatch_once(&g_utilityPipelinesOnce, ^{
        @autoreleasepool {
            NSError *error = nil;
            MetalContext& ctx = MetalContext::getInstance();
            id<MTLDevice> device = ctx.device;
            
            NSString *kernelSource = [NSString stringWithUTF8String:utilityKernelsSource];
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
            
            if (error) {
                NSLog(@"Error compiling utility kernels: %@", error.localizedDescription);
                return;
            }
            
            // Create compute pipeline states
            id<MTLFunction> trimapFunc = [library newFunctionWithName:@"trimapFromRectKernel"];
            g_trimapFromRectPipeline = [device newComputePipelineStateWithFunction:trimapFunc error:&error];
            
            if (error) {
                NSLog(@"Error creating trimapFromRectKernel pipeline: %@", error.localizedDescription);
            } else if (!g_trimapFromRectPipeline) {
                NSLog(@"trimapFromRectKernel pipeline is nil");
            } else {
                // Pipeline created successfully - no need to log every time
            }
            
            id<MTLFunction> edgeCuesFunc = [library newFunctionWithName:@"edgeCuesKernel"];
            g_edgeCuesPipeline = [device newComputePipelineStateWithFunction:edgeCuesFunc error:&error];
            
            id<MTLFunction> betaCalcFunc = [library newFunctionWithName:@"betaCalculationKernel"];
            g_betaCalculationPipeline = [device newComputePipelineStateWithFunction:betaCalcFunc error:&error];
            
            id<MTLFunction> finalizeBetaFunc = [library newFunctionWithName:@"finalizeBetaKernel"];
            g_finalizeBetaPipeline = [device newComputePipelineStateWithFunction:finalizeBetaFunc error:&error];
            
            id<MTLFunction> applyMatteFunc = [library newFunctionWithName:@"applyMatteKernel"];
            g_applyMattePipeline = [device newComputePipelineStateWithFunction:applyMatteFunc error:&error];
            
            id<MTLFunction> convergenceFunc = [library newFunctionWithName:@"convergenceCheckKernel"];
            g_convergenceCheckPipeline = [device newComputePipelineStateWithFunction:convergenceFunc error:&error];
        }
    });
}

// Utility function implementations
void trimapFromRect(MetalMat& mask, const Rect& rect, Stream& stream) {
    initializeUtilityPipelines();
    
    if (!g_trimapFromRectPipeline) {
        CV_Error(Error::StsError, "Failed to create trimap from rect pipeline");
    }
    
    // Clamp rectangle to image bounds to handle invalid rectangles gracefully
    int img_width = mask.cols();
    int img_height = mask.rows();
    
    // Ensure rect has some valid area within image bounds, but leave background border
    // Minimum 1-pixel border on each side to ensure background samples exist
    int min_border = 1;
    
    // Clamp and ensure minimum dimensions
    int clamped_x = std::max(min_border, std::min(rect.x, img_width - min_border - 1));
    int clamped_y = std::max(min_border, std::min(rect.y, img_height - min_border - 1));
    int clamped_x2 = std::max(clamped_x + 1, std::min(rect.x + rect.width, img_width - min_border));
    int clamped_y2 = std::max(clamped_y + 1, std::min(rect.y + rect.height, img_height - min_border));
    
    int clamped_width = clamped_x2 - clamped_x;
    int clamped_height = clamped_y2 - clamped_y;
    
    // Use encoder accessor for proper pipeline integration - no command buffer storage
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:g_trimapFromRectPipeline];
        [encoder setTexture:mask.texture() atIndex:0];
        
        uint32_t rectData[4] = {
            static_cast<uint32_t>(clamped_x),
            static_cast<uint32_t>(clamped_y),
            static_cast<uint32_t>(clamped_width),
            static_cast<uint32_t>(clamped_height)
        };
        [encoder setBytes:rectData length:sizeof(uint32_t) * 2 atIndex:0]; // origin
        [encoder setBytes:&rectData[2] length:sizeof(uint32_t) * 2 atIndex:1]; // size
        
        MTLSize gridSize = MTLSizeMake((mask.cols() + 15) / 16, (mask.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Note: Let caller handle stream synchronization - don't commit here
}

// Async beta calculation that returns buffer instead of computed value
id<MTLBuffer> calcBetaAsync(const MetalMat& image, Stream& stream) {
    initializeUtilityPipelines();
    
    if (!g_betaCalculationPipeline || !g_finalizeBetaPipeline) {
        CV_Error(Error::StsError, "Failed to create beta calculation pipelines");
    }
    
    // Create buffers for reduction
    MetalContext& ctx = MetalContext::getInstance();
    id<MTLDevice> device = ctx.device;
    
    id<MTLBuffer> betaSumBuffer = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pixelCountBuffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> betaResultBuffer = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    
    // Initialize to zero
    *((float*)betaSumBuffer.contents) = 0.0f;
    *((uint32_t*)pixelCountBuffer.contents) = 0;
    
    // Phase 1: Calculate sum and count
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:g_betaCalculationPipeline];
        [encoder setTexture:image.texture() atIndex:0];
        [encoder setBuffer:betaSumBuffer offset:0 atIndex:0];
        [encoder setBuffer:pixelCountBuffer offset:0 atIndex:1];
        
        MTLSize gridSize = MTLSizeMake(image.cols(), image.rows(), 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Phase 2: Finalize beta calculation
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:g_finalizeBetaPipeline];
        [encoder setBuffer:betaSumBuffer offset:0 atIndex:0];
        [encoder setBuffer:pixelCountBuffer offset:0 atIndex:1];
        [encoder setBuffer:betaResultBuffer offset:0 atIndex:2];
        uint32_t width = image.cols();
        uint32_t height = image.rows();
        [encoder setBytes:&width length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&height length:sizeof(uint32_t) atIndex:4];
        
        [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [encoder endEncoding];
    }
    
    return betaResultBuffer;
}

double calcBeta(const MetalMat& image, Stream& stream) {
    #if 0
    printf("DEBUG: calcBeta - Starting\n");
    initializeUtilityPipelines();
    printf("DEBUG: calcBeta - Pipelines initialized\n");
    printf("DEBUG: calcBeta - Pipeline exists\n");
    printf("DEBUG: calcBeta - Got Metal context and device\n");
    printf("DEBUG: calcBeta - Created buffers\n");
    printf("DEBUG: calcBeta - Initialized buffer contents\n");
    printf("DEBUG: calcBeta - About to create compute encoder\n");
    printf("DEBUG: calcBeta - Created compute encoder: %p\n", encoder);
    printf("DEBUG: calcBeta - Set pipeline state\n");
    printf("DEBUG: calcBeta - Set texture\n");
    printf("DEBUG: calcBeta - Set buffers\n");
    printf("DEBUG: calcBeta - Grid size: %lux%lu, Threadgroup: %lux%lu\n", 
           gridSize.width, gridSize.height, threadgroupSize.width, threadgroupSize.height);
    printf("DEBUG: calcBeta - Dispatched threads\n");
    printf("DEBUG: calcBeta - Ended encoding\n");
    printf("DEBUG: calcBeta - About to sync stream (outside pool)\n");
    printf("DEBUG: calcBeta - Stream synced (outside pool)\n");
    printf("DEBUG: calcBeta - Exited autoreleasepool with valid results\n");
    #endif
    
    initializeUtilityPipelines();
    
    if (!g_betaCalculationPipeline) {
        CV_Error(Error::StsError, "Failed to create beta calculation pipeline");
    }
    
    // Create atomic buffers for reduction
    MetalContext& ctx = MetalContext::getInstance();
    id<MTLDevice> device = ctx.device;
    
    id<MTLBuffer> betaSumBuffer = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pixelCountBuffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    
    // Initialize to zero
    *((float*)betaSumBuffer.contents) = 0.0f;
    *((uint32_t*)pixelCountBuffer.contents) = 0;
    
    // Use caller's stream to integrate with their pipeline
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        
        [encoder setComputePipelineState:g_betaCalculationPipeline];
        
        [encoder setTexture:image.texture() atIndex:0];
        
        [encoder setBuffer:betaSumBuffer offset:0 atIndex:0];
        [encoder setBuffer:pixelCountBuffer offset:0 atIndex:1];
        
        // Use dispatchThreads to ensure exactly one thread per pixel
        MTLSize gridSize = MTLSizeMake(image.cols(), image.rows(), 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        
        [encoder endEncoding];
        
    }
    
    // CRITICAL FIX: Sync OUTSIDE autorelease pool to avoid buffer deallocation
    stream.syncCPU();
    
    float totalSum = *((float*)betaSumBuffer.contents);
    uint32_t totalCount = *((uint32_t*)pixelCountBuffer.contents);
    
    if (totalCount == 0 || totalSum <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }
    
    // CPU formula: beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );
    int rows = image.rows();
    int cols = image.cols();
    double expectedCount = 4.0 * cols * rows - 3.0 * cols - 3.0 * rows + 2.0;
    double beta = 1.0 / (2.0 * totalSum / expectedCount);
    printf("Metal Beta calculation: totalSum=%.6f expectedCount=%.0f beta=%.6f\n", 
           totalSum, expectedCount, beta);
    return beta;
}

void calcNWeights(const MetalMat& image, MetalMat& leftW, MetalMat& topleftW, MetalMat& topW, MetalMat& toprightW, 
                  id<MTLBuffer> betaBuffer, double gamma, Stream& stream) {
    initializeUtilityPipelines();
    
    if (!g_edgeCuesPipeline) {
        CV_Error(Error::StsError, "Failed to create edge cues pipeline");
    }
    
    Size imageSize = image.size();
    leftW.create(imageSize, CV_32FC4);
    topleftW.create(imageSize, CV_32FC4);
    topW.create(imageSize, CV_32FC4);
    toprightW.create(imageSize, CV_32FC4);
    
    // Use encoder accessor to integrate with caller's stream
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:g_edgeCuesPipeline];
        [encoder setTexture:image.texture() atIndex:0];
        [encoder setTexture:leftW.texture() atIndex:1];
        [encoder setTexture:topleftW.texture() atIndex:2];
        [encoder setTexture:topW.texture() atIndex:3];
        [encoder setTexture:toprightW.texture() atIndex:4];
        
        float gammaFloat = static_cast<float>(gamma);
        [encoder setBytes:&gammaFloat length:sizeof(float) atIndex:0];
        [encoder setBuffer:betaBuffer offset:0 atIndex:1]; // Use beta buffer instead of constant
        
        MTLSize gridSize = MTLSizeMake((image.cols() + 15) / 16, (image.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    // Note: Let caller handle stream synchronization - don't commit here
}

// Async convergence check - returns buffer with changed pixel count
id<MTLBuffer> checkConvergenceAsync(const MetalMat& oldMask, const MetalMat& newMask, Stream& stream) {
    initializeUtilityPipelines();
    
    if (!g_convergenceCheckPipeline) {
        CV_Error(Error::StsError, "Failed to create convergence check pipeline");
    }
    
    // Create atomic buffer for changed pixel count
    MetalContext& ctx = MetalContext::getInstance();
    id<MTLDevice> device = ctx.device;
    id<MTLBuffer> changedCountBuffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    *((uint32_t*)changedCountBuffer.contents) = 0;
    
    // Use caller's stream to integrate with their pipeline
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:g_convergenceCheckPipeline];
        [encoder setTexture:oldMask.texture() atIndex:0];
        [encoder setTexture:newMask.texture() atIndex:1];
        [encoder setBuffer:changedCountBuffer offset:0 atIndex:0];
        
        MTLSize gridSize = MTLSizeMake((oldMask.cols() + 15) / 16, (oldMask.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    return changedCountBuffer;
}

bool checkConvergence(const MetalMat& oldMask, const MetalMat& newMask, Stream& stream) {
    initializeUtilityPipelines();
    
    if (!g_convergenceCheckPipeline) {
        CV_Error(Error::StsError, "Failed to create convergence check pipeline");
    }
    
    // Create atomic buffer for changed pixel count
    MetalContext& ctx = MetalContext::getInstance();
    id<MTLDevice> device = ctx.device;
    id<MTLBuffer> changedCountBuffer = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    *((uint32_t*)changedCountBuffer.contents) = 0;
    
    // Use caller's stream to integrate with their pipeline
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:g_convergenceCheckPipeline];
        [encoder setTexture:oldMask.texture() atIndex:0];
        [encoder setTexture:newMask.texture() atIndex:1];
        [encoder setBuffer:changedCountBuffer offset:0 atIndex:0];
        
        MTLSize gridSize = MTLSizeMake((oldMask.cols() + 15) / 16, (oldMask.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // CRITICAL FIX: Sync OUTSIDE autorelease pool to avoid buffer deallocation
    stream.syncCPU();
    
    uint32_t changedPixels = *((uint32_t*)changedCountBuffer.contents);
    return changedPixels == 0; // Converged if no pixels changed
}

}} // cv::metal

#endif // HAVE_METAL 