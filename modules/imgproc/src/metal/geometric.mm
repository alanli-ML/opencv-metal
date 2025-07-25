// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_precomp.hpp"

namespace cv { namespace metal {

//==============================================================================
// Custom Resize Implementation (OpenCV-compatible)
//==============================================================================

// Custom resize Metal shaders that exactly match OpenCV implementation
static const char* resizeShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void resize_nearest(texture2d<float, access::sample> inTexture [[texture(0)]],
                          texture2d<float, access::write> outTexture [[texture(1)]],
                          constant float& scale_x [[buffer(0)]],
                          constant float& scale_y [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
    
    // OpenCV INTER_NEAREST maps destination pixel (x,y) to source coordinate floor(x * fx)
    float src_x = float(gid.x) * scale_x;
    float src_y = float(gid.y) * scale_y;

    // Use floor to select the top-left source texel like OpenCV
    int int_x = int(floor(src_x));
    int int_y = int(floor(src_y));
    
    // Clamp to valid source bounds
    int_x = clamp(int_x, 0, int(inTexture.get_width()) - 1);
    int_y = clamp(int_y, 0, int(inTexture.get_height()) - 1);
    
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    float4 pixel = inTexture.sample(s, float2(float(int_x) + 0.5f, float(int_y) + 0.5f));
    
    outTexture.write(pixel, gid);
}

kernel void resize_linear(texture2d<float, access::sample> inTexture [[texture(0)]],
                         texture2d<float, access::write> outTexture [[texture(1)]],
                         constant float& scale_x [[buffer(0)]],
                         constant float& scale_y [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
    
    // FIXED: OpenCV bilinear coordinate mapping
    float src_x = (float(gid.x) + 0.5f) * scale_x - 0.5f;
    float src_y = (float(gid.y) + 0.5f) * scale_y - 0.5f;
    
    // Get integer and fractional parts
    int x0 = int(floor(src_x));
    int y0 = int(floor(src_y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    float fx = src_x - x0;
    float fy = src_y - y0;
    
    // Clamp coordinates to source bounds
    x0 = clamp(x0, 0, int(inTexture.get_width()) - 1);
    y0 = clamp(y0, 0, int(inTexture.get_height()) - 1);
    x1 = clamp(x1, 0, int(inTexture.get_width()) - 1);
    y1 = clamp(y1, 0, int(inTexture.get_height()) - 1);
    
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    
    // Sample four neighboring pixels
    float4 p00 = inTexture.sample(s, float2(x0 + 0.5f, y0 + 0.5f));
    float4 p01 = inTexture.sample(s, float2(x0 + 0.5f, y1 + 0.5f));
    float4 p10 = inTexture.sample(s, float2(x1 + 0.5f, y0 + 0.5f));
    float4 p11 = inTexture.sample(s, float2(x1 + 0.5f, y1 + 0.5f));
    
    // Bilinear interpolation
    float4 p0 = mix(p00, p10, fx);
    float4 p1 = mix(p01, p11, fx);
    float4 result = mix(p0, p1, fy);
    
    outTexture.write(result, gid);
}
)";

// Pipeline state cache following Connected Components pattern
static id<MTLComputePipelineState> g_resizeNearestPipeline = nil;
static id<MTLComputePipelineState> g_resizeLinearPipeline = nil;
static dispatch_once_t g_resizeOnceToken;

// Create pipeline states
static void createResizePipelines() {
    dispatch_once(&g_resizeOnceToken, ^{
        @autoreleasepool {
            MetalContext& ctx = MetalContext::getInstance();
            id<MTLDevice> device = ctx.device;
            
            NSError* error = nil;
            
            // Compile shader library
            NSString* shaderSource = [NSString stringWithUTF8String:resizeShaderSource];
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            
            if (!library) {
                CV_Error(Error::StsBadFunc, "Failed to compile resize shader");
                return;
            }
            
            // Create pipeline states
            id<MTLFunction> nearestFunction = [library newFunctionWithName:@"resize_nearest"];
            g_resizeNearestPipeline = [device newComputePipelineStateWithFunction:nearestFunction error:&error];
            
            id<MTLFunction> linearFunction = [library newFunctionWithName:@"resize_linear"];
            g_resizeLinearPipeline = [device newComputePipelineStateWithFunction:linearFunction error:&error];
            
            if (!g_resizeNearestPipeline || !g_resizeLinearPipeline) {
                CV_Error(Error::StsBadFunc, "Failed to create resize pipeline states");
            }
        }
    });
}

//==============================================================================
// Resize Implementation
//==============================================================================

void resize(const MetalMat& src, MetalMat& dst, Size dsize, double fx, double fy, int interpolation, Stream& stream)
{
    CV_Assert(!src.empty());
    Size ssize = src.size();

    if (dsize.empty())
    {
        CV_Assert(fx > 0 && fy > 0);
        dsize = Size(saturate_cast<int>(ssize.width * fx), saturate_cast<int>(ssize.height * fy));
    }
    else
    {
        CV_Assert(dsize.width > 0 && dsize.height > 0);
    }
    CV_Assert(dst.empty() || dst.type() == src.type());

    dst.create(dsize, src.type());

    // Use custom implementation for better OpenCV compatibility
    createResizePipelines();
    
    // Calculate scale factors
    float scale_x = (float)ssize.width / (float)dsize.width;
    float scale_y = (float)ssize.height / (float)dsize.height;
    
    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MetalContext& ctx = MetalContext::getInstance();
        
        id<MTLComputePipelineState> pipeline = nil;
        
        // Select appropriate pipeline based on interpolation
        switch (interpolation)
        {
            case INTER_NEAREST:
                pipeline = g_resizeNearestPipeline;
                break;
            case INTER_LINEAR:
                pipeline = g_resizeLinearPipeline;
                break;
            case INTER_LANCZOS4:
                // Fall back to linear for now - Lanczos4 requires more complex implementation
                pipeline = g_resizeLinearPipeline;
                break;
            default:
                CV_Error(Error::StsBadArg, "Unsupported interpolation type for Metal backend");
        }
        CV_Assert(pipeline != nil);

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:src.texture() atIndex:0];
        [encoder setTexture:dst.texture() atIndex:1];
        [encoder setBytes:&scale_x length:sizeof(float) atIndex:0];
        [encoder setBytes:&scale_y length:sizeof(float) atIndex:1];

        // Use Connected Components style threadgroup sizing
        MTLSize threadsPerGrid = MTLSizeMake(dst.cols(), dst.rows(), 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1); // Optimal for Apple GPUs

        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }
}

void resize(const MetalMat& src, MetalMat& dst, Size dsize, double fx, double fy, int interpolation)
{
    Stream stream;
    resize(src, dst, dsize, fx, fy, interpolation, stream);
    stream.commitAndWait();
}

}} // cv::metal 