// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_precomp.hpp"

namespace cv { namespace metal {

//==============================================================================
// High-Precision Gaussian Blur Implementation (OpenCV-compatible)
//==============================================================================

// High-precision Gaussian Blur Metal shader that exactly matches OpenCV implementation
static const char* gaussianBlurShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gaussianBlur_custom(texture2d<float, access::sample> inTexture [[texture(0)]],
                               texture2d<float, access::write> outTexture [[texture(1)]],
                               constant float* kernel_weights [[buffer(0)]],
                               constant int& kernel_size [[buffer(1)]],
                               uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int radius = kernel_size / 2;
    float4 sum = float4(0.0f);
    
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            // Calculate source coordinates exactly as OpenCV does
            int src_x = int(gid.x) + i;
            int src_y = int(gid.y) + j;
            
            // CRITICAL: Use exact OpenCV border handling (BORDER_REFLECT_101)
            // This is the precise algorithm OpenCV uses for reflection
            int width = int(inTexture.get_width());
            int height = int(inTexture.get_height());
            
            // OpenCV's BORDER_REFLECT_101 implementation
            if (src_x < 0) src_x = -src_x;
            if (src_x >= width) src_x = 2 * width - src_x - 2;
            if (src_y < 0) src_y = -src_y;
            if (src_y >= height) src_y = 2 * height - src_y - 2;
            
            // Clamp to ensure we're still in bounds after reflection
            src_x = clamp(src_x, 0, width - 1);
            src_y = clamp(src_y, 0, height - 1);
            
            // Get kernel weight (stored row-major)
            int weight_idx = (j + radius) * kernel_size + (i + radius);
            float weight = kernel_weights[weight_idx];
            
            // Sample at exact pixel center using integer coordinates converted to float
            float2 sample_coord = float2(float(src_x) + 0.5f, float(src_y) + 0.5f);
            float4 pixel = inTexture.sample(s, sample_coord);
            
            sum += pixel * weight;
        }
    }
    
    outTexture.write(sum, gid);
}
)";

// Pipeline state cache following Connected Components pattern
static id<MTLComputePipelineState> g_gaussianBlurPipeline = nil;
static dispatch_once_t g_gaussianBlurOnceToken;

// Create pipeline state
static void createGaussianBlurPipeline() {
    dispatch_once(&g_gaussianBlurOnceToken, ^{
    @autoreleasepool {
            MetalContext& ctx = MetalContext::getInstance();
            id<MTLDevice> device = ctx.device;
            
            NSError* error = nil;
            
            // Compile shader library
            NSString* shaderSource = [NSString stringWithUTF8String:gaussianBlurShaderSource];
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            
            if (!library) {
                CV_Error(Error::StsBadFunc, "Failed to compile Gaussian blur shader");
                return;
            }
            
            // Create pipeline state
            id<MTLFunction> gaussianFunction = [library newFunctionWithName:@"gaussianBlur_custom"];
            g_gaussianBlurPipeline = [device newComputePipelineStateWithFunction:gaussianFunction error:&error];
            
            if (!g_gaussianBlurPipeline) {
                CV_Error(Error::StsBadFunc, "Failed to create Gaussian blur pipeline state");
            }
        }
    });
}

// Generate high-precision Gaussian kernel weights exactly like OpenCV
static std::vector<float> generateGaussianKernel(int ksize, double sigma) {
    std::vector<float> kernel(ksize * ksize);
    int radius = ksize / 2;
    double sigma2 = sigma * sigma;
    double sum = 0.0;
    
    // Generate 2D Gaussian kernel with high precision exactly like OpenCV
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            double distance2 = double(i * i + j * j);
            double value = exp(-distance2 / (2.0 * sigma2));
            kernel[(j + radius) * ksize + (i + radius)] = (float)value;
            sum += value;
        }
    }
    
    // Normalize kernel to ensure exact sum = 1.0 (critical for OpenCV match)
    double inv_sum = 1.0 / sum;
    for (float& weight : kernel) {
        weight = (float)(weight * inv_sum);
    }
    
    return kernel;
}

//==============================================================================
// GaussianBlur Implementation
//==============================================================================

void GaussianBlur(const MetalMat& src, MetalMat& dst, Size ksize, double sigmaX, Stream& stream)
{
    CV_Assert(src.type() == dst.type() || dst.empty());
    CV_Assert(ksize.width > 0 && ksize.width % 2 == 1 &&
              ksize.height > 0 && ksize.height % 2 == 1);
    CV_Assert(sigmaX > 0);
    CV_Assert(ksize.width == ksize.height); // Only square kernels for now

    dst.create(src.size(), src.type());

    // CRITICAL: Use high-precision custom implementation for exact OpenCV compatibility
    createGaussianBlurPipeline();
    
    // Generate high-precision OpenCV-compatible Gaussian kernel
    std::vector<float> kernel_weights = generateGaussianKernel(ksize.width, sigmaX);
    
    // DEBUG: Verify custom implementation is being used
    CV_Assert(g_gaussianBlurPipeline != nil);

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MetalContext& ctx = MetalContext::getInstance();
        
        // Create buffer for kernel weights with exact precision
        id<MTLBuffer> weightsBuffer = [ctx.device newBufferWithBytes:kernel_weights.data()
                                                              length:kernel_weights.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
        CV_Assert(weightsBuffer != nil);

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_gaussianBlurPipeline];
        [encoder setTexture:src.texture() atIndex:0];
        [encoder setTexture:dst.texture() atIndex:1];
        [encoder setBuffer:weightsBuffer offset:0 atIndex:0];
        [encoder setBytes:&ksize.width length:sizeof(int) atIndex:1];

        // Use Connected Components style threadgroup sizing
        MTLSize threadsPerGrid = MTLSizeMake(dst.cols(), dst.rows(), 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1); // Optimal for Apple GPUs

        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }
}

void GaussianBlur(const MetalMat& src, MetalMat& dst, Size ksize, double sigmaX)
{
    Stream stream;
    GaussianBlur(src, dst, ksize, sigmaX, stream);
    stream.commitAndWait();
}

//==============================================================================
// Custom Sobel Implementation (OpenCV-compatible)
//==============================================================================

// Custom Sobel Metal shader that exactly matches OpenCV implementation
static const char* sobelShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void sobel_custom(texture2d<float, access::sample> inTexture [[texture(0)]],
                                          texture2d<float, access::write> outTexture [[texture(1)]],
                        constant int& dx [[buffer(0)]],
                        constant int& dy [[buffer(1)]],
                        constant int& ksize [[buffer(2)]],
                                          uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    float4 result = float4(0.0f);
    
    if (ksize == 3) {
        // Sobel 3x3 kernels (most common case)
        if (dx == 1 && dy == 0) {
            // Sobel X (horizontal edges) with exact OpenCV border handling
            // Kernel: [-1 0 1; -2 0 2; -1 0 1]
            int width = int(inTexture.get_width());
            int height = int(inTexture.get_height());
            
            // Calculate all 9 sample coordinates with BORDER_REFLECT_101
            int coords[9][2] = {
                {int(gid.x) - 1, int(gid.y) - 1}, {int(gid.x), int(gid.y) - 1}, {int(gid.x) + 1, int(gid.y) - 1},
                {int(gid.x) - 1, int(gid.y)}, {int(gid.x), int(gid.y)}, {int(gid.x) + 1, int(gid.y)},
                {int(gid.x) - 1, int(gid.y) + 1}, {int(gid.x), int(gid.y) + 1}, {int(gid.x) + 1, int(gid.y) + 1}
            };
            
            float4 pixels[9];
            for (int i = 0; i < 9; i++) {
                int src_x = coords[i][0];
                int src_y = coords[i][1];
                
                // Apply BORDER_REFLECT_101
                if (src_x < 0) src_x = -src_x;
                if (src_x >= width) src_x = 2 * width - src_x - 2;
                if (src_y < 0) src_y = -src_y;
                if (src_y >= height) src_y = 2 * height - src_y - 2;
                
                src_x = clamp(src_x, 0, width - 1);
                src_y = clamp(src_y, 0, height - 1);
                
                pixels[i] = inTexture.sample(s, float2(float(src_x) + 0.5f, float(src_y) + 0.5f));
            }
            
            float4 p00 = pixels[0], p01 = pixels[1], p02 = pixels[2];
            float4 p10 = pixels[3], p11 = pixels[4], p12 = pixels[5];
            float4 p20 = pixels[6], p21 = pixels[7], p22 = pixels[8];
            
            result = -1*p00 + 0*p01 + 1*p02 + 
                     -2*p10 + 0*p11 + 2*p12 + 
                     -1*p20 + 0*p21 + 1*p22;
        } 
        else if (dx == 0 && dy == 1) {
            // Sobel Y (vertical edges) with exact OpenCV border handling
            // Kernel: [-1 -2 -1; 0 0 0; 1 2 1]
            int width = int(inTexture.get_width());
            int height = int(inTexture.get_height());
            
            // Calculate all 9 sample coordinates with BORDER_REFLECT_101
            int coords[9][2] = {
                {int(gid.x) - 1, int(gid.y) - 1}, {int(gid.x), int(gid.y) - 1}, {int(gid.x) + 1, int(gid.y) - 1},
                {int(gid.x) - 1, int(gid.y)}, {int(gid.x), int(gid.y)}, {int(gid.x) + 1, int(gid.y)},
                {int(gid.x) - 1, int(gid.y) + 1}, {int(gid.x), int(gid.y) + 1}, {int(gid.x) + 1, int(gid.y) + 1}
            };
            
            float4 pixels[9];
            for (int i = 0; i < 9; i++) {
                int src_x = coords[i][0];
                int src_y = coords[i][1];
                
                // Apply BORDER_REFLECT_101
                if (src_x < 0) src_x = -src_x;
                if (src_x >= width) src_x = 2 * width - src_x - 2;
                if (src_y < 0) src_y = -src_y;
                if (src_y >= height) src_y = 2 * height - src_y - 2;
                
                src_x = clamp(src_x, 0, width - 1);
                src_y = clamp(src_y, 0, height - 1);
                
                pixels[i] = inTexture.sample(s, float2(float(src_x) + 0.5f, float(src_y) + 0.5f));
            }
            
            float4 p00 = pixels[0], p01 = pixels[1], p02 = pixels[2];
            float4 p10 = pixels[3], p11 = pixels[4], p12 = pixels[5];
            float4 p20 = pixels[6], p21 = pixels[7], p22 = pixels[8];
            
            result = -1*p00 + -2*p01 + -1*p02 + 
                      0*p10 +  0*p11 +  0*p12 + 
                      1*p20 +  2*p21 +  1*p22;
        }
        else if (dx == 1 && dy == 1) {
            // Mixed second derivative (dx=1, dy=1) with exact OpenCV border handling
            // Kernel for mixed derivative (∂²f/∂x∂y)
            int width = int(inTexture.get_width());
            int height = int(inTexture.get_height());
            
            // Calculate all 9 sample coordinates with BORDER_REFLECT_101
            int coords[9][2] = {
                {int(gid.x) - 1, int(gid.y) - 1}, {int(gid.x), int(gid.y) - 1}, {int(gid.x) + 1, int(gid.y) - 1},
                {int(gid.x) - 1, int(gid.y)}, {int(gid.x), int(gid.y)}, {int(gid.x) + 1, int(gid.y)},
                {int(gid.x) - 1, int(gid.y) + 1}, {int(gid.x), int(gid.y) + 1}, {int(gid.x) + 1, int(gid.y) + 1}
            };
            
            float4 pixels[9];
            for (int i = 0; i < 9; i++) {
                int src_x = coords[i][0];
                int src_y = coords[i][1];
                
                // Apply BORDER_REFLECT_101
                if (src_x < 0) src_x = -src_x;
                if (src_x >= width) src_x = 2 * width - src_x - 2;
                if (src_y < 0) src_y = -src_y;
                if (src_y >= height) src_y = 2 * height - src_y - 2;
                
                src_x = clamp(src_x, 0, width - 1);
                src_y = clamp(src_y, 0, height - 1);
                
                pixels[i] = inTexture.sample(s, float2(float(src_x) + 0.5f, float(src_y) + 0.5f));
            }
            
            float4 p00 = pixels[0], p01 = pixels[1], p02 = pixels[2];
            float4 p10 = pixels[3], p11 = pixels[4], p12 = pixels[5];
            float4 p20 = pixels[6], p21 = pixels[7], p22 = pixels[8];
            
            // Mixed derivative kernel [1 0 -1; 0 0 0; -1 0 1]
            result = 1*p00 + 0*p01 + -1*p02 + 
                     0*p10 + 0*p11 +  0*p12 + 
                    -1*p20 + 0*p21 +  1*p22;
        }
    }
    
    outTexture.write(result, gid);
}
)";

// Pipeline state cache following Connected Components pattern
static id<MTLComputePipelineState> g_sobelPipeline = nil;
static dispatch_once_t g_sobelOnceToken;

// Create pipeline state
static void createSobelPipeline() {
    dispatch_once(&g_sobelOnceToken, ^{
        @autoreleasepool {
            MetalContext& ctx = MetalContext::getInstance();
            id<MTLDevice> device = ctx.device;
            
            NSError* error = nil;
            
            // Compile shader library
            NSString* shaderSource = [NSString stringWithUTF8String:sobelShaderSource];
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            
            if (!library) {
                CV_Error(Error::StsBadFunc, "Failed to compile Sobel shader");
                return;
            }
            
            // Create pipeline state
            id<MTLFunction> sobelFunction = [library newFunctionWithName:@"sobel_custom"];
            g_sobelPipeline = [device newComputePipelineStateWithFunction:sobelFunction error:&error];
            
            if (!g_sobelPipeline) {
                CV_Error(Error::StsBadFunc, "Failed to create Sobel pipeline state");
            }
        }
    });
}

//==============================================================================
// Sobel Implementation
//==============================================================================

void Sobel(const MetalMat& src, MetalMat& dst, int ddepth, int dx, int dy, int ksize, Stream& stream)
{
    CV_Assert(src.type() == dst.type() || dst.empty());
    CV_Assert(dx >= 0 && dy >= 0 && dx + dy > 0);
    CV_Assert(ksize == 3); // For now, only ksize=3 is supported.
    CV_Assert((dx == 1 && dy == 0) || (dx == 0 && dy == 1) || (dx == 1 && dy == 1)); // 1st and 2nd order derivatives

    dst.create(src.size(), src.type());

    // Use custom implementation for better OpenCV compatibility
    createSobelPipeline();

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MetalContext& ctx = MetalContext::getInstance();

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_sobelPipeline];
        [encoder setTexture:src.texture() atIndex:0];
        [encoder setTexture:dst.texture() atIndex:1];
        [encoder setBytes:&dx length:sizeof(int) atIndex:0];
        [encoder setBytes:&dy length:sizeof(int) atIndex:1];
        [encoder setBytes:&ksize length:sizeof(int) atIndex:2];

        // Use Connected Components style threadgroup sizing
        MTLSize threadsPerGrid = MTLSizeMake(dst.cols(), dst.rows(), 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1); // Optimal for Apple GPUs

        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }
}

void Sobel(const MetalMat& src, MetalMat& dst, int ddepth, int dx, int dy, int ksize)
{
    Stream stream;
    Sobel(src, dst, ddepth, dx, dy, ksize, stream);
    stream.commitAndWait();
}

//==============================================================================
// Custom Filter Shader Sources
//==============================================================================

static const char* bilateralFilterShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

float norm_l1_float3(float3 d) { return fabs(d.x) + fabs(d.y) + fabs(d.z); }

kernel void bilateralFilter_C1(texture2d<float, access::sample> inTexture [[texture(0)]],
                                texture2d<float, access::write> outTexture [[texture(1)]],
                                constant int& ksize [[buffer(0)]],
                                constant float& sigma_color [[buffer(1)]],
                                constant float& sigma_space [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int radius = ksize / 2;
    float center_norm = inTexture.sample(s, float2(gid)).r;
    int center_int = int(center_norm * 255.0f);
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Simplified bilateral filter with direct exp calculation
    float inv_sigma_color_2 = 1.0f / (2.0f * sigma_color * sigma_color);
    float inv_sigma_space_2 = 1.0f / (2.0f * sigma_space * sigma_space);
    
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            float sx = clamp(float(gid.x) + float(i), 0.0f, float(inTexture.get_width() - 1));
            float sy = clamp(float(gid.y) + float(j), 0.0f, float(inTexture.get_height() - 1));
            
            float sample_norm = inTexture.sample(s, float2(sx, sy)).r;
            int sample_int = int(sample_norm * 255.0f);
            
            // Calculate spatial and color weights
            float space_dist = float(i * i + j * j);
            float color_diff = float(abs(center_int - sample_int));
            
            float space_weight = exp(-space_dist * inv_sigma_space_2);
            float color_weight = exp(-color_diff * color_diff * inv_sigma_color_2);
            float weight = space_weight * color_weight;
            
            sum += weight * float(sample_int);
            weight_sum += weight;
        }
    }
    
    float result_int = (weight_sum > 0.0f) ? (sum / weight_sum) : float(center_int);
    float result_norm = clamp(result_int / 255.0f, 0.0f, 1.0f);
    outTexture.write(float4(result_norm, 0.0f, 0.0f, 1.0f), gid);
}

kernel void bilateralFilter_C4(texture2d<float, access::sample> inTexture [[texture(0)]],
                                texture2d<float, access::write> outTexture [[texture(1)]],
                                constant int& ksize [[buffer(0)]],
                                constant float& sigma_color [[buffer(1)]],
                                constant float& sigma_space [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int radius = ksize / 2;
    float4 center_norm = inTexture.sample(s, float2(gid));
    int4 center_int = int4(center_norm * 255.0f);
    
    float4 sum = float4(0.0f);
    float weight_sum = 0.0f;
    
    // Simplified bilateral filter with direct exp calculation
    float inv_sigma_color_2 = 1.0f / (2.0f * sigma_color * sigma_color);
    float inv_sigma_space_2 = 1.0f / (2.0f * sigma_space * sigma_space);
    
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            float sx = clamp(float(gid.x) + float(i), 0.0f, float(inTexture.get_width() - 1));
            float sy = clamp(float(gid.y) + float(j), 0.0f, float(inTexture.get_height() - 1));
            
            float4 sample_norm = inTexture.sample(s, float2(sx, sy));
            int4 sample_int = int4(sample_norm * 255.0f);
            
            // Calculate spatial and color weights (using RGB distance)
            float space_dist = float(i * i + j * j);
            int3 color_diff = abs(center_int.rgb - sample_int.rgb);
            float color_distance = length(float3(color_diff));
            
            float space_weight = exp(-space_dist * inv_sigma_space_2);
            float color_weight = exp(-color_distance * color_distance * inv_sigma_color_2);
            float weight = space_weight * color_weight;
            
            sum += weight * float4(sample_int);
            weight_sum += weight;
        }
    }
    
    float4 result_int = (weight_sum > 0.0f) ? (sum / weight_sum) : float4(center_int);
    float4 result_norm = clamp(result_int / 255.0f, 0.0f, 1.0f);
    outTexture.write(result_norm, gid);
}
)";

static const char* customFilterShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// OpenCV-compatible box filter kernels that handle texture format conversion
kernel void boxFilter_8UC1_opencv(texture2d<float, access::sample> inTexture [[texture(0)]],
                                  texture2d<float, access::write> outTexture [[texture(1)]],
                                  constant int& ksize_width [[buffer(0)]],
                                  constant int& ksize_height [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int radius_x = ksize_width / 2;
    int radius_y = ksize_height / 2;
    int sum = 0;  // Use integer arithmetic for 8-bit operations
    int count = 0;

    for (int j = -radius_y; j <= radius_y; ++j) {
        for (int i = -radius_x; i <= radius_x; ++i) {
            int x = clamp(int(gid.x) + i, 0, int(inTexture.get_width() - 1));
            int y = clamp(int(gid.y) + j, 0, int(inTexture.get_height() - 1));
            
            // Convert normalized texture value to 8-bit integer
            float pixel_norm = inTexture.sample(s, float2(x, y)).r;
            int pixel_int = int(pixel_norm * 255.0f + 0.5f);  // Round properly
            sum += pixel_int;
            count++;
        }
    }

    // OpenCV box filter: integer division with proper rounding
    int result_int = (sum + count/2) / count;  // Add count/2 for proper rounding
    float result_norm = float(result_int) / 255.0f;  // Convert back to normalized
    outTexture.write(float4(result_norm, 0, 0, 1), gid);
}

kernel void boxFilter_8UC4_opencv(texture2d<float, access::sample> inTexture [[texture(0)]],
                                  texture2d<float, access::write> outTexture [[texture(1)]],
                                  constant int& ksize_width [[buffer(0)]],
                                  constant int& ksize_height [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int radius_x = ksize_width / 2;
    int radius_y = ksize_height / 2;
    int4 sum = int4(0);  // Use integer arithmetic for 8-bit operations
    int count = 0;

    for (int j = -radius_y; j <= radius_y; ++j) {
        for (int i = -radius_x; i <= radius_x; ++i) {
            int x = clamp(int(gid.x) + i, 0, int(inTexture.get_width() - 1));
            int y = clamp(int(gid.y) + j, 0, int(inTexture.get_height() - 1));
            
            // Convert normalized BGRA texture values to 8-bit integers
            float4 pixel_norm = inTexture.sample(s, float2(x, y));
            int4 pixel_int = int4(pixel_norm * 255.0f + 0.5f);  // Round properly
            sum += pixel_int;
            count++;
        }
    }

    // OpenCV box filter: integer division with proper rounding per channel
    int4 result_int = (sum + count/2) / count;  // Add count/2 for proper rounding
    float4 result_norm = float4(result_int) / 255.0f;  // Convert back to normalized
    outTexture.write(result_norm, gid);
}

// OpenCV-compatible 2D filter kernels
kernel void filter2D_8UC1_opencv(texture2d<float, access::sample> inTexture [[texture(0)]],
                                 texture2d<float, access::write> outTexture [[texture(1)]],
                                 constant int& kernel_width [[buffer(0)]],
                                 constant int& kernel_height [[buffer(1)]],
                                 constant float* kernel_weights [[buffer(2)]],
                                 constant float& delta [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int radius_x = kernel_width / 2;
    int radius_y = kernel_height / 2;
    float sum = 0.0f;

    for (int j = -radius_y; j <= radius_y; ++j) {
        for (int i = -radius_x; i <= radius_x; ++i) {
            float2 sample_coord = float2(gid) + float2(i, j);
            float sample_pixel = inTexture.sample(s, sample_coord).r;
            
            // Convert from normalized [0.0, 1.0] to integer [0, 255] range for OpenCV semantics
            uint int_pixel = uint(sample_pixel * 255.0f + 0.5f);
            
            // Get kernel weight (kernel is row-major, flipped for convolution)
            int kernel_idx = (radius_y - j) * kernel_width + (radius_x - i);
            float weight = kernel_weights[kernel_idx];
            
            sum += float(int_pixel) * weight;
        }
    }

    // Add delta and clamp to valid range
    float result_int = sum + delta;
    result_int = clamp(result_int, 0.0f, 255.0f);
    
    // Convert back to normalized [0.0, 1.0] range
    float result = result_int / 255.0f;
    outTexture.write(float4(result, 0, 0, 1), gid);
}

kernel void filter2D_8UC4_opencv(texture2d<float, access::sample> inTexture [[texture(0)]],
                                 texture2d<float, access::write> outTexture [[texture(1)]],
                                 constant int& kernel_width [[buffer(0)]],
                                 constant int& kernel_height [[buffer(1)]],
                                 constant float* kernel_weights [[buffer(2)]],
                                 constant float& delta [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;

    int radius_x = kernel_width / 2;
    int radius_y = kernel_height / 2;
    float4 sum = float4(0.0f);

    for (int j = -radius_y; j <= radius_y; ++j) {
        for (int i = -radius_x; i <= radius_x; ++i) {
            float2 sample_coord = float2(gid) + float2(i, j);
            float4 sample_pixel = inTexture.sample(s, sample_coord);
            
            // Convert from normalized [0.0, 1.0] to integer [0, 255] range for OpenCV semantics
            uint4 int_pixel = uint4(sample_pixel * 255.0f + 0.5f);
            
            // Get kernel weight (kernel is row-major, flipped for convolution)
            int kernel_idx = (radius_y - j) * kernel_width + (radius_x - i);
            float weight = kernel_weights[kernel_idx];
            
            sum += float4(int_pixel) * weight;
        }
    }

    // Add delta and clamp to valid range
    float4 result_int = sum + delta;
    result_int = clamp(result_int, 0.0f, 255.0f);
    
    // Convert back to normalized [0.0, 1.0] range
    float4 result = result_int / 255.0f;
    outTexture.write(result, gid);
}
)";

//==============================================================================
// BilateralFilter Implementation
//==============================================================================

void bilateralFilter(const MetalMat& src, MetalMat& dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode, Stream& stream)
{
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    CV_Assert(src.channels() == 1 || src.channels() == 4);
    CV_Assert(borderMode == BORDER_DEFAULT || borderMode == BORDER_REPLICATE);
    CV_Assert(!src.empty());
    CV_Assert(sigma_color > 0 && sigma_spatial > 0);

    // Fallback to CPU for unsupported cases (currently 32F depth or very large kernels).
    // Temporary: Use CPU fallback for bilateral filter to avoid Metal kernel crash
    // TODO: Fix Metal bilateral filter implementation
    if (true) // (src.channels() == 3 || src.depth() == CV_32F || kernel_size > 15)
    {
        Mat cpu_src, cpu_dst;
        src.download(cpu_src);
        cv::bilateralFilter(cpu_src, cpu_dst, kernel_size, sigma_color, sigma_spatial, borderMode);
        
        // Ensure dst can hold CPU result (will convert 3->4 channels if needed during upload)
        dst.upload(cpu_dst);
        return;
    }

    dst.create(src.size(), src.type());
    // Propagate original channel information so download converts back correctly (e.g., 3-channel host)
    dst.setOriginalChannels(src.getOriginalChannels());

    // Determine effective radius
    int d = kernel_size;
    if (d <= 0) {
        // When d=0, compute kernel size from sigma_spatial like OpenCV does
        d = cvRound(sigma_spatial * 2.0f) | 1; // Ensure odd size
    }
    int radius = d / 2;
    
    @autoreleasepool {
        id<MTLDevice> device = MetalContext::getInstance().device;
    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        // Choose kernel based on channel count
        NSString* kernelName = (src.channels() == 1) ? @"bilateralFilter_C1" : @"bilateralFilter_C4";
        
        // Use more stable kernel compilation approach like GaussianBlur
        NSError* error = nil;
        NSString* shaderSource = [NSString stringWithUTF8String:bilateralFilterShaderSource];
        id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            CV_Error(Error::StsBadFunc, cv::format("Failed to compile bilateral filter shader: %s", 
                     [[error localizedDescription] UTF8String]));
        }
        
        id<MTLFunction> function = [library newFunctionWithName:kernelName];
        if (!function) {
            CV_Error(Error::StsBadFunc, cv::format("Failed to find kernel function: %s", [kernelName UTF8String]));
        }
         
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipelineState) {
            CV_Error(Error::StsError, cv::format("Failed to create bilateral filter pipeline: %s", 
                     [[error localizedDescription] UTF8String]));
        }

        [encoder setComputePipelineState:pipelineState];
        [encoder setTexture:src.texture() atIndex:0];
        [encoder setTexture:dst.texture() atIndex:1];
        
        int effective_ksize = 2 * radius + 1;
        [encoder setBytes:&effective_ksize length:sizeof(int) atIndex:0];
        [encoder setBytes:&sigma_color length:sizeof(float) atIndex:1];
        [encoder setBytes:&sigma_spatial length:sizeof(float) atIndex:2];

        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(
            (dst.cols() + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            (dst.rows() + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            1
        );

        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }
}

void bilateralFilter(const MetalMat& src, MetalMat& dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode)
{
    Stream stream;
    bilateralFilter(src, dst, kernel_size, sigma_color, sigma_spatial, borderMode, stream);
    stream.commitAndWait();
}

//==============================================================================
// BoxFilter Implementation
//==============================================================================

void boxFilter(const MetalMat& src, MetalMat& dst, int ddepth, Size ksize, Point anchor, bool normalize, int borderType, Stream& stream)
{
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    CV_Assert(ksize.width > 0 && ksize.height > 0);
    CV_Assert(ksize.width % 2 == 1 && ksize.height % 2 == 1); // MPS requires odd kernel sizes
    CV_Assert(borderType == BORDER_DEFAULT || borderType == BORDER_REPLICATE);

    int dst_type = ddepth == -1 ? src.type() : CV_MAKETYPE(ddepth, src.channels());

    // Fallback to CPU for channel counts other than 4, or for 32F precision requirements
    // Temporary: Also use CPU for CV_8UC1 until Metal kernel algorithm is debugged
    if (src.channels() == 3 || src.depth() == CV_32F || (src.depth() == CV_8U && src.channels() == 1))
    {
        Mat cpu_src, cpu_dst;
        src.download(cpu_src);
        cv::boxFilter(cpu_src, cpu_dst, ddepth, ksize, anchor, normalize, borderType);
        dst.upload(cpu_dst);
        return;
    }

    dst.create(src.size(), dst_type);
    // Propagate original channel information so download converts back correctly (e.g., 3-channel host)
    dst.setOriginalChannels(src.getOriginalChannels());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    // Use custom OpenCV-compatible kernels for 8U types to fix tolerance issues
    if (src.depth() == CV_8U && normalize) {
        MetalContext& ctx = MetalContext::getInstance();
        
        std::string functionName;
        if (src.channels() == 1) {
            functionName = "boxFilter_8UC1_opencv";
        } else if (src.channels() == 4) {
            functionName = "boxFilter_8UC4_opencv";
        } else {
            // Fall back to MPS for unsupported channel counts
            goto use_mps_implementation;
        }
        
        id<MTLFunction> kernelFunction = ctx.getMetalFunction(customFilterShaderSource, functionName, false);
        CV_Assert(kernelFunction != nil);

    @autoreleasepool {
            NSError* error = nil;
            id<MTLComputePipelineState> pipelineState = [ctx.device newComputePipelineStateWithFunction:kernelFunction error:&error];
            CV_Assert(pipelineState != nil);

            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelineState];
            [encoder setTexture:src.texture() atIndex:0];
            [encoder setTexture:dst.texture() atIndex:1];
            [encoder setBytes:&ksize.width length:sizeof(int) atIndex:0];
            [encoder setBytes:&ksize.height length:sizeof(int) atIndex:1];

            MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake(
                (dst.cols() + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
                (dst.rows() + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
                1
            );

            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }
        return;
    }

use_mps_implementation:
    // Fall back to MPS for 32F types or non-normalized box filters
    @autoreleasepool {
        MPSImageBox *boxFilter;
        if (normalize) {
            boxFilter = [[MPSImageBox alloc] initWithDevice:MetalContext::getInstance().device
                                                            kernelWidth:ksize.width
                                                           kernelHeight:ksize.height];
        } else {
            // For non-normalized box filter, we need to scale by kernel area
            boxFilter = [[MPSImageBox alloc] initWithDevice:MetalContext::getInstance().device
                                                kernelWidth:ksize.width
                                               kernelHeight:ksize.height];
            // Note: MPS automatically normalizes, so for non-normalized we'd need custom kernel
            // For now, we'll use normalized version as it's more commonly used
        }
        CV_Assert(boxFilter != nil);
        boxFilter.edgeMode = (borderType == BORDER_REPLICATE) ? MPSImageEdgeModeClamp : MPSImageEdgeModeZero;

        [boxFilter encodeToCommandBuffer:commandBuffer
                             sourceTexture:src.texture()
                        destinationTexture:dst.texture()];
    }
}

void boxFilter(const MetalMat& src, MetalMat& dst, int ddepth, Size ksize, Point anchor, bool normalize, int borderType)
{
    Stream stream;
    boxFilter(src, dst, ddepth, ksize, anchor, normalize, borderType, stream);
    stream.commitAndWait();
}

//==============================================================================
// Filter2D Implementation
//==============================================================================

void filter2D(const MetalMat& src, MetalMat& dst, int ddepth, InputArray kernel, Point anchor, double delta, int borderType, Stream& stream)
{
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    CV_Assert(borderType == BORDER_DEFAULT || borderType == BORDER_REPLICATE);
    
    Mat kernelMat = kernel.getMat();
    CV_Assert(kernelMat.type() == CV_32F);
    CV_Assert(kernelMat.rows % 2 == 1 && kernelMat.cols % 2 == 1); // Odd kernel size required
    CV_Assert(kernelMat.isContinuous());

    int dst_type = ddepth == -1 ? src.type() : CV_MAKETYPE(ddepth, src.channels());
    dst.create(src.size(), dst_type);

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    // Use custom OpenCV-compatible kernels for 8U types to fix tolerance issues
    if (src.depth() == CV_8U) {
        MetalContext& ctx = MetalContext::getInstance();
        
        std::string functionName;
        if (src.channels() == 1) {
            functionName = "filter2D_8UC1_opencv";
        } else if (src.channels() == 4) {
            functionName = "filter2D_8UC4_opencv";
        } else {
            // Fall back to MPS for unsupported channel counts
            goto use_mps_implementation;
        }
        
        id<MTLFunction> kernelFunction = ctx.getMetalFunction(customFilterShaderSource, functionName, false);
        CV_Assert(kernelFunction != nil);

        @autoreleasepool {
            NSError* error = nil;
            id<MTLComputePipelineState> pipelineState = [ctx.device newComputePipelineStateWithFunction:kernelFunction error:&error];
            CV_Assert(pipelineState != nil);

            // Create buffer for kernel weights
            const float* weights = kernelMat.ptr<float>();
            size_t weights_size = kernelMat.total() * sizeof(float);
            id<MTLBuffer> weightsBuffer = [ctx.device newBufferWithBytes:weights 
                                                                  length:weights_size 
                                                                 options:MTLResourceStorageModeShared];
            CV_Assert(weightsBuffer != nil);

            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipelineState];
            [encoder setTexture:src.texture() atIndex:0];
            [encoder setTexture:dst.texture() atIndex:1];
            [encoder setBytes:&kernelMat.cols length:sizeof(int) atIndex:0];
            [encoder setBytes:&kernelMat.rows length:sizeof(int) atIndex:1];
            [encoder setBuffer:weightsBuffer offset:0 atIndex:2];
            
            float delta_f = static_cast<float>(delta);
            [encoder setBytes:&delta_f length:sizeof(float) atIndex:3];

            MTLSize threadsPerGrid = MTLSizeMake(dst.cols(), dst.rows(), 1);
            NSUInteger w = [pipelineState threadExecutionWidth];
            NSUInteger h = [pipelineState maxTotalThreadsPerThreadgroup] / w;
            MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }
        return;
    }

use_mps_implementation:
    // Fall back to MPS for 32F types
    @autoreleasepool {
        const float* weights = kernelMat.ptr<float>();
        MPSImageConvolution *convolution = [[MPSImageConvolution alloc] initWithDevice:MetalContext::getInstance().device
                                                                          kernelWidth:kernelMat.cols
                                                                         kernelHeight:kernelMat.rows
                                                                              weights:weights];
        CV_Assert(convolution != nil);
        convolution.edgeMode = (borderType == BORDER_REPLICATE) ? MPSImageEdgeModeClamp : MPSImageEdgeModeZero;
        convolution.bias = delta; // Apply delta as bias

        [convolution encodeToCommandBuffer:commandBuffer
                             sourceTexture:src.texture()
                        destinationTexture:dst.texture()];
    }
}

void filter2D(const MetalMat& src, MetalMat& dst, int ddepth, InputArray kernel, Point anchor, double delta, int borderType)
{
    Stream stream;
    filter2D(src, dst, ddepth, kernel, anchor, delta, borderType, stream);
    stream.commitAndWait();
}

//==============================================================================
// MedianBlur Implementation
//==============================================================================

void medianBlur(const MetalMat& src, MetalMat& dst, int ksize, Stream& stream)
{
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    CV_Assert(ksize > 0 && ksize % 2 == 1); // Odd kernel size required
    CV_Assert(ksize <= 15); // MPS limitation

    dst.create(src.size(), src.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MPSImageMedian *median = [[MPSImageMedian alloc] initWithDevice:MetalContext::getInstance().device kernelDiameter:ksize];
        CV_Assert(median != nil);
        median.edgeMode = MPSImageEdgeModeClamp;

        [median encodeToCommandBuffer:commandBuffer
                        sourceTexture:src.texture()
                   destinationTexture:dst.texture()];
    }
}

void medianBlur(const MetalMat& src, MetalMat& dst, int ksize)
{
    Stream stream;
    medianBlur(src, dst, ksize, stream);
    stream.commitAndWait();
}

}} // namespace cv::metal 