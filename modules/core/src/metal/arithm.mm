// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "metal_precomp.hpp"
#include "metal_wrapper.hpp"

namespace cv { namespace metal {

void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(dst.empty() || (dst.size() == src1.size() && dst.type() == src1.type()));

    dst.create(src1.size(), src1.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MPSImageAdd *adder = [[MPSImageAdd alloc] initWithDevice:MetalContext::getInstance().device];
        CV_Assert(adder != nil);

        [adder encodeToCommandBuffer:commandBuffer
                      primaryTexture:src1.texture()
                    secondaryTexture:src2.texture()
                  destinationTexture:dst.texture()];
    }
}

void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    Stream stream;
    add(src1, src2, dst, stream);
    stream.commitAndWait();
}

void subtract(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(dst.empty() || (dst.size() == src1.size() && dst.type() == src1.type()));

    dst.create(src1.size(), src1.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MPSImageSubtract *subtractor = [[MPSImageSubtract alloc] initWithDevice:MetalContext::getInstance().device];
        CV_Assert(subtractor != nil);

        [subtractor encodeToCommandBuffer:commandBuffer
                           primaryTexture:src1.texture()
                         secondaryTexture:src2.texture()
                       destinationTexture:dst.texture()];
    }
}

void subtract(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    Stream stream;
    subtract(src1, src2, dst, stream);
    stream.commitAndWait();
}

static id<MTLComputePipelineState> getMultiplyPipeline(int type)
{
    static id<MTLComputePipelineState> multiply_8UC4_pipeline = nil;
    static id<MTLComputePipelineState> multiply_32FC1_pipeline = nil;
    static id<MTLComputePipelineState> multiply_32FC4_pipeline = nil;
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        id<MTLDevice> device = MetalContext::getInstance().device;
        
        // Load the Metal library containing our kernels
        NSError *error = nil;
        NSString *libraryPath = [[NSBundle mainBundle] pathForResource:@"arithm_kernels" ofType:@"metallib"];
        id<MTLLibrary> library = nil;
        
        if (libraryPath) {
            NSURL *libraryURL = [NSURL fileURLWithPath:libraryPath];
            library = [device newLibraryWithURL:libraryURL error:&error];
        } else {
            // Fallback: compile from source (for development)
            NSString *kernelSource = @R"(
                #include <metal_stdlib>
                using namespace metal;
                
                kernel void multiply_8UC4_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                                 texture2d<float, access::read> src2 [[texture(1)]],
                                                 texture2d<float, access::write> dst [[texture(2)]],
                                                 uint2 gid [[thread_position_in_grid]])
                {
                    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
                    float4 pixel1 = src1.read(gid);
                    float4 pixel2 = src2.read(gid);
                    uint4 int_pixel1 = uint4(pixel1 * 255.0f + 0.5f);
                    uint4 int_pixel2 = uint4(pixel2 * 255.0f + 0.5f);
                    uint4 int_result;
                    int_result.r = min(255u, int_pixel1.r * int_pixel2.r);
                    int_result.g = min(255u, int_pixel1.g * int_pixel2.g);
                    int_result.b = min(255u, int_pixel1.b * int_pixel2.b);
                    int_result.a = min(255u, int_pixel1.a * int_pixel2.a);
                    float4 result = float4(int_result) / 255.0f;
                    dst.write(result, gid);
                }
                
                kernel void multiply_32FC1_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                                  texture2d<float, access::read> src2 [[texture(1)]],
                                                  texture2d<float, access::write> dst [[texture(2)]],
                                                  uint2 gid [[thread_position_in_grid]])
                {
                    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
                    float4 pixel1 = src1.read(gid);
                    float4 pixel2 = src2.read(gid);
                    float4 result = pixel1 * pixel2;
                    dst.write(result, gid);
                }
                
                kernel void multiply_32FC4_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                                  texture2d<float, access::read> src2 [[texture(1)]],
                                                  texture2d<float, access::write> dst [[texture(2)]],
                                                  uint2 gid [[thread_position_in_grid]])
                {
                    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
                    float4 pixel1 = src1.read(gid);
                    float4 pixel2 = src2.read(gid);
                    float4 result = pixel1 * pixel2;
                    dst.write(result, gid);
                }
            )";
            
            library = [device newLibraryWithSource:kernelSource options:nil error:&error];
        }
        
        if (!library) {
            NSLog(@"Failed to create Metal library: %@", error.localizedDescription);
            return;
        }
        
        // Create compute pipeline states
        id<MTLFunction> multiply_8UC4_func = [library newFunctionWithName:@"multiply_8UC4_opencv"];
        id<MTLFunction> multiply_32FC1_func = [library newFunctionWithName:@"multiply_32FC1_opencv"];
        id<MTLFunction> multiply_32FC4_func = [library newFunctionWithName:@"multiply_32FC4_opencv"];
        
        if (multiply_8UC4_func) {
            multiply_8UC4_pipeline = [device newComputePipelineStateWithFunction:multiply_8UC4_func error:&error];
        }
        if (multiply_32FC1_func) {
            multiply_32FC1_pipeline = [device newComputePipelineStateWithFunction:multiply_32FC1_func error:&error];
        }
        if (multiply_32FC4_func) {
            multiply_32FC4_pipeline = [device newComputePipelineStateWithFunction:multiply_32FC4_func error:&error];
        }
    });
    
    switch (type) {
        case CV_8UC4: return multiply_8UC4_pipeline;
        case CV_32FC1: return multiply_32FC1_pipeline;
        case CV_32FC4: return multiply_32FC4_pipeline;
        default: return nil;
    }
}

void multiply(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(dst.empty() || (dst.size() == src1.size() && dst.type() == src1.type()));

    dst.create(src1.size(), src1.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    // Use custom OpenCV-compatible kernels instead of MPS
    id<MTLComputePipelineState> pipeline = getMultiplyPipeline(src1.type());
    if (!pipeline) {
        CV_Error(Error::StsUnsupportedFormat, "Unsupported type for Metal multiply operation");
        return;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:src1.texture() atIndex:0];
    [encoder setTexture:src2.texture() atIndex:1];
    [encoder setTexture:dst.texture() atIndex:2];
    
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake((src1.cols() + threadgroupSize.width - 1) / threadgroupSize.width,
                                   (src1.rows() + threadgroupSize.height - 1) / threadgroupSize.height,
                                   1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void multiply(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    Stream stream;
    multiply(src1, src2, dst, stream);
    stream.commitAndWait();
}

static id<MTLComputePipelineState> getDividePipeline(int type)
{
    static id<MTLComputePipelineState> divide_8UC4_pipeline = nil;
    static id<MTLComputePipelineState> divide_32FC1_pipeline = nil;
    static id<MTLComputePipelineState> divide_32FC4_pipeline = nil;
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        id<MTLDevice> device = MetalContext::getInstance().device;
        
        // Compile divide kernels from source
        NSError *error = nil;
        NSString *kernelSource = @R"(
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void divide_8UC4_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                           texture2d<float, access::read> src2 [[texture(1)]],
                                           texture2d<float, access::write> dst [[texture(2)]],
                                           uint2 gid [[thread_position_in_grid]])
            {
                if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
                float4 pixel1 = src1.read(gid);
                float4 pixel2 = src2.read(gid);
                uint4 int_pixel1 = uint4(pixel1 * 255.0f + 0.5f);
                uint4 int_pixel2 = uint4(pixel2 * 255.0f + 0.5f);
                uint4 int_result;
                int_result.r = (int_pixel2.r == 0) ? 0 : int_pixel1.r / int_pixel2.r;
                int_result.g = (int_pixel2.g == 0) ? 0 : int_pixel1.g / int_pixel2.g;
                int_result.b = (int_pixel2.b == 0) ? 0 : int_pixel1.b / int_pixel2.b;
                int_result.a = (int_pixel2.a == 0) ? 0 : int_pixel1.a / int_pixel2.a;
                float4 result = float4(int_result) / 255.0f;
                dst.write(result, gid);
            }
            
            kernel void divide_32FC1_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                            texture2d<float, access::read> src2 [[texture(1)]],
                                            texture2d<float, access::write> dst [[texture(2)]],
                                            uint2 gid [[thread_position_in_grid]])
            {
                if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
                float4 pixel1 = src1.read(gid);
                float4 pixel2 = src2.read(gid);
                float4 result;
                result.r = (pixel2.r == 0.0f) ? 0.0f : pixel1.r / pixel2.r;
                result.g = (pixel2.g == 0.0f) ? 0.0f : pixel1.g / pixel2.g;
                result.b = (pixel2.b == 0.0f) ? 0.0f : pixel1.b / pixel2.b;
                result.a = (pixel2.a == 0.0f) ? 0.0f : pixel1.a / pixel2.a;
                dst.write(result, gid);
            }
            
            kernel void divide_32FC4_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                            texture2d<float, access::read> src2 [[texture(1)]],
                                            texture2d<float, access::write> dst [[texture(2)]],
                                            uint2 gid [[thread_position_in_grid]])
            {
                if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
                float4 pixel1 = src1.read(gid);
                float4 pixel2 = src2.read(gid);
                float4 result;
                result.r = (pixel2.r == 0.0f) ? 0.0f : pixel1.r / pixel2.r;
                result.g = (pixel2.g == 0.0f) ? 0.0f : pixel1.g / pixel2.g;
                result.b = (pixel2.b == 0.0f) ? 0.0f : pixel1.b / pixel2.b;
                result.a = (pixel2.a == 0.0f) ? 0.0f : pixel1.a / pixel2.a;
                dst.write(result, gid);
            }
        )";
        
        id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
        if (!library) {
            NSLog(@"Failed to create Metal divide library: %@", error.localizedDescription);
            return;
        }
        
        // Create compute pipeline states
        id<MTLFunction> divide_8UC4_func = [library newFunctionWithName:@"divide_8UC4_opencv"];
        id<MTLFunction> divide_32FC1_func = [library newFunctionWithName:@"divide_32FC1_opencv"];
        id<MTLFunction> divide_32FC4_func = [library newFunctionWithName:@"divide_32FC4_opencv"];
        
        if (divide_8UC4_func) {
            divide_8UC4_pipeline = [device newComputePipelineStateWithFunction:divide_8UC4_func error:&error];
        }
        if (divide_32FC1_func) {
            divide_32FC1_pipeline = [device newComputePipelineStateWithFunction:divide_32FC1_func error:&error];
        }
        if (divide_32FC4_func) {
            divide_32FC4_pipeline = [device newComputePipelineStateWithFunction:divide_32FC4_func error:&error];
        }
    });
    
    switch (type) {
        case CV_8UC4: return divide_8UC4_pipeline;
        case CV_32FC1: return divide_32FC1_pipeline;
        case CV_32FC4: return divide_32FC4_pipeline;
        default: return nil;
    }
}

void divide(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(dst.empty() || (dst.size() == src1.size() && dst.type() == src1.type()));

    dst.create(src1.size(), src1.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    // Use custom OpenCV-compatible kernels instead of MPS
    id<MTLComputePipelineState> pipeline = getDividePipeline(src1.type());
    if (!pipeline) {
        CV_Error(Error::StsUnsupportedFormat, "Unsupported type for Metal divide operation");
        return;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:src1.texture() atIndex:0];
    [encoder setTexture:src2.texture() atIndex:1];
    [encoder setTexture:dst.texture() atIndex:2];
    
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake((src1.cols() + threadgroupSize.width - 1) / threadgroupSize.width,
                                   (src1.rows() + threadgroupSize.height - 1) / threadgroupSize.height,
                                   1);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void divide(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    Stream stream;
    divide(src1, src2, dst, stream);
    stream.commitAndWait();
}

}} // cv::metal