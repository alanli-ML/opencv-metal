// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_precomp.hpp"
#include <set>

#ifdef HAVE_METAL

using namespace cv;
using namespace cv::metal;

// Block dimensions for connected components algorithm
static constexpr int kBlockRows = 16;
static constexpr int kBlockCols = 16;

namespace {

// MSL source code for Connected Components kernels
static const char* connectedComponentsShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Block dimensions (must match C++ constants)
constant int kblock_rows = 16;
constant int kblock_cols = 16;

// Info bits for inter-block communication (matching CUDA enum)
// Bits 0,1,2,3: pixel a,b,c,d are foreground
// Bits 4,5,6,7: blocks P,Q,R,S need merging
enum Info : unsigned char {
    a = 0, b = 1, c = 2, d = 3,
    P = 4, Q = 5, R = 6, S = 7
};

// Helper functions for bit manipulation
bool has_bit(uint bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}

void set_bit(thread unsigned char& bitmap, unsigned char pos) {
    bitmap |= (1 << pos);
}

// Union-Find helper functions
uint find_root(device int* labels, uint n) {
    while (labels[n] != n) {
        n = labels[n];
    }
    return n;
}

uint find_and_compress(device int* labels, uint n) {
    uint id = n;
    while (labels[n] != n) {
        n = labels[n];
        labels[id] = n;
    }
    return n;
}

// Union operation using atomic operations (FIXED TO MATCH CUDA EXACTLY)
void union_sets(device int* labels, uint a, uint b) {
    bool done;
    
    do {
        a = find_root(labels, a);
        b = find_root(labels, b);
        
        if (a < b) {
            int old = atomic_fetch_min_explicit((device atomic_int*)&labels[b], a, memory_order_relaxed);
            done = (old == b);  // CUDA condition - only check if we won the race
            b = old;
        } else if (b < a) {
            int old = atomic_fetch_min_explicit((device atomic_int*)&labels[a], b, memory_order_relaxed);
            done = (old == a);  // CUDA condition - only check if we won the race  
            a = old;
        } else {
            done = true;  // a == b, already in same set
        }
        
    } while (!done);
}

// Kernel 1: Initialize labeling (BKE algorithm) - use uint texture for 8-bit inputs
kernel void ccl_init_labeling(texture2d<uint, access::read> image [[texture(0)]],
                              device int* labels [[buffer(0)]],
                              device unsigned char* last_pixel [[buffer(1)]],
                              constant uint* image_size [[buffer(2)]],
                              uint2 blockIdx [[threadgroup_position_in_grid]],
                              uint2 threadIdx [[thread_position_in_threadgroup]]) {
    
    uint row = (blockIdx.y * kblock_rows + threadIdx.y) * 2;
    uint col = (blockIdx.x * kblock_cols + threadIdx.x) * 2;
    uint labels_index = row * image_size[0] + col;
    
    if (row >= image_size[1] || col >= image_size[0]) return;
    
    uint P = 0;
    unsigned char info = 0;
    
    // Read 2x2 block of pixels (matching CUDA buffer approach)
    unsigned char pixels[4] = {0, 0, 0, 0};
    
    // Pixel a (top-left)
    if (row < image_size[1] && col < image_size[0]) {
        pixels[0] = (image.read(uint2(col, row)).r > 0) ? 1 : 0;
    }
    // Pixel b (top-right)  
    if (row < image_size[1] && col + 1 < image_size[0]) {
        pixels[1] = (image.read(uint2(col + 1, row)).r > 0) ? 1 : 0;
    }
    // Pixel c (bottom-left)
    if (row + 1 < image_size[1] && col < image_size[0]) {
        pixels[2] = (image.read(uint2(col, row + 1)).r > 0) ? 1 : 0;
    }
    // Pixel d (bottom-right)
    if (row + 1 < image_size[1] && col + 1 < image_size[0]) {
        pixels[3] = (image.read(uint2(col + 1, row + 1)).r > 0) ? 1 : 0;
    }
    
    // Build P-mask exactly like CUDA (CRITICAL FIX)
    if (pixels[0]) {
        P |= 0x777;
        set_bit(info, 0); // Info::a
    }
    if (pixels[1]) {
        P |= (0x777 << 1);
        set_bit(info, 1); // Info::b
    }
    if (pixels[2]) {
        P |= (0x777 << 4);
        set_bit(info, 2); // Info::c
    }
    if (pixels[3]) {
        set_bit(info, 3); // Info::d
        // NOTE: pixel d doesn't contribute to P-mask in CUDA implementation
    }
    
    // Apply boundary masks exactly like CUDA (CRITICAL FIX)
    if (col == 0) {
        P &= 0xEEEE;
    }
    if (col + 1 >= image_size[0]) {
        P &= 0x3333;
    } else if (col + 2 >= image_size[0]) {
        P &= 0x7777;
    }
    
    if (row == 0) {
        P &= 0xFFF0;
    }
    if (row + 1 >= image_size[1]) {
        P &= 0x00FF;
    } else if (row + 2 >= image_size[1]) {
        P &= 0x0FFF;
    }
    
    int father_offset = 0;
    
    // Check connections to neighboring blocks (EXACT CUDA LOGIC)
    
    // P square (top-left) - Test bit 0 of P-mask
    if (has_bit(P, 0) && row > 0 && col > 0) {
        uint neighbor = image.read(uint2(col - 1, row - 1)).r;
        if (neighbor > 0) {
            father_offset = -(2 * image_size[0] + 2);
        }
    }
    
    // Q square (top) - Test bits 1 and 2 of P-mask
    if ((has_bit(P, 1) && row > 0) || (has_bit(P, 2) && row > 0 && col + 1 < image_size[0])) {
        bool has_neighbor = false;
        if (has_bit(P, 1) && row > 0) {
            uint neighbor = image.read(uint2(col, row - 1)).r;
            if (neighbor > 0) has_neighbor = true;
        }
        if (has_bit(P, 2) && row > 0 && col + 1 < image_size[0]) {
            uint neighbor = image.read(uint2(col + 1, row - 1)).r;
            if (neighbor > 0) has_neighbor = true;
        }
        
        if (has_neighbor) {
            if (!father_offset) {
                father_offset = -(2 * image_size[0]);
            } else {
                set_bit(info, 5); // Info::Q
            }
        }
    }
    
    // R square (top-right) - Test bit 3 of P-mask
    if (has_bit(P, 3) && row > 0 && col + 2 < image_size[0]) {
        uint neighbor = image.read(uint2(col + 2, row - 1)).r;
        if (neighbor > 0) {
            if (!father_offset) {
                father_offset = -(2 * image_size[0] - 2);
            } else {
                set_bit(info, 6); // Info::R
            }
        }
    }
    
    // S square (left) - Test bits 4 and 8 of P-mask (CRITICAL FIX)
    if ((has_bit(P, 4) && col > 0) || (has_bit(P, 8) && col > 0 && row + 1 < image_size[1])) {
        bool has_neighbor = false;
        if (has_bit(P, 4) && col > 0) {
            uint neighbor = image.read(uint2(col - 1, row)).r;
            if (neighbor > 0) has_neighbor = true;
        }
        if (has_bit(P, 8) && col > 0 && row + 1 < image_size[1]) {
            uint neighbor = image.read(uint2(col - 1, row + 1)).r;
            if (neighbor > 0) has_neighbor = true;
        }
        
        if (has_neighbor) {
            if (!father_offset) {
                father_offset = -2;
            } else {
                set_bit(info, 7); // Info::S
            }
        }
    }
    
    // Set label exactly like CUDA
    labels[labels_index] = labels_index + father_offset;
    
    // Store info bits in separate metadata buffer (REVERTED)
    if (col + 1 < image_size[0]) {
        last_pixel[labels_index + 1] = info;
    } else if (row + 1 < image_size[1]) {
        last_pixel[labels_index + image_size[0]] = info;
    }
}

// Kernel 2: Merge labels across block boundaries
kernel void ccl_merge(device int* labels [[buffer(0)]],
                      device unsigned char* last_pixel [[buffer(1)]],
                      constant uint* image_size [[buffer(2)]],
                      uint2 blockIdx [[threadgroup_position_in_grid]],
                      uint2 threadIdx [[thread_position_in_threadgroup]]) {
    
    uint row = (blockIdx.y * kblock_rows + threadIdx.y) * 2;
    uint col = (blockIdx.x * kblock_cols + threadIdx.x) * 2;
    uint labels_index = row * image_size[0] + col;
    
    if (row >= image_size[1] || col >= image_size[0]) return;
    
    unsigned char info;
    if (col + 1 < image_size[0]) {
        info = last_pixel[labels_index + 1];
    } else if (row + 1 < image_size[1]) {
        info = last_pixel[labels_index + image_size[0]];
    } else {
        return;
    }
    
    // Merge based on info bits (following CUDA logic)
    if (has_bit(info, 5)) { // Info::Q
        union_sets(labels, labels_index, labels_index - 2 * image_size[0]);
    }
    if (has_bit(info, 6)) { // Info::R
        union_sets(labels, labels_index, labels_index - 2 * image_size[0] + 2);
    }
    if (has_bit(info, 7)) { // Info::S
        union_sets(labels, labels_index, labels_index - 2);
    }
}

// Kernel 3: Path compression
kernel void ccl_compression(device int* labels [[buffer(0)]],
                           constant uint* image_size [[buffer(1)]],
                           uint2 blockIdx [[threadgroup_position_in_grid]],
                           uint2 threadIdx [[thread_position_in_threadgroup]]) {
    
    uint row = (blockIdx.y * kblock_rows + threadIdx.y) * 2;
    uint col = (blockIdx.x * kblock_cols + threadIdx.x) * 2;
    uint labels_index = row * image_size[0] + col;
    
    if (row >= image_size[1] || col >= image_size[0]) return;
    
    find_and_compress(labels, labels_index);
}

// Kernel 4: Final labeling (integer texture input)
kernel void ccl_final_labeling(texture2d<uint, access::read> image [[texture(0)]],
                               device int* labels [[buffer(0)]],
                               texture2d<int, access::write> output [[texture(1)]],
                               constant uint* image_size [[buffer(1)]],
                               device unsigned char* last_pixel [[buffer(2)]],
                               uint2 blockIdx [[threadgroup_position_in_grid]],
                               uint2 threadIdx [[thread_position_in_threadgroup]]) {
    
    uint row = (blockIdx.y * kblock_rows + threadIdx.y) * 2;
    uint col = (blockIdx.x * kblock_cols + threadIdx.x) * 2;
    uint labels_index = row * image_size[0] + col;
    
    if (row >= image_size[1] || col >= image_size[0]) return;
    
    int label;
    unsigned char info;
    
    // Read the compressed label directly (REVERTED FROM find_root)
    label = labels[labels_index] + 1;
    
    // Read stored info from metadata buffer
    if (col + 1 < image_size[0]) {
        info = last_pixel[labels_index + 1];
    } else if (row + 1 < image_size[1]) {
        info = last_pixel[labels_index + image_size[0]];
    } else {
        // Fallback: build info from image directly
        info = 0;
        if (image.read(uint2(col, row)).r > 0) info |= 1; // a bit
        if (col + 1 < image_size[0] && image.read(uint2(col + 1, row)).r > 0) info |= 2; // b bit
        if (row + 1 < image_size[1] && image.read(uint2(col, row + 1)).r > 0) info |= 4; // c bit
        if (col + 1 < image_size[0] && row + 1 < image_size[1] && 
           image.read(uint2(col + 1, row + 1)).r > 0) info |= 8; // d bit
    }
    
    // Apply labels to individual pixels based on stored info bits (FIXED LOGIC)
    // Pixel a (top-left)
    if (row < image_size[1] && col < image_size[0]) {
        output.write(has_bit(info, 0) ? label : 0, uint2(col, row));
    }
    
    // Pixel b (top-right)
    if (row < image_size[1] && col + 1 < image_size[0]) {
        output.write(has_bit(info, 1) ? label : 0, uint2(col + 1, row));
    }
    
    // Pixel c (bottom-left)
    if (row + 1 < image_size[1] && col < image_size[0]) {
        output.write(has_bit(info, 2) ? label : 0, uint2(col, row + 1));
    }
    
    // Pixel d (bottom-right)
    if (row + 1 < image_size[1] && col + 1 < image_size[0]) {
        output.write(has_bit(info, 3) ? label : 0, uint2(col + 1, row + 1));
    }
}
)";

// Pipeline state cache
static id<MTLComputePipelineState> g_initLabelingPipeline = nil;
static id<MTLComputePipelineState> g_mergePipeline = nil;
static id<MTLComputePipelineState> g_compressionPipeline = nil;
static id<MTLComputePipelineState> g_finalLabelingPipeline = nil;
static dispatch_once_t g_pipelinesOnceToken;

// Create all pipeline states
static void createPipelines() {
    dispatch_once(&g_pipelinesOnceToken, ^{
        @autoreleasepool {
            MetalContext& ctx = MetalContext::getInstance();
            id<MTLDevice> device = ctx.device;
            
            NSError* error = nil;
            
            // Compile shader library
            NSString* shaderSource = [NSString stringWithUTF8String:connectedComponentsShaderSource];
            id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
            
            if (!library) {
                CV_Error(Error::StsBadFunc, "Failed to compile connected components shader");
                return;
            }
            
            // Create pipeline states for each kernel
            id<MTLFunction> initFunction = [library newFunctionWithName:@"ccl_init_labeling"];
            g_initLabelingPipeline = [device newComputePipelineStateWithFunction:initFunction error:&error];
            
            id<MTLFunction> mergeFunction = [library newFunctionWithName:@"ccl_merge"];
            g_mergePipeline = [device newComputePipelineStateWithFunction:mergeFunction error:&error];
            
            id<MTLFunction> compressionFunction = [library newFunctionWithName:@"ccl_compression"];
            g_compressionPipeline = [device newComputePipelineStateWithFunction:compressionFunction error:&error];
            
            id<MTLFunction> finalFunction = [library newFunctionWithName:@"ccl_final_labeling"];
            g_finalLabelingPipeline = [device newComputePipelineStateWithFunction:finalFunction error:&error];
            
            if (!g_initLabelingPipeline || !g_mergePipeline || !g_compressionPipeline || !g_finalLabelingPipeline) {
                CV_Error(Error::StsBadFunc, "Failed to create connected components pipeline states");
            }
        }
    });
}

} // anonymous namespace

namespace cv { namespace metal {

int connectedComponents(const MetalMat& image, MetalMat& labels, int connectivity, int ltype, Stream& stream) {
    // Input validation
    CV_Assert(image.channels() == 1);
    CV_Assert(connectivity == 8); // Only 8-connectivity supported for now
    CV_Assert(ltype == CV_32S);
    CV_Assert(image.depth() == CV_8U);
    
    // Create output labels matrix - use CV_32S directly now that it's supported
    labels.create(image.size(), CV_32S);
    
    // Get Metal context and command buffer (CRITICAL: outside autorelease pool)
    MetalContext& ctx = MetalContext::getInstance();
    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    
    // Create pipelines
    createPipelines();
    
    @autoreleasepool {
        // Create Metal buffers for intermediate data
        cv::Size imgSize = image.size();
        size_t labelBufferSize = imgSize.width * imgSize.height * sizeof(int);
        size_t metadataBufferSize = imgSize.width * imgSize.height * sizeof(unsigned char);
        
        id<MTLBuffer> labelsBuffer = [ctx.device newBufferWithLength:labelBufferSize 
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> metadataBuffer = [ctx.device newBufferWithLength:metadataBufferSize 
                                                              options:MTLResourceStorageModeShared];
        
        // Image size constant
        uint32_t imageSizeData[2] = {static_cast<uint32_t>(imgSize.width), static_cast<uint32_t>(imgSize.height)};
        id<MTLBuffer> imageSizeBuffer = [ctx.device newBufferWithBytes:imageSizeData 
                                                               length:sizeof(imageSizeData) 
                                                              options:MTLResourceStorageModeShared];
        
        // Calculate grid dimensions with optimized thread group sizes for Apple Silicon
        uint blocksX = (((imgSize.width + 1) / 2) - 1) / kBlockCols + 1;
        uint blocksY = (((imgSize.height + 1) / 2) - 1) / kBlockRows + 1;
        MTLSize gridSize = MTLSizeMake(blocksX, blocksY, 1);
        MTLSize blockSize = MTLSizeMake(kBlockCols, kBlockRows, 1);
        
        // Optimized thread group size for Apple Silicon (prefer 32 threads per SIMD group)
        MTLSize optimizedBlockSize = MTLSizeMake(8, 4, 1); // 32 threads total
        MTLSize optimizedGridSize = MTLSizeMake((blocksX * kBlockCols + 7) / 8, (blocksY * kBlockRows + 3) / 4, 1);
        
        // Kernel 1: Init Labeling
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_initLabelingPipeline];
        [encoder setTexture:image.texture() atIndex:0];
        [encoder setBuffer:labelsBuffer offset:0 atIndex:0];
        [encoder setBuffer:metadataBuffer offset:0 atIndex:1];
        [encoder setBuffer:imageSizeBuffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
        [encoder endEncoding];
        
        // Kernel 2: First Compression
        encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_compressionPipeline];
        [encoder setBuffer:labelsBuffer offset:0 atIndex:0];
        [encoder setBuffer:imageSizeBuffer offset:0 atIndex:1];
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
        [encoder endEncoding];
        
        // Kernel 3: Merge
        encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_mergePipeline];
        [encoder setBuffer:labelsBuffer offset:0 atIndex:0];
        [encoder setBuffer:metadataBuffer offset:0 atIndex:1];
        [encoder setBuffer:imageSizeBuffer offset:0 atIndex:2];
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
        [encoder endEncoding];
        
        // Kernel 4: Second Compression
        encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:g_compressionPipeline];
        [encoder setBuffer:labelsBuffer offset:0 atIndex:0];
        [encoder setBuffer:imageSizeBuffer offset:0 atIndex:1];
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
        [encoder endEncoding];
        
        // Kernel 5: Final Labeling (REVERTED TO METADATA BUFFER APPROACH)
        id<MTLComputeCommandEncoder> copyEncoder = [commandBuffer computeCommandEncoder];
        [copyEncoder setComputePipelineState:g_finalLabelingPipeline];
        [copyEncoder setTexture:image.texture() atIndex:0];
        [copyEncoder setBuffer:labelsBuffer offset:0 atIndex:0];
        [copyEncoder setTexture:labels.texture() atIndex:1];
        [copyEncoder setBuffer:imageSizeBuffer offset:0 atIndex:1];
        [copyEncoder setBuffer:metadataBuffer offset:0 atIndex:2];
        [copyEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:blockSize];
        [copyEncoder endEncoding];
    }
    
    // Return component count (simplified for now)
    return 1;
}

int connectedComponents(const MetalMat& image, MetalMat& labels, int connectivity, int ltype) {
    Stream stream;
    int result = connectedComponents(image, labels, connectivity, ltype, stream);
    stream.commitAndWait();
    
    // Proper component counting AFTER stream synchronization (CRITICAL FIX)
    int component_count = 1;
    @autoreleasepool {
        // Create temporary Mat to download labels
        Mat temp_labels;
        labels.download(temp_labels);
        
        // Count unique non-zero labels
        std::set<int> unique_labels;
        for (int i = 0; i < temp_labels.rows; ++i) {
            const int* ptr = temp_labels.ptr<int>(i);
            for (int j = 0; j < temp_labels.cols; ++j) {
                int label = ptr[j];
                if (label > 0) {
                    unique_labels.insert(label);
                }
            }
        }
        component_count = unique_labels.size();
    }
    
    return component_count;
}

}} // namespace cv::metal

#endif // HAVE_METAL 