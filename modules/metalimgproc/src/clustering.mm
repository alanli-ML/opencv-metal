// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <random>
#include <iostream>

#ifdef HAVE_METAL

// MPS includes for GEMM optimization
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

using namespace cv;
using namespace cv::metal;

namespace {

// Centroid accumulator structure for host code
struct CentroidAccumulator {
    std::atomic<int> r;
    std::atomic<int> g;
    std::atomic<int> b;
    std::atomic<int> count;
};

// Metal Shading Language kernels for K-means clustering
static const char* kmeansShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// Centroid accumulator structure for atomic operations
struct CentroidAccumulator {
    atomic_int r;
    atomic_int g;
    atomic_int b;
    atomic_int count;
};

// Helper function to compute squared distance between two 3D points
inline float compute_distance_squared(float3 a, float3 b) {
    float3 diff = a - b;
    return dot(diff, diff);
}

// Main K-means assignment and accumulation kernel
kernel void assignAndAccumulate(
    texture2d<float, access::read>  inTexture [[texture(0)]],
    texture2d<int, access::write>   outLabels [[texture(1)]],
    device const float* centroids [[buffer(0)]],
    device CentroidAccumulator* accumulators [[buffer(1)]],
    constant int& K [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    // 1. Read the pixel color (our data point) - BGRA texture format
    float4 pixelColor = inTexture.read(gid);
    float3 point = float3(pixelColor.b, pixelColor.g, pixelColor.r);  // BGR order to match buffer layout
    
    // 2. Assignment Step: Find the closest centroid
    float minDistance = FLT_MAX;
    int bestClusterIndex = 0;
    
    for (int i = 0; i < K; ++i) {
        // Read centroid from flat buffer: [B0,G0,R0,B1,G1,R1,...] (BGR order)
        float3 centroid = float3(centroids[i * 3 + 0], centroids[i * 3 + 1], centroids[i * 3 + 2]);
        
        // Explicit distance calculation
        float3 diff = point - centroid;
        float dist = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        
        if (dist < minDistance) {
            minDistance = dist;
            bestClusterIndex = i;
        }
    }
    
    // 3. Write the assigned cluster index to the output label texture
    outLabels.write(bestClusterIndex, gid);
    
    // 4. Update Step: Atomically update the accumulator for the assigned cluster
    // Scale float colors to integer range for atomic operations (BGR order)
    int b = int(point.x * 255.0 + 0.5);  // B channel (point.x)
    int g = int(point.y * 255.0 + 0.5);  // G channel (point.y)
    int r = int(point.z * 255.0 + 0.5);  // R channel (point.z)
    
    atomic_fetch_add_explicit(&accumulators[bestClusterIndex].b, b, memory_order_relaxed);
    atomic_fetch_add_explicit(&accumulators[bestClusterIndex].g, g, memory_order_relaxed);
    atomic_fetch_add_explicit(&accumulators[bestClusterIndex].r, r, memory_order_relaxed);
    atomic_fetch_add_explicit(&accumulators[bestClusterIndex].count, 1, memory_order_relaxed);
}

// Mask-based K-means assignment and accumulation kernel for GrabCut
kernel void assignAndAccumulateWithMask(
    texture2d<float, access::read>  inTexture [[texture(0)]],
    texture2d<int, access::write>   outLabels [[texture(1)]],
    texture2d<uint, access::read>   mask [[texture(2)]],
    device const float* centroids [[buffer(0)]],
    device CentroidAccumulator* accumulators [[buffer(1)]],
    constant int& K [[buffer(2)]],
    constant bool& useBackground [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    // Read mask value to determine if this pixel should be processed
    uint maskValue = mask.read(gid).x;
    
    // GrabCut mask values:
    // 0 = GC_BGD (certain background)
    // 1 = GC_FGD (certain foreground) 
    // 2 = GC_PR_BGD (probably background)
    // 3 = GC_PR_FGD (probably foreground)
    
    bool isBackground = (maskValue == 0 || maskValue == 2);
    bool shouldProcess = (useBackground == isBackground);
    
    if (!shouldProcess) {
        // Pixel belongs to the other class, write default label and skip
        outLabels.write(0, gid);
        return;
    }
    
    // 1. Read the pixel color (our data point) - BGRA texture format
    float4 pixelColor = inTexture.read(gid);
    float3 point = float3(pixelColor.b, pixelColor.g, pixelColor.r);  // BGR order to match buffer layout
    
    // 2. Assignment Step: Find the closest centroid
    float minDistance = FLT_MAX;
    int bestClusterIndex = 0;
    
    for (int i = 0; i < K; ++i) {
        // Read centroid from flat buffer: [B0,G0,R0,B1,G1,R1,...] (BGR order)
        float3 centroid = float3(centroids[i * 3 + 0], centroids[i * 3 + 1], centroids[i * 3 + 2]);
        
        // Explicit distance calculation
        float3 diff = point - centroid;
        float dist = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        
        if (dist < minDistance) {
            minDistance = dist;
            bestClusterIndex = i;
        }
    }
    
    // 3. Write the assigned cluster index to the output label texture
    outLabels.write(bestClusterIndex, gid);
    
    // 4. Update Step: Atomically update the accumulator for the assigned cluster
    // Scale float colors to integer range for atomic operations (BGR order)
    int b = int(point.x * 255.0 + 0.5);  // B channel (point.x)
    int g = int(point.y * 255.0 + 0.5);  // G channel (point.y)
    int r = int(point.z * 255.0 + 0.5);  // R channel (point.z)
    
    atomic_fetch_add_explicit(&accumulators[bestClusterIndex].b, b, memory_order_relaxed);
    atomic_fetch_add_explicit(&accumulators[bestClusterIndex].g, g, memory_order_relaxed);
    atomic_fetch_add_explicit(&accumulators[bestClusterIndex].r, r, memory_order_relaxed);
    atomic_fetch_add_explicit(&accumulators[bestClusterIndex].count, 1, memory_order_relaxed);
}

// Centroid update and reset kernel
kernel void updateCentroids(
    device float* centroids [[buffer(0)]],
    device CentroidAccumulator* accumulators [[buffer(1)]],
    constant uint& K [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    // Bounds check for large K values with multiple threadgroups
    if (tid >= K) {
        return;
    }
    
    int count = atomic_load_explicit(&accumulators[tid].count, memory_order_relaxed);
    
    if (count > 0) {
        // Calculate the new average centroid and store in flat buffer (BGR order)
        centroids[tid * 3 + 0] = float(atomic_load_explicit(&accumulators[tid].b, memory_order_relaxed)) / float(count) / 255.0;  // B
        centroids[tid * 3 + 1] = float(atomic_load_explicit(&accumulators[tid].g, memory_order_relaxed)) / float(count) / 255.0;  // G
        centroids[tid * 3 + 2] = float(atomic_load_explicit(&accumulators[tid].r, memory_order_relaxed)) / float(count) / 255.0;  // R
    }
    
    // Reset the accumulator for the next iteration
    atomic_store_explicit(&accumulators[tid].b, 0, memory_order_relaxed);
    atomic_store_explicit(&accumulators[tid].g, 0, memory_order_relaxed);
    atomic_store_explicit(&accumulators[tid].r, 0, memory_order_relaxed);
    atomic_store_explicit(&accumulators[tid].count, 0, memory_order_relaxed);
}

// GEMM optimization: Convert texture to contiguous buffer for MPS
kernel void textureToBuffer(
    texture2d<float, access::read> inTexture [[texture(0)]],
    device float* outBuffer [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    // Read BGRA pixel (Metal texture uses BGRA format)
    float4 pixel = inTexture.read(gid);
    
    // Calculate linear index for this pixel
    uint width = inTexture.get_width();
    uint linearIndex = gid.y * width + gid.x;
    
    // Store in BGR order to match OpenCV convention: [B0, G0, R0, B1, G1, R1, ...]
    outBuffer[linearIndex * 3 + 0] = pixel.b;  // B (blue channel)
    outBuffer[linearIndex * 3 + 1] = pixel.g;  // G (green channel)  
    outBuffer[linearIndex * 3 + 2] = pixel.r;  // R (red channel)
}

// Compute squared sums for GEMM optimization: Σ(Xi²) for each sample
kernel void computeSampleSquaredSums(
    device const float* dataBuffer [[buffer(0)]],
    device float* squaredSums [[buffer(1)]],
    constant uint& numSamples [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= numSamples) {
        return;
    }
    
    // Calculate squared sum for sample tid: Σ(Xi²)
    float sum = 0.0;
    uint baseIndex = tid * 3;
    sum += dataBuffer[baseIndex + 0] * dataBuffer[baseIndex + 0]; // R²
    sum += dataBuffer[baseIndex + 1] * dataBuffer[baseIndex + 1]; // G²
    sum += dataBuffer[baseIndex + 2] * dataBuffer[baseIndex + 2]; // B²
    
    squaredSums[tid] = sum;
}

// Compute squared sums for centroids: Σ(Yj²) for each centroid
kernel void computeCentroidSquaredSums(
    device const float* centroids [[buffer(0)]],
    device float* squaredSums [[buffer(1)]],
    constant uint& K [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= K) {
        return;
    }
    
    // Calculate squared sum for centroid tid: Σ(Yj²)
    float sum = 0.0;
    uint baseIndex = tid * 3;
    sum += centroids[baseIndex + 0] * centroids[baseIndex + 0]; // R²
    sum += centroids[baseIndex + 1] * centroids[baseIndex + 1]; // G²
    sum += centroids[baseIndex + 2] * centroids[baseIndex + 2]; // B²
    
    squaredSums[tid] = sum;
}

// Find minimum distance and assign labels from GEMM result
kernel void argminAndAssign(
    device const float* sampleSquaredSums [[buffer(0)]],
    device const float* centroidSquaredSums [[buffer(1)]],
    device const float* gemmResult [[buffer(2)]],        // -2 * X·Y^T from MPS
    device int* labels [[buffer(3)]],
    constant uint& numSamples [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= numSamples) {
        return;
    }
    
    float minDistance = FLT_MAX;
    int bestLabel = 0;
    
    for (uint k = 0; k < K; ++k) {
        // Distance = Σ(Xi²) + Σ(Yj²) - 2·(X·Y^T)
        // Note: gemmResult already contains -2·(X·Y^T), so we ADD it (since it's negative)
        float distance = sampleSquaredSums[tid] + centroidSquaredSums[k] + gemmResult[tid * K + k];
        
        if (distance < minDistance) {
            minDistance = distance;
            bestLabel = int(k);
        }
    }
    
    labels[tid] = bestLabel;
}

// GPU Component Assignment kernel for GrabCut optimization
// Replaces CPU pixel-by-pixel loops with GPU parallel processing
kernel void assignComponentsKernel(
    texture2d<uint, access::read>   mask [[texture(0)]],        // GrabCut mask (GC_BGD, GC_FGD, etc.)
    texture2d<int, access::read>    bgLabels [[texture(1)]],    // Background k-means labels
    texture2d<int, access::read>    fgLabels [[texture(2)]],    // Foreground k-means labels
    texture2d<uint, access::write>  compIdxs [[texture(3)]],    // Output component assignments
    uint2 gid [[thread_position_in_grid]])
{
    // Check bounds
    if (gid.x >= mask.get_width() || gid.y >= mask.get_height()) {
        return;
    }
    
    // Read mask value to determine background vs foreground
    uint maskValue = mask.read(gid).x;
    
    // GrabCut mask values:
    // 0 = GC_BGD (certain background)
    // 1 = GC_FGD (certain foreground) 
    // 2 = GC_PR_BGD (probably background)
    // 3 = GC_PR_FGD (probably foreground)
    
    int component = 0;
    
    // Background pixels (GC_BGD = 0 or GC_PR_BGD = 2)
    if (maskValue == 0 || maskValue == 2) {
        component = bgLabels.read(gid).x;
    }
    // Foreground pixels (GC_FGD = 1 or GC_PR_FGD = 3)
    else if (maskValue == 1 || maskValue == 3) {
        component = fgLabels.read(gid).x;
    }
    
    // Ensure component is in valid range [0, 4] (5 components per class)
    component = clamp(component, 0, 4);
    
    // Write component assignment - always 0-4 for both BG and FG
    // (mask value determines which GMM to use, not the component index)
    compIdxs.write(uint(component), gid);
}

// GPU K-means Optimization Kernels (Phase 1-3)

// Phase 1: Count valid pixels on GPU (eliminates CPU pixel scanning)
kernel void countValidPixelsKernel(
    texture2d<uint, access::read> mask [[texture(0)]],
    device atomic<uint>* pixelCount [[buffer(0)]],
    device uint2* validPixelIndices [[buffer(1)]],
    constant bool& useBackground [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= mask.get_width() || gid.y >= mask.get_height()) return;
    
    uint maskVal = mask.read(gid).x;
    bool isBackground = (maskVal == 0 || maskVal == 2);
    
    if (isBackground == useBackground) {
        uint index = atomic_fetch_add_explicit(pixelCount, 1, memory_order_relaxed);
        if (validPixelIndices && index < mask.get_width() * mask.get_height()) {
            validPixelIndices[index] = gid;
        }
    }
}

// Phase 2: Compute bounding box on GPU (eliminates CPU min/max computation)
kernel void computeBoundingBoxKernel(
    texture2d<float, access::read> image [[texture(0)]],
    texture2d<uint, access::read> mask [[texture(1)]],
    device atomic<uint>* minValues [[buffer(0)]], // [B_min, G_min, R_min] as uint (for atomic ops)
    device atomic<uint>* maxValues [[buffer(1)]], // [B_max, G_max, R_max] as uint (for atomic ops)
    constant bool& useBackground [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= image.get_width() || gid.y >= image.get_height()) return;
    
    uint maskVal = mask.read(gid).x;
    bool isBackground = (maskVal == 0 || maskVal == 2);
    
    if (isBackground == useBackground) {
        float4 pixel = image.read(gid);
        float3 bgr = float3(pixel.b, pixel.g, pixel.r) * 255.0f;
        
        // Convert to uint for atomic operations (preserve precision)
        uint3 bgrUint = uint3(bgr.x, bgr.y, bgr.z);
        
        // Atomic min/max operations for bounding box
        atomic_fetch_min_explicit(&minValues[0], bgrUint.x, memory_order_relaxed); // B
        atomic_fetch_min_explicit(&minValues[1], bgrUint.y, memory_order_relaxed); // G
        atomic_fetch_min_explicit(&minValues[2], bgrUint.z, memory_order_relaxed); // R
        atomic_fetch_max_explicit(&maxValues[0], bgrUint.x, memory_order_relaxed); // B
        atomic_fetch_max_explicit(&maxValues[1], bgrUint.y, memory_order_relaxed); // G
        atomic_fetch_max_explicit(&maxValues[2], bgrUint.z, memory_order_relaxed); // R
    }
}

// Phase 3: Initialize centroids on GPU (eliminates CPU random generation)
kernel void initializeCentroidsKernel(
    device float* centroids [[buffer(0)]],      // Output centroids [K×3]
    constant uint* boundingBoxMin [[buffer(1)]], // [min_B, min_G, min_R] as uint
    constant uint* boundingBoxMax [[buffer(2)]], // [max_B, max_G, max_R] as uint
    constant uint& randomSeed [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= K) return;
    
    // Per-thread random number generation (Linear Congruential Generator)
    uint rngState = randomSeed + tid * 1103515245u + 12345u;
    
    const float margin = 1.0f / 3.0f;
    
    for (int c = 0; c < 3; ++c) {
        float minVal = float(boundingBoxMin[c]);
        float maxVal = float(boundingBoxMax[c]);
        float range = maxVal - minVal;
        
        // Handle degenerate case where min == max
        if (range < 1.0f) {
            range = 1.0f;
        }
        
        // Generate random value in expanded range (matches OpenCV CPU behavior)
        rngState = rngState * 1103515245u + 12345u;
        float randomFloat = float(rngState) / float(UINT_MAX);
        float expandedRandom = randomFloat * (1.0f + margin * 2.0f) - margin;
        
        float centroidVal = expandedRandom * range + minVal;
        centroidVal = clamp(centroidVal, 0.0f, 255.0f); // Clamp to valid color range
        centroids[tid * 3 + c] = centroidVal / 255.0f; // Normalize to [0,1] for Metal
    }
}

)";

// Pipeline state cache using dispatch_once pattern
static id<MTLComputePipelineState> getAssignAndAccumulateWithMaskPipeline() {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            id<MTLDevice> device = MetalContext::getInstance().device;
            NSError *error = nil;
            
            NSString *kernelSource = [NSString stringWithCString:kmeansShaderSource 
                                                        encoding:NSUTF8StringEncoding];
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource 
                                                          options:nil error:&error];
            
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"assignAndAccumulateWithMask"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create assignAndAccumulateWithMask function");
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

static id<MTLComputePipelineState> getUpdateCentroidsPipeline() {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            id<MTLDevice> device = MetalContext::getInstance().device;
            NSError *error = nil;
            
            NSString *kernelSource = [NSString stringWithCString:kmeansShaderSource 
                                                        encoding:NSUTF8StringEncoding];
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource 
                                                          options:nil error:&error];
            
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"updateCentroids"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create updateCentroids function");
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

// NEW: GPU K-means Optimization Pipeline Accessors (Phase 1-3)
static id<MTLComputePipelineState> getCountValidPixelsPipeline() {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            id<MTLDevice> device = MetalContext::getInstance().device;
            NSError *error = nil;
            
            NSString *kernelSource = [NSString stringWithCString:kmeansShaderSource 
                                                        encoding:NSUTF8StringEncoding];
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource 
                                                          options:nil error:&error];
            
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"countValidPixelsKernel"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create countValidPixelsKernel function");
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

static id<MTLComputePipelineState> getComputeBoundingBoxPipeline() {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            id<MTLDevice> device = MetalContext::getInstance().device;
            NSError *error = nil;
            
            NSString *kernelSource = [NSString stringWithCString:kmeansShaderSource 
                                                        encoding:NSUTF8StringEncoding];
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource 
                                                          options:nil error:&error];
            
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"computeBoundingBoxKernel"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create computeBoundingBoxKernel function");
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

static id<MTLComputePipelineState> getInitializeCentroidsPipeline() {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            id<MTLDevice> device = MetalContext::getInstance().device;
            NSError *error = nil;
            
            NSString *kernelSource = [NSString stringWithCString:kmeansShaderSource 
                                                        encoding:NSUTF8StringEncoding];
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource 
                                                          options:nil error:&error];
            
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"initializeCentroidsKernel"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create initializeCentroidsKernel function");
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

static id<MTLComputePipelineState> getAssignComponentsPipeline() {
    static id<MTLComputePipelineState> pipeline = nil;
    static dispatch_once_t onceToken;
    
    dispatch_once(&onceToken, ^{
        @autoreleasepool {
            id<MTLDevice> device = MetalContext::getInstance().device;
            NSError *error = nil;
            
            NSString *kernelSource = [NSString stringWithCString:kmeansShaderSource 
                                                        encoding:NSUTF8StringEncoding];
            id<MTLLibrary> library = [device newLibraryWithSource:kernelSource 
                                                          options:nil error:&error];
            
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"assignComponentsKernel"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create assignComponentsKernel function");
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

} // anonymous namespace

namespace cv { namespace metal {

// Internal implementation for general k-means
double kmeansImpl(const MetalMat& data, int K, MetalMat& bestLabels, 
                 TermCriteria criteria, int attempts, int flags, MetalMat& centers, Stream* stream) {
    CV_Error(Error::StsNotImplemented, "General k-means not implemented yet. Use kmeansClusterByMask for GrabCut.");
    return 0.0;
}

// Public API implementation - sync version
double kmeans(const MetalMat& data, int K, MetalMat& bestLabels, 
             TermCriteria criteria, int attempts, int flags, MetalMat& centers) {
    Stream defaultStream;
    double result = kmeansImpl(data, K, bestLabels, criteria, attempts, flags, centers, &defaultStream);
    defaultStream.commitAndWait();
    return result;
}

// Public API implementation - stream version
double kmeans(const MetalMat& data, int K, MetalMat& bestLabels, 
             TermCriteria criteria, int attempts, int flags, MetalMat& centers, Stream& stream) {
    return kmeansImpl(data, K, bestLabels, criteria, attempts, flags, centers, &stream);
}

// Mask-based K-means for GrabCut initialization
void kmeansClusterByMask(const MetalMat& inImg, const MetalMat& mask, bool useBackground,
                        MetalMat& outLabels, cv::Mat& centroids, Stream& stream) {
    CV_Assert(!inImg.empty());
    CV_Assert(!mask.empty());
    CV_Assert(inImg.size() == mask.size());
    CV_Assert(inImg.type() == CV_8UC4 || inImg.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_8UC1);
    
    const int K = 5; // Always 5 components for GrabCut GMM
    const int maxIterations = 10;
    const float epsilon = 1.0f;
    
    // Get Metal context
    MetalContext& ctx = MetalContext::getInstance();
    id<MTLDevice> device = ctx.device;
    
    // Create output labels texture (same size as input)
    outLabels.create(inImg.size(), CV_32SC1);
    
    // Create centroid and accumulator buffers
    size_t centroidSize = K * 3 * sizeof(float);  // K centroids × 3 channels (BGR)
    size_t accumulatorSize = K * sizeof(CentroidAccumulator);
    
    id<MTLBuffer> centroidsBuffer = [device newBufferWithLength:centroidSize 
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> accumulatorsBuffer = [device newBufferWithLength:accumulatorSize 
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuffer = [device newBufferWithBytes:&K length:sizeof(int) 
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> useBackgroundBuffer = [device newBufferWithBytes:&useBackground 
                                                             length:sizeof(bool)
                                                            options:MTLResourceStorageModeShared];
    
    // 🚀 PHASE 1-3 GPU OPTIMIZATION: Replace ALL CPU bottlenecks with GPU kernels
    // Eliminates: 10.4MB downloads + 2M CPU iterations + sequential computation
    
    // Phase 1: Count valid pixels on GPU (eliminates CPU pixel scanning)
    id<MTLBuffer> pixelCountBuffer = [device newBufferWithLength:sizeof(uint32_t) 
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> validIndicesBuffer = [device newBufferWithLength:inImg.cols() * inImg.rows() * sizeof(uint64_t)
                                                           options:MTLResourceStorageModeShared];
    
    // Clear pixel count
    *(uint32_t*)pixelCountBuffer.contents = 0;
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getCountValidPixelsPipeline()];
        [encoder setTexture:mask.texture() atIndex:0];
        [encoder setBuffer:pixelCountBuffer offset:0 atIndex:0];
        [encoder setBuffer:validIndicesBuffer offset:0 atIndex:1];
        [encoder setBuffer:useBackgroundBuffer offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake((mask.cols() + 15) / 16, (mask.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Sync to read pixel count
    stream.syncCPU();
    uint32_t validPixelCount = *(uint32_t*)pixelCountBuffer.contents;
    
    if (validPixelCount < K) {
        CV_Error(Error::StsBadArg, cv::format("Not enough pixels in the target class for k-means: %d < %d", 
                                              validPixelCount, K));
    }
    
    // Phase 2: Compute bounding box on GPU (eliminates CPU min/max computation)
    id<MTLBuffer> boundingBoxMinBuffer = [device newBufferWithLength:3 * sizeof(uint32_t) 
                                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> boundingBoxMaxBuffer = [device newBufferWithLength:3 * sizeof(uint32_t) 
                                                             options:MTLResourceStorageModeShared];
    
    // Initialize bounding box with extreme values
    uint32_t* minPtr = (uint32_t*)boundingBoxMinBuffer.contents;
    uint32_t* maxPtr = (uint32_t*)boundingBoxMaxBuffer.contents;
    for (int i = 0; i < 3; i++) {
        minPtr[i] = UINT32_MAX;  // Will be reduced to actual min
        maxPtr[i] = 0;           // Will be increased to actual max
    }
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getComputeBoundingBoxPipeline()];
        [encoder setTexture:inImg.texture() atIndex:0];
        [encoder setTexture:mask.texture() atIndex:1];
        [encoder setBuffer:boundingBoxMinBuffer offset:0 atIndex:0];
        [encoder setBuffer:boundingBoxMaxBuffer offset:0 atIndex:1];
        [encoder setBuffer:useBackgroundBuffer offset:0 atIndex:2];
        
        MTLSize gridSize = MTLSizeMake((inImg.cols() + 15) / 16, (inImg.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Phase 3: Initialize centroids on GPU (eliminates CPU random generation)
    uint32_t randomSeed = cv::theRNG().next(); // Get seed from OpenCV RNG for consistency
    
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:getInitializeCentroidsPipeline()];
        [encoder setBuffer:centroidsBuffer offset:0 atIndex:0];
        [encoder setBuffer:boundingBoxMinBuffer offset:0 atIndex:1];
        [encoder setBuffer:boundingBoxMaxBuffer offset:0 atIndex:2];
        [encoder setBytes:&randomSeed length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&K length:sizeof(uint32_t) atIndex:4];
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(std::min(K, 64), 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((K + 63) / 64, 1, 1);
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }
    
    // Get pipelines
    id<MTLComputePipelineState> assignPipeline = getAssignAndAccumulateWithMaskPipeline();
    id<MTLComputePipelineState> updatePipeline = getUpdateCentroidsPipeline();
    
    // Main k-means loop
    for (int iter = 0; iter < maxIterations; iter++) {
        // Clear accumulators
        memset(accumulatorsBuffer.contents, 0, accumulatorSize);
        
        // Assignment step with mask filtering
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
            [encoder setComputePipelineState:assignPipeline];
            [encoder setTexture:inImg.texture() atIndex:0];  // input image
            [encoder setTexture:outLabels.texture() atIndex:1];  // output labels
            [encoder setTexture:mask.texture() atIndex:2];  // mask
            [encoder setBuffer:centroidsBuffer offset:0 atIndex:0];
            [encoder setBuffer:accumulatorsBuffer offset:0 atIndex:1];
            [encoder setBuffer:kBuffer offset:0 atIndex:2];
            [encoder setBuffer:useBackgroundBuffer offset:0 atIndex:3];
            
            MTLSize gridSize = MTLSizeMake((inImg.cols() + 15) / 16, (inImg.rows() + 15) / 16, 1);
            MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }
        
        // Update centroids
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
            [encoder setComputePipelineState:updatePipeline];
            [encoder setBuffer:centroidsBuffer offset:0 atIndex:0];
            [encoder setBuffer:accumulatorsBuffer offset:0 atIndex:1];
            [encoder setBuffer:kBuffer offset:0 atIndex:2];
            
            MTLSize threadsPerThreadgroup = MTLSizeMake(std::min(K, 64), 1, 1);
            MTLSize threadgroups = MTLSizeMake((K + 63) / 64, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }
    }
    
    // Sync to read final centroids
    stream.syncCPU();
    
    // Convert centroids to output format (K×3 Mat in BGR order, [0,255] range)
    centroids.create(K, 3, CV_32F);
    float* centroidsPtr = (float*)centroidsBuffer.contents;
    for (int i = 0; i < K; i++) {
        centroids.at<float>(i, 0) = centroidsPtr[i * 3 + 0] * 255.0f;  // B
        centroids.at<float>(i, 1) = centroidsPtr[i * 3 + 1] * 255.0f;  // G
        centroids.at<float>(i, 2) = centroidsPtr[i * 3 + 2] * 255.0f;  // R
    }
}

// GPU Component Assignment for GrabCut optimization (Phase 1.3)
void assignComponents(const MetalMat& mask, const MetalMat& bgLabels, const MetalMat& fgLabels, 
                     MetalMat& compIdxs, Stream& stream) {
    CV_Assert(!mask.empty());
    CV_Assert(!bgLabels.empty());
    CV_Assert(!fgLabels.empty());
    CV_Assert(mask.size() == bgLabels.size());
    CV_Assert(mask.size() == fgLabels.size());
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(bgLabels.type() == CV_32SC1);
    CV_Assert(fgLabels.type() == CV_32SC1);
    
    // Create output texture for component assignments
    compIdxs.create(mask.size(), CV_8UC1);
    
    // Get pipeline
    id<MTLComputePipelineState> pipeline = getAssignComponentsPipeline();
    if (!pipeline) {
        CV_Error(Error::StsError, "Failed to get assignComponents pipeline");
    }
    
    // Encode GPU component assignment kernel
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(stream);
        [encoder setComputePipelineState:pipeline];
        [encoder setTexture:mask.texture() atIndex:0];      // GrabCut mask
        [encoder setTexture:bgLabels.texture() atIndex:1];  // Background k-means labels  
        [encoder setTexture:fgLabels.texture() atIndex:2];  // Foreground k-means labels
        [encoder setTexture:compIdxs.texture() atIndex:3];  // Output component assignments
        
        // Dispatch with 16x16 thread groups
        MTLSize gridSize = MTLSizeMake((mask.cols() + 15) / 16, (mask.rows() + 15) / 16, 1);
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
    }
    
    // Note: No sync/commit here - caller manages stream lifecycle
}

}} // namespace cv::metal

#else // !HAVE_METAL

namespace cv { namespace metal {

double kmeans(const cv::Mat&, int, cv::Mat&, cv::TermCriteria, int, int, cv::Mat&, cv::metal::Stream&)
{
    CV_Error(cv::Error::StsNotImplemented, "Metal backend is not available in this build.");
    return 0.0;
}

double kmeans(const cv::Mat& data, int K, cv::Mat& bestLabels,
              cv::TermCriteria criteria, int attempts, int flags)
{
    cv::Mat centers;
    return kmeans(data, K, bestLabels, criteria, attempts, flags, centers);
}

double kmeans(const cv::Mat&, int, cv::Mat&, cv::TermCriteria, int, int, cv::Mat&)
{
    CV_Error(cv::Error::StsNotImplemented, "Metal backend is not available in this build.");
    return 0.0;
}

}} // namespace cv::metal

#endif // HAVE_METAL 