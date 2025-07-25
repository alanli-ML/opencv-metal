// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_precomp.hpp"
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
)";

// Pipeline state cache using dispatch_once pattern
static id<MTLComputePipelineState> getAssignAndAccumulatePipeline() {
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
                return;
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"assignAndAccumulate"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create assignAndAccumulate function");
                return;
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
                return;
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"updateCentroids"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create updateCentroids function");
                return;
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

// GEMM optimization pipeline getters
static id<MTLComputePipelineState> getTextureToBufferPipeline() {
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
                return;
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"textureToBuffer"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create textureToBuffer function");
                return;
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

static id<MTLComputePipelineState> getComputeSampleSquaredSumsPipeline() {
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
                return;
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"computeSampleSquaredSums"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create computeSampleSquaredSums function");
                return;
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

static id<MTLComputePipelineState> getComputeCentroidSquaredSumsPipeline() {
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
                return;
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"computeCentroidSquaredSums"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create computeCentroidSquaredSums function");
                return;
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

static id<MTLComputePipelineState> getArgminAndAssignPipeline() {
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
                return;
            }
            
            id<MTLFunction> function = [library newFunctionWithName:@"argminAndAssign"];
            if (!function) {
                CV_Error(Error::StsError, "Failed to create argminAndAssign function");
                return;
            }
            
            pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (error) {
                CV_Error(Error::StsError, [[error localizedDescription] UTF8String]);
            }
        }
    });
    
    return pipeline;
}

// Helper function to initialize centroids randomly from image
void initializeCentroidsRandom(const MetalMat& src, std::vector<cv::Point3f>& centroids, int K, Stream& stream) {
    Mat cpu_src;
    src.download(cpu_src, stream, true);  // Synchronous download since we need the data immediately
    
    // Use OpenCV's RNG for consistent seeding with CPU implementation
    RNG& rng = theRNG();
    
    centroids.clear();
    centroids.reserve(K);
    
    // CRITICAL FIX: Match CPU k-means KMEANS_RANDOM_CENTERS behavior
    // CPU uses generateRandomCenter() which generates centers within bounding box, not actual pixel samples!
    
    // Calculate bounding box for each channel (match CPU k-means logic exactly)
    Vec2f box[3]; // [min, max] for each BGR channel
    
    // Initialize with first pixel
    if (cpu_src.channels() == 3) {
        Vec3b first_pixel = cpu_src.at<Vec3b>(0, 0);
        for (int j = 0; j < 3; j++) {
            float val = first_pixel[j] / 255.0f;
            box[j] = Vec2f(val, val);
        }
        
        // Find actual min/max for each channel across all pixels
        for (int y = 0; y < cpu_src.rows; y++) {
            for (int x = 0; x < cpu_src.cols; x++) {
                Vec3b pixel = cpu_src.at<Vec3b>(y, x);
                for (int j = 0; j < 3; j++) {
                    float val = pixel[j] / 255.0f;
                    box[j][0] = std::min(box[j][0], val);  // min
                    box[j][1] = std::max(box[j][1], val);  // max
                }
            }
        }
    } else if (cpu_src.channels() == 4) {
        Vec4b first_pixel = cpu_src.at<Vec4b>(0, 0);
        for (int j = 0; j < 3; j++) {
            float val = first_pixel[j] / 255.0f;
            box[j] = Vec2f(val, val);
        }
        
        // Find actual min/max for each channel across all pixels (ignore alpha)
        for (int y = 0; y < cpu_src.rows; y++) {
            for (int x = 0; x < cpu_src.cols; x++) {
                Vec4b pixel = cpu_src.at<Vec4b>(y, x);
                for (int j = 0; j < 3; j++) {
                    float val = pixel[j] / 255.0f;
                    box[j][0] = std::min(box[j][0], val);  // min
                    box[j][1] = std::max(box[j][1], val);  // max
                }
            }
        }
    } else {
        CV_Error(Error::StsUnsupportedFormat, "Unsupported number of channels for K-means");
    }
    
    // Generate K random centers within the bounding box (exact copy of CPU generateRandomCenter logic)
    const int dims = 3;  // BGR channels
    float margin = 1.0f / dims;
    
    for (int i = 0; i < K; i++) {
        Point3f center;
        
        // Generate random center within bounding box for each channel (match CPU exactly)
        center.x = ((float)rng * (1.0f + margin * 2.0f) - margin) * (box[0][1] - box[0][0]) + box[0][0]; // B
        center.y = ((float)rng * (1.0f + margin * 2.0f) - margin) * (box[1][1] - box[1][0]) + box[1][0]; // G  
        center.z = ((float)rng * (1.0f + margin * 2.0f) - margin) * (box[2][1] - box[2][0]) + box[2][0]; // R
        
        centroids.push_back(center);
    }
}

// Helper function to compute compactness (sum of squared distances to centroids)
// FIXED: Now calculates compactness in [0,255] space to match CPU implementation
double computeCompactness(const MetalMat& data, const MetalMat& labels, 
                         const std::vector<cv::Point3f>& centroids, Stream& stream) {
    Mat cpu_data, cpu_labels;
    data.download(cpu_data, stream, true);
    labels.download(cpu_labels, stream, true);
    
    double compactness = 0.0;
    
    for (int y = 0; y < cpu_data.rows; y++) {
        for (int x = 0; x < cpu_data.cols; x++) {
            int label = cpu_labels.at<int>(y, x);
            
            cv::Point3f pixel;
            if (cpu_data.channels() == 3) {
                Vec3b bgr = cpu_data.at<Vec3b>(y, x);
                // CRITICAL FIX: Keep pixel values in [0,255] range like CPU implementation
                pixel = cv::Point3f(bgr[0], bgr[1], bgr[2]); // No normalization - use actual pixel values
            } else if (cpu_data.channels() == 4) {
                Vec4b bgra = cpu_data.at<Vec4b>(y, x);
                // CRITICAL FIX: Keep pixel values in [0,255] range like CPU implementation
                pixel = cv::Point3f(bgra[0], bgra[1], bgra[2]); // No normalization - use actual pixel values
            }
            
            // CRITICAL FIX: Scale centroids from [0,1] back to [0,255] for comparison
            const cv::Point3f& centroid_normalized = centroids[label];
            cv::Point3f centroid_scaled(centroid_normalized.x * 255.0f, 
                                       centroid_normalized.y * 255.0f, 
                                       centroid_normalized.z * 255.0f);
            
            double dx = pixel.x - centroid_scaled.x;
            double dy = pixel.y - centroid_scaled.y;
            double dz = pixel.z - centroid_scaled.z;
            compactness += dx*dx + dy*dy + dz*dz;
        }
    }
    
    return compactness;
}

// Helper function to compute compactness for GEMM implementation (centroids in [0,1] space)
double computeCompactnessGEMM(const MetalMat& data, const MetalMat& labels, 
                              const std::vector<cv::Point3f>& centroids_01, Stream& stream) {
    Mat cpu_data, cpu_labels;
    data.download(cpu_data, stream, true);
    labels.download(cpu_labels, stream, true);
    
    double compactness = 0.0;
    
    for (int y = 0; y < cpu_data.rows; y++) {
        for (int x = 0; x < cpu_data.cols; x++) {
            int label = cpu_labels.at<int>(y, x);
            
            cv::Point3f pixel;
            if (cpu_data.channels() == 3) {
                // Metal texture downloads as uchar values in [0,255] range
                Vec3b bgr = cpu_data.at<Vec3b>(y, x);
                // Convert to [0,1] range to match centroids
                pixel = cv::Point3f(bgr[0] / 255.0f, bgr[1] / 255.0f, bgr[2] / 255.0f); 
            } else if (cpu_data.channels() == 4) {
                // Metal texture downloads as uchar values in [0,255] range  
                Vec4b bgra = cpu_data.at<Vec4b>(y, x);
                // Convert to [0,1] range to match centroids
                pixel = cv::Point3f(bgra[0] / 255.0f, bgra[1] / 255.0f, bgra[2] / 255.0f);
            }
            
            // Both pixel and centroids are now in [0,1] space
            const cv::Point3f& centroid_01 = centroids_01[label];
            
            double dx = pixel.x - centroid_01.x;
            double dy = pixel.y - centroid_01.y;
            double dz = pixel.z - centroid_01.z;
            compactness += dx*dx + dy*dy + dz*dz;
        }
    }
    
    return compactness;
}

// Check convergence by comparing old and new centroids
bool checkConvergence(const std::vector<cv::Point3f>& oldCentroids, 
                     const std::vector<cv::Point3f>& newCentroids, 
                     double epsilon) {
    for (size_t i = 0; i < oldCentroids.size(); i++) {
        double dx = oldCentroids[i].x - newCentroids[i].x;
        double dy = oldCentroids[i].y - newCentroids[i].y;
        double dz = oldCentroids[i].z - newCentroids[i].z;
        double distance = sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance > epsilon) {
            return false;
        }
    }
    return true;
}

// GEMM-optimized K-means implementation using MPS for fast distance calculation
double kmeansGEMMOptimized(const MetalMat& data, int K, MetalMat& bestLabels,
                          TermCriteria criteria, int attempts, int flags,
                          MetalMat& centers, Stream* stream = nullptr) {
    
    // Input validation
    CV_Assert(data.type() == CV_8UC3 || data.type() == CV_8UC4);
    CV_Assert(K > 0);
    CV_Assert(attempts > 0);
    
    if (flags == KMEANS_RANDOM_CENTERS || (flags & (KMEANS_PP_CENTERS | KMEANS_USE_INITIAL_LABELS)) == 0) {
        // Supported initialization
    } else {
        CV_Error(Error::StsNotImplemented, "Only KMEANS_RANDOM_CENTERS is currently supported in GEMM-optimized version");
    }
    
    // Create stream if none provided
    Stream defaultStream;
    bool useDefaultStream = (stream == nullptr);
    if (useDefaultStream) {
        stream = &defaultStream;
    }
    
    // Get device and context
    id<MTLDevice> device = MetalContext::getInstance().device;
    
    // Create MPS context for GEMM operations
    MPSGraphDevice* mpsDevice = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraph* graph = [[MPSGraph alloc] init];
    
    int numSamples = data.rows() * data.cols();
    
    // Buffer for texture-to-buffer conversion (keep outside autorelease pool for later use)
    id<MTLBuffer> dataBuffer = [device newBufferWithLength:numSamples * 3 * sizeof(float) 
                                                    options:MTLResourceStorageModeShared];
    
    // Convert texture to buffer for MPS processing
    @autoreleasepool {
        id<MTLComputePipelineState> textureToBufferPipeline = getTextureToBufferPipeline();
        id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(*stream);
        
        [encoder setComputePipelineState:textureToBufferPipeline];
        [encoder setTexture:data.texture() atIndex:0];
        [encoder setBuffer:dataBuffer offset:0 atIndex:0];
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(
            (data.cols() + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            (data.rows() + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            1
        );
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }
    
    // Create output matrices
    bestLabels.create(data.size(), CV_32S);
    centers.create(K, 1, CV_32FC4); // MetalMat format requirement
    
    double bestCompactness = DBL_MAX;
    
    for (int attempt = 0; attempt < attempts; attempt++) {
        // Initialize centroids randomly
        std::vector<cv::Point3f> initialCentroids;
        initializeCentroidsRandom(data, initialCentroids, K, *stream); // Pass stream
        
        // Create centroid buffer for MPS (K×3)
        id<MTLBuffer> centroidBuffer = [device newBufferWithLength:K * 3 * sizeof(float) 
                                                           options:MTLResourceStorageModeShared];
        float* initCentroidPtr = (float*)[centroidBuffer contents];
        
        // FIXED: No need for BGR to RGB conversion - MetalMat upload preserves BGR order correctly
        for (int i = 0; i < K; i++) {
            initCentroidPtr[i * 3 + 0] = initialCentroids[i].x;  // B = BGR.x (Blue)
            initCentroidPtr[i * 3 + 1] = initialCentroids[i].y;  // G = BGR.y (Green)
            initCentroidPtr[i * 3 + 2] = initialCentroids[i].z;  // R = BGR.z (Red)
            
            // CRITICAL FIX: Ensure no centroid is exactly zero (breaks GEMM distance calculation)
            float magnitude = sqrt(initCentroidPtr[i * 3 + 0] * initCentroidPtr[i * 3 + 0] +
                                  initCentroidPtr[i * 3 + 1] * initCentroidPtr[i * 3 + 1] +
                                  initCentroidPtr[i * 3 + 2] * initCentroidPtr[i * 3 + 2]);
            if (magnitude < 1e-6f) {
                // Use much smaller offset to minimize compactness impact
                initCentroidPtr[i * 3 + 0] += 1e-6f;  // Was 1e-4f, now 1e-6f
                initCentroidPtr[i * 3 + 1] += 1e-6f;  // This becomes 0.000255 instead of 0.0255 in [0,255] space
                initCentroidPtr[i * 3 + 2] += 1e-6f;
            }
        }
        
        // Buffers for squared sums
        id<MTLBuffer> sampleSquaredSums = [device newBufferWithLength:numSamples * sizeof(float) 
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> centroidSquaredSums = [device newBufferWithLength:K * sizeof(float) 
                                                                options:MTLResourceStorageModeShared];
        
        // Create label buffer outside iteration loop 
        id<MTLBuffer> labelBuffer = [device newBufferWithLength:numSamples * sizeof(int) 
                                                        options:MTLResourceStorageModeShared];
        
        // Main iteration loop
        for (int iter = 0; iter < criteria.maxCount; iter++) {
            // Compute squared sums for both samples and centroids
            @autoreleasepool {
                id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(*stream);
                
                // Compute sample squared sums: Σ(Xi²)
                [encoder setComputePipelineState:getComputeSampleSquaredSumsPipeline()];
                [encoder setBuffer:dataBuffer offset:0 atIndex:0];
                [encoder setBuffer:sampleSquaredSums offset:0 atIndex:1];
                [encoder setBytes:&numSamples length:sizeof(uint) atIndex:2];
                [encoder dispatchThreads:MTLSizeMake(numSamples, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                
                // Compute centroid squared sums: Σ(Yj²)
                [encoder setComputePipelineState:getComputeCentroidSquaredSumsPipeline()];
                [encoder setBuffer:centroidBuffer offset:0 atIndex:0];
                [encoder setBuffer:centroidSquaredSums offset:0 atIndex:1];
                uint KValue = K;
                [encoder setBytes:&KValue length:sizeof(uint) atIndex:2];
                [encoder dispatchThreads:MTLSizeMake(K, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                
                [encoder endEncoding];
            }
            
            // MPS GEMM operation: Compute -2 * X·Y^T
            // This provides the 25x speedup mentioned in the performance analysis
            id<MTLBuffer> gemmResultBuffer = [device newBufferWithLength:numSamples * K * sizeof(float) 
                                                                 options:MTLResourceStorageModeShared];
            
            @autoreleasepool {
                // Create MPS matrix descriptors
                MPSMatrixDescriptor* dataDescriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:numSamples 
                                                                                            columns:3 
                                                                                           rowBytes:3 * sizeof(float) 
                                                                                           dataType:MPSDataTypeFloat32];
                
                MPSMatrixDescriptor* centroidDescriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:K 
                                                                                               columns:3 
                                                                                              rowBytes:3 * sizeof(float) 
                                                                                              dataType:MPSDataTypeFloat32];
                
                MPSMatrixDescriptor* resultDescriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:numSamples 
                                                                                              columns:K 
                                                                                             rowBytes:K * sizeof(float) 
                                                                                             dataType:MPSDataTypeFloat32];
                
                // Create MPS matrices
                MPSMatrix* dataMatrix = [[MPSMatrix alloc] initWithBuffer:dataBuffer descriptor:dataDescriptor];
                MPSMatrix* centroidMatrix = [[MPSMatrix alloc] initWithBuffer:centroidBuffer descriptor:centroidDescriptor];
                MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:gemmResultBuffer descriptor:resultDescriptor];
                
                // Create and encode matrix multiplication: X·Y^T
                MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device 
                                                                                      transposeLeft:NO 
                                                                                     transposeRight:YES 
                                                                                         resultRows:numSamples 
                                                                                      resultColumns:K 
                                                                                    interiorColumns:3 
                                                                                              alpha:-2.0  // Multiply by -2
                                                                                               beta:0.0];
                
                id<MTLCommandBuffer> gemmCmdBuf = StreamAccessor::getCommandBuffer(*stream);
                [matmul encodeToCommandBuffer:gemmCmdBuf 
                                   leftMatrix:dataMatrix 
                                  rightMatrix:centroidMatrix 
                                 resultMatrix:resultMatrix];
            }
            
            // Now use argmin kernel to combine distance components and assign labels
            @autoreleasepool {
                id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(*stream);
                
                [encoder setComputePipelineState:getArgminAndAssignPipeline()];
                [encoder setBuffer:sampleSquaredSums offset:0 atIndex:0];
                [encoder setBuffer:centroidSquaredSums offset:0 atIndex:1];
                [encoder setBuffer:gemmResultBuffer offset:0 atIndex:2];
                [encoder setBuffer:labelBuffer offset:0 atIndex:3];
                uint numSamplesValue = numSamples;
                uint KValue = K;
                [encoder setBytes:&numSamplesValue length:sizeof(uint) atIndex:4];
                [encoder setBytes:&KValue length:sizeof(uint) atIndex:5];
                
                [encoder dispatchThreads:MTLSizeMake(numSamples, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                [encoder endEncoding];
            }
            
            // Sync to read label data and perform centroid update on CPU
            stream->syncCPU();
            
            // Copy labels to output texture (convert from 1D buffer to 2D texture)
            @autoreleasepool {
                // For now, we'll use a simple approach - copy to CPU and back
                // TODO: Implement a direct buffer-to-texture copy kernel for better performance
                int* labelData = (int*)[labelBuffer contents];
                Mat labelsCpu(data.rows(), data.cols(), CV_32S);
                
                // Copy labels from buffer to Mat (accounting for row-major layout)
                for (int y = 0; y < data.rows(); y++) {
                    for (int x = 0; x < data.cols(); x++) {
                        int linearIndex = y * data.cols() + x;
                        labelsCpu.at<int>(y, x) = labelData[linearIndex];
                    }
                }
                
                bestLabels.upload(labelsCpu);
            }
            
            // Update centroids using optimized batched reduction
            @autoreleasepool {
                // Create temporary buffers for new centroids
                id<MTLBuffer> newCentroidBuffer = [device newBufferWithLength:K * 3 * sizeof(float) 
                                                                      options:MTLResourceStorageModeShared];
                id<MTLBuffer> centroidCountBuffer = [device newBufferWithLength:K * sizeof(int) 
                                                                        options:MTLResourceStorageModeShared];
                
                // Zero out the buffers
                memset([newCentroidBuffer contents], 0, K * 3 * sizeof(float));
                memset([centroidCountBuffer contents], 0, K * sizeof(int));
                
                // Use a simple CPU-based reduction for now
                // TODO: Implement GPU-based reduction kernel for better performance
                int* labelData = (int*)[labelBuffer contents];
                float* dataPtr = (float*)[dataBuffer contents];
                float* newCentroidPtr = (float*)[newCentroidBuffer contents];
                int* countPtr = (int*)[centroidCountBuffer contents];
                
                // Accumulate samples for each cluster
                for (int i = 0; i < numSamples; i++) {
                    int label = labelData[i];
                    if (label >= 0 && label < K) {
                        countPtr[label]++;
                        // CRITICAL FIX: dataPtr is in BGR order, not RGB!
                        newCentroidPtr[label * 3 + 0] += dataPtr[i * 3 + 0]; // B (blue channel)
                        newCentroidPtr[label * 3 + 1] += dataPtr[i * 3 + 1]; // G (green channel)
                        newCentroidPtr[label * 3 + 2] += dataPtr[i * 3 + 2]; // R (red channel)
                    }
                }
                
                // Compute averages and check for convergence
                bool converged = true;
                float* oldCentroidPtr = (float*)[centroidBuffer contents];
                
                for (int k = 0; k < K; k++) {
                    if (countPtr[k] > 0) {
                        // CRITICAL FIX: oldCentroidPtr is in BGR order, not RGB!
                        float oldB = oldCentroidPtr[k * 3 + 0];  // B (blue channel)
                        float oldG = oldCentroidPtr[k * 3 + 1];  // G (green channel)
                        float oldR = oldCentroidPtr[k * 3 + 2];  // R (red channel)
                        
                        newCentroidPtr[k * 3 + 0] /= countPtr[k];
                        newCentroidPtr[k * 3 + 1] /= countPtr[k];
                        newCentroidPtr[k * 3 + 2] /= countPtr[k];
                        
                        // Check convergence (simple epsilon check) - match BGR order
                        float diffB = oldB - newCentroidPtr[k * 3 + 0];  // B channel diff
                        float diffG = oldG - newCentroidPtr[k * 3 + 1];  // G channel diff
                        float diffR = oldR - newCentroidPtr[k * 3 + 2];  // R channel diff
                        float distance = sqrt(diffR*diffR + diffG*diffG + diffB*diffB);
                        
                        if (distance > criteria.epsilon) {
                            converged = false;
                        }
                        
                        // Update centroid
                        oldCentroidPtr[k * 3 + 0] = newCentroidPtr[k * 3 + 0];
                        oldCentroidPtr[k * 3 + 1] = newCentroidPtr[k * 3 + 1];
                        oldCentroidPtr[k * 3 + 2] = newCentroidPtr[k * 3 + 2];
                        
                        // CRITICAL FIX: Prevent zero centroids which break GEMM distance calculation
                        float centroid_magnitude = sqrt(oldCentroidPtr[k * 3 + 0] * oldCentroidPtr[k * 3 + 0] +
                                                        oldCentroidPtr[k * 3 + 1] * oldCentroidPtr[k * 3 + 1] +
                                                        oldCentroidPtr[k * 3 + 2] * oldCentroidPtr[k * 3 + 2]);
                        if (centroid_magnitude < 1e-6f) {
                            oldCentroidPtr[k * 3 + 0] += 1e-6f;  // Was 1e-4f, now 1e-6f for minimal compactness impact
                            oldCentroidPtr[k * 3 + 1] += 1e-6f;
                            oldCentroidPtr[k * 3 + 2] += 1e-6f;
                        }
                    } else {
                        // CRITICAL FIX: Handle empty clusters by reinitializing them
                        // Use a simpler, more robust approach: random sample selection with distance bias
                        
                        // Count non-empty clusters for strategy selection
                        int nonEmptyClusters = 0;
                        for (int c = 0; c < K; c++) {
                            if (countPtr[c] > 0) {
                                nonEmptyClusters++;
                            }
                        }
                        
                        if (nonEmptyClusters > 0) {
                            // Strategy 1: Find sample farthest from existing non-empty centroids
                            float maxMinDistance = -1;
                            int bestSample = 0;
                            
                            for (int s = 0; s < numSamples; s++) {
                                float minDistToExistingCentroids = FLT_MAX;
                                
                                // Check distance to all existing (non-empty) centroids
                                for (int c = 0; c < K; c++) {
                                    if (c != k && countPtr[c] > 0) {  // Skip current empty cluster
                                        float dr = dataPtr[s * 3 + 0] - oldCentroidPtr[c * 3 + 0];
                                        float dg = dataPtr[s * 3 + 1] - oldCentroidPtr[c * 3 + 1];
                                        float db = dataPtr[s * 3 + 2] - oldCentroidPtr[c * 3 + 2];
                                        float dist = sqrt(dr*dr + dg*dg + db*db);
                                        if (dist < minDistToExistingCentroids) {
                                            minDistToExistingCentroids = dist;
                                        }
                                    }
                                }
                                
                                // Select sample with maximum distance to nearest existing centroid
                                if (minDistToExistingCentroids > maxMinDistance) {
                                    maxMinDistance = minDistToExistingCentroids;
                                    bestSample = s;
                                }
                            }
                            
                            // Reinitialize with the farthest sample
                            oldCentroidPtr[k * 3 + 0] = dataPtr[bestSample * 3 + 0]; // R
                            oldCentroidPtr[k * 3 + 1] = dataPtr[bestSample * 3 + 1]; // G
                            oldCentroidPtr[k * 3 + 2] = dataPtr[bestSample * 3 + 2]; // B
                            
                        } else {
                            // Strategy 2: All clusters empty - use distributed random sampling
                            // This handles the case where initial centroids are poorly placed
                            
                            // Use a deterministic but distributed approach based on cluster index
                            int sampleStride = numSamples / (K - nonEmptyClusters + 1);
                            int targetSample = (k * sampleStride + k) % numSamples;  // Add k for distribution
                            
                            oldCentroidPtr[k * 3 + 0] = dataPtr[targetSample * 3 + 0]; // R
                            oldCentroidPtr[k * 3 + 1] = dataPtr[targetSample * 3 + 1]; // G
                            oldCentroidPtr[k * 3 + 2] = dataPtr[targetSample * 3 + 2]; // B
                        }
                        
                        converged = false; // Force another iteration
                    }
                }
                
                // Early termination check
                if (converged || iter >= criteria.maxCount - 1) {
                    break;
                }
            }
        }
        
        // Compute compactness for this attempt
        std::vector<cv::Point3f> finalCentroids(K);
        float* finalCentroidPtr = (float*)[centroidBuffer contents];
        for (int i = 0; i < K; i++) {
            // FIXED: Keep BGR order consistent - no conversion needed
            finalCentroids[i].x = finalCentroidPtr[i * 3 + 0];  // BGR.x = B
            finalCentroids[i].y = finalCentroidPtr[i * 3 + 1];  // BGR.y = G
            finalCentroids[i].z = finalCentroidPtr[i * 3 + 2];  // BGR.z = R
        }
        
        double compactness = computeCompactnessGEMM(data, bestLabels, finalCentroids, *stream);
        if (compactness < bestCompactness) {
            bestCompactness = compactness;
            
            // Store compactness-only; centers upload handled later
        }
    }
    
    // Final commit and wait if we created our own stream
    if (useDefaultStream) {
        defaultStream.commitAndWait();
    } else {
        // Caller supplied stream – commit commands but leave synchronization to the caller
        stream->commit();
    }
    
    return bestCompactness;
}

// Internal implementation that takes command buffer
double kmeansImpl(const MetalMat& data, int K, MetalMat& bestLabels, 
                 TermCriteria criteria, int attempts, int flags, 
                 MetalMat& centers, Stream* stream = nullptr) {
    
    // Validate inputs
    CV_Assert(!data.empty());
    CV_Assert(K > 0 && K <= 1000); // Reasonable K limit
    CV_Assert(data.type() == CV_8UC3 || data.type() == CV_8UC4);
    CV_Assert(attempts > 0);
    
    // Create stream if none provided
    Stream defaultStream;
    bool useDefaultStream = (stream == nullptr);
    if (useDefaultStream) {
        stream = &defaultStream;
    }
    
    // Get Metal device and pipelines
    id<MTLDevice> device = MetalContext::getInstance().device;
    id<MTLComputePipelineState> assignPipeline = getAssignAndAccumulatePipeline();
    id<MTLComputePipelineState> updatePipeline = getUpdateCentroidsPipeline();
    
    if (!assignPipeline || !updatePipeline) {
        CV_Error(Error::StsError, "Failed to create Metal compute pipelines");
    }
    
    // Setup output matrices
    bestLabels.create(data.size(), CV_32S);
    int initCols = (CV_MAT_CN(data.type()) == 4) ? 4 : 3;
    centers.create(cv::Size(initCols, K), CV_32F);
    
    double bestCompactness = DBL_MAX;
    MetalMat bestLabelsTemp;
    std::vector<cv::Point3f> bestCentroids;
    
    // Create Metal buffers
    id<MTLBuffer> centroidsBuffer = [device newBufferWithLength:K * sizeof(float) * 3 
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> accumulatorsBuffer = [device newBufferWithLength:K * sizeof(int) * 4 
                                                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> kBuffer = [device newBufferWithBytes:&K length:sizeof(int) 
                                              options:MTLResourceStorageModeShared];
    
    if (!centroidsBuffer || !accumulatorsBuffer || !kBuffer) {
        CV_Error(Error::StsError, "Failed to create Metal buffers");
    }
    
    // Attempt loop
    for (int attempt = 0; attempt < attempts; attempt++) {
        std::vector<cv::Point3f> centroids;
        MetalMat labels;
        labels.create(data.size(), CV_32S);
        
        // Initialize centroids
        if (flags == KMEANS_RANDOM_CENTERS || (flags & (KMEANS_PP_CENTERS | KMEANS_USE_INITIAL_LABELS)) == 0) {
            initializeCentroidsRandom(data, centroids, K, *stream); // Pass stream
        } else {
            CV_Error(Error::StsNotImplemented, "Only KMEANS_RANDOM_CENTERS is currently supported");
        }
        
        // Copy initial centroids to Metal buffer (keep BGR order consistent)
        float* centroidsPtr = (float*)centroidsBuffer.contents;
        
        for (int i = 0; i < K; i++) {
            centroidsPtr[i * 3 + 0] = centroids[i].x;  // B = centroids BGR.x
            centroidsPtr[i * 3 + 1] = centroids[i].y;  // G = centroids BGR.y  
            centroidsPtr[i * 3 + 2] = centroids[i].z;  // R = centroids BGR.z
        }
        
        // Iteration loop
        int maxIterations = (criteria.type & TermCriteria::MAX_ITER) ? criteria.maxCount : 100;
        double epsilon = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : 1e-6;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            std::vector<cv::Point3f> oldCentroids = centroids;
            
            // Clear accumulators
            memset(accumulatorsBuffer.contents, 0, K * sizeof(int) * 4);
            
            // Encode assignment and accumulation kernel
            @autoreleasepool {
                id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(*stream);
                [encoder setComputePipelineState:assignPipeline];
                [encoder setTexture:data.texture() atIndex:0];
                [encoder setTexture:labels.texture() atIndex:1];
                [encoder setBuffer:centroidsBuffer offset:0 atIndex:0];
                [encoder setBuffer:accumulatorsBuffer offset:0 atIndex:1];
                [encoder setBuffer:kBuffer offset:0 atIndex:2];
                
                // Dispatch assignment kernel
                MTLSize gridSize = MTLSizeMake(
                    (data.cols() + 15) / 16, (data.rows() + 15) / 16, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }
            
            // Encode centroid update kernel
            @autoreleasepool {
                id<MTLComputeCommandEncoder> encoder = StreamAccessor::createComputeEncoder(*stream);
                [encoder setComputePipelineState:updatePipeline];
                [encoder setBuffer:centroidsBuffer offset:0 atIndex:0];
                [encoder setBuffer:accumulatorsBuffer offset:0 atIndex:1];
                [encoder setBuffer:kBuffer offset:0 atIndex:2];
                
                // Dispatch update kernel (exactly K threads)
                // Use dispatchThreadgroups instead of dispatchThreads for better handling of large K
                MTLSize threadsPerThreadgroup = MTLSizeMake(min(K, 64), 1, 1);
                MTLSize threadgroups = MTLSizeMake((K + 63) / 64, 1, 1);  // Ceiling division to cover all K
                [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
                [encoder endEncoding];
            }
            
            // Sync to read back updated centroids
            stream->syncCPU();
            
            // Copy updated centroids back (keep BGR order consistent)
            centroidsPtr = (float*)centroidsBuffer.contents;
            for (int i = 0; i < K; i++) {
                centroids[i].x = centroidsPtr[i * 3 + 0];  // BGR.x = B
                centroids[i].y = centroidsPtr[i * 3 + 1];  // BGR.y = G
                centroids[i].z = centroidsPtr[i * 3 + 2];  // BGR.z = R
            }
            
            // Check convergence
            if (criteria.type & TermCriteria::EPS) {
                if (checkConvergence(oldCentroids, centroids, epsilon)) {
                    break;
                }
            }
        }
        
        // Compute compactness for this attempt
        double compactness = computeCompactness(data, labels, centroids, *stream);
        
        // Keep the best result
        if (compactness < bestCompactness) {
            bestCompactness = compactness;
            bestLabelsTemp = labels;
            bestCentroids = centroids;
        }
    }
    
    // Copy best results to output
    bestLabels = bestLabelsTemp.clone();
    
    // Copy best centroids to centers Mat (match CPU k-means format: K rows × 3 or 4 columns)
    int centerCols = (CV_MAT_CN(data.type()) == 4) ? 4 : 3;
    Mat cpu_centers(K, centerCols, CV_32F);
    for (int i = 0; i < K; i++) {
        cpu_centers.at<float>(i, 0) = bestCentroids[i].x; // B
        cpu_centers.at<float>(i, 1) = bestCentroids[i].y; // G
        cpu_centers.at<float>(i, 2) = bestCentroids[i].z; // R
        if (centerCols == 4) {
            cpu_centers.at<float>(i, 3) = 1.0f; // dummy alpha
        }
    }
    // Create MetalMat centers with matching channel count (3 or 4 float channels)
    centers.create(cv::Size(centerCols, K), CV_32F);
    centers.upload(cpu_centers);
    
    // Final commit and wait if we created our own stream
    if (useDefaultStream) {
        defaultStream.commitAndWait();
    } else {
        // Ensure GPU commands are complete so caller can safely read data
        stream->syncCPU();
    }
    
    return bestCompactness;
}

} // anonymous namespace

namespace cv { namespace metal {

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