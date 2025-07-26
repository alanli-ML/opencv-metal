# Metal GrabCut Performance Optimization Plan


###use this test for performance evaluation:

./bin/opencv_test_metalimgproc --gtest_filter="*RealImagePerformanceTest*"


## Current Performance Issues
- Metal efficiency: Only 1.08x vs CPU (should be 2-5x)
- Data transfer overhead: ~80MB per iteration
- Multiple GPU pipeline stalls from sync points
- Hybrid CPU/GPU architecture instead of GPU-first

## Phase 1: Immediate Fixes (2-3x speedup target)

### 3. GPU Component Assignment
Current: Downloads k-means labels then CPU processes every pixel

```cpp
// BEFORE: Download + CPU processing
bgLabels.download(h_bgLabels, *stream, true);  // Download
fgLabels.download(h_fgLabels, *stream, true);  // Download

for (p.y = 0; p.y < img.rows; p.y++) {        // CPU loop
    for (p.x = 0; p.x < img.cols; p.x++) {    // Every pixel
        uchar maskVal = mask.at<uchar>(p);
        if (maskVal == cv::GC_BGD || maskVal == cv::GC_PR_BGD) {
            int label = h_bgLabels.at<int>(p.y, p.x);
            compIdxs.at<uchar>(p) = (uchar)label;
        }
        // ... more CPU processing
    }
}

// AFTER: GPU kernel for component assignment
// New Metal kernel: assignComponentsKernel
kernel void assignComponentsKernel(
    texture2d<float, access::read> mask [[texture(0)]],
    texture2d<int, access::read> bgLabels [[texture(1)]],
    texture2d<int, access::read> fgLabels [[texture(2)]],
    texture2d<uchar, access::write> compIdxs [[texture(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= mask.get_width() || gid.y >= mask.get_height()) return;
    
    float maskVal = mask.read(gid).r;
    int component = 0;
    
    if (maskVal <= 0.33f || (maskVal >= 0.5f && maskVal <= 0.83f)) { // BGD or PR_BGD
        component = bgLabels.read(gid).r;
    } else { // FGD or PR_FGD
        component = fgLabels.read(gid).r;
    }
    
    compIdxs.write(component, gid);
}

// Usage: No downloads, GPU-only processing
cv::metal::assignComponents(mask, bgLabels, fgLabels, compIdxs, stream);
// No CPU processing required
```

## Phase 2: Medium-term Optimizations (3-5x speedup target)

### 4. GPU Graph Construction
Move graph building to GPU to eliminate massive data transfers

```cpp
// NEW: GPU-based graph construction
struct GPUGraphData {
    MetalMat nodeCapacities;  // Unary terms
    MetalMat edgeWeights;     // Pairwise terms  
    MetalMat edgeIndices;     // Graph topology
};

class MetalGraphBuilder {
public:
    void buildGraph(const MetalMat& bgTerm, const MetalMat& fgTerm,
                   const MetalMat& leftW, const MetalMat& topW, 
                   const MetalMat& rightW, const MetalMat& bottomW,
                   GPUGraphData& graphData, Stream& stream);
    
    // Only download compressed graph structure, not full matrices
    void downloadForCPUMaxFlow(cv::detail::GCGraph<double>& graph, Stream& stream);
};

// Metal kernel for graph construction
kernel void buildGraphKernel(
    texture2d<float, access::read> bgTerm [[texture(0)]],
    texture2d<float, access::read> fgTerm [[texture(1)]],
    texture2d<float4, access::read> pairwiseWeights [[texture(2)]],
    device float* nodeCapacities [[buffer(0)]],
    device float* edgeWeights [[buffer(1)]],
    device uint2* edgeIndices [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Build graph structure on GPU
    // Significantly reduce data transfer requirements
}
```

### 5. Persistent GPU Buffers
Reuse allocated GPU memory across iterations

```cpp
class GrabCutGPUBuffers {
private:
    MetalMat m_bgTerm, m_fgTerm;
    MetalMat m_pairwiseWeights[4];
    MetalMat m_components, m_prevMask;
    MetalMat m_graphNodes, m_graphEdges;
    
public:
    void allocateForSize(cv::Size imageSize) {
        // Allocate once, reuse across all iterations
        if (m_bgTerm.empty() || m_bgTerm.size() != imageSize) {
            m_bgTerm.create(imageSize, CV_32FC1);
            m_fgTerm.create(imageSize, CV_32FC1);
            // ... allocate other buffers
        }
    }
    
    void reuseBuffers(MetalMat& bgTerm, MetalMat& fgTerm, ...) {
        // Return references to pre-allocated buffers
        bgTerm = m_bgTerm;
        fgTerm = m_fgTerm;
        // ...
    }
};
```

## Phase 3: Full GPU Pipeline (5-10x speedup target)

### 6. Complete Metal Graph Cut
Finish the USE_METAL_GRAPHCUT=1 implementation

```cpp
#define USE_METAL_GRAPHCUT 1  // Enable when implementation is complete

class MetalMaxFlow {
public:
    void solve(const GPUGraphData& graph, MetalMat& result, Stream& stream);
    
private:
    void pushRelabelGPU(const GPUGraphData& graph, Stream& stream);
    void updateLabels(Stream& stream);
    void pushExcess(Stream& stream);
};

// Metal kernels for max-flow algorithm
kernel void pushExcessKernel(...) { /* GPU push-relabel */ }
kernel void updateLabelsKernel(...) { /* GPU label updates */ }
```

## Implementation Timeline

### Week 1-2: Quick Wins
1. Implement batched downloads (Phase 1.1)
2. Remove sync points (Phase 1.2)  
3. Add async GMM extraction (Phase 1.4)

**Expected: 2-2.5x speedup**

### Week 3-4: GPU Component Assignment  
4. Implement assignComponentsKernel (Phase 1.3)
5. Add GPU-based k-means result processing

**Expected: 2.5-3x speedup**

### Week 5-8: GPU Graph Construction
6. Implement GPU graph building kernels (Phase 2.1)
7. Add persistent buffer management (Phase 2.2)

**Expected: 3-4x speedup**

### Long-term: Full GPU Pipeline
8. Complete Metal max-flow implementation (Phase 3)
9. End-to-end GPU memory management

**Expected: 5-10x speedup**

## Performance Validation

```cpp
// Add benchmarking to validate improvements
TEST(MetalImgproc_GrabCut, PerformanceValidation) {
    // Measure each optimization phase
    // Target: >2x speedup for Phase 1, >3x for Phase 2, >5x for Phase 3
    EXPECT_GT(metal_speedup_phase1, 2.0);
    EXPECT_GT(metal_speedup_phase2, 3.0); 
    EXPECT_GT(metal_speedup_phase3, 5.0);
}
```

## Risk Mitigation

1. **Backward Compatibility**: Keep CPU fallbacks during transition
2. **Incremental Testing**: Validate each phase separately  
3. **Performance Regression**: Continuous benchmarking
4. **Memory Management**: Careful GPU buffer lifecycle management 