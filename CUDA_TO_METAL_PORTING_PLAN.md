# CUDA to Metal Porting Implementation Plan

## 🎯 **Overview**

This document provides a comprehensive plan for porting all CUDA functionality from `cudafilters` and `cudaimgproc` modules to Metal equivalents. The plan is organized to enable **parallel development** while managing **shared dependencies** and **helper functions**.

**Modules to Port:**
- **cudafilters** → **metalfilters** (14 filter types)
- **cudaimgproc** → **metalimgproc** (25+ advanced operations)

**Total Functions:** ~40 major algorithms + supporting infrastructure

---

## 📋 **Dependency Groups & Development Phases**

### **🏗️ Phase 0: Core Infrastructure (Prerequisites)**
**Team:** Infrastructure Team  
**Duration:** 2-3 weeks  
**Dependencies:** None

#### **Group 0A: Base Filter Infrastructure**
```cpp
// Shared infrastructure for all filters
class MetalFilter : public Algorithm {
    virtual void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
};

// Helper functions needed by multiple filter groups
namespace metal::internal {
    // Border handling utilities (shared by most filters)
    void handleBorderExtension(MetalMat& src, BorderTypes borderType, Scalar borderValue);
    
    // Kernel coefficient management (shared by linear filters)
    void uploadKernelCoefficients(const Mat& kernel, id<MTLBuffer>& coeffBuffer);
    
    // Multi-channel processing utilities
    void processChannelsSeparately(MetalMat& src, MetalMat& dst, std::function<void(int)> channelProcessor);
}
```

**Dependencies Used By:**
- Group 1A (Linear Filters) 
- Group 1B (Morphological Filters)
- Group 1C (Rank Filters)
- Group 2A-2E (All imgproc operations)

---

## 🔧 **Phase 1: Basic Filtering Operations (metalfilters)**

### **🔴 Group 1A: Linear & Separable Filters (Priority 1)**
**Team:** Linear Filters Team  
**Duration:** 3-4 weeks  
**Dependencies:** Group 0A

**Shared Helper Functions:**
```cpp
namespace metal::linear_filters {
    // Row/column filtering shared by separable operations
    void applyRowFilter(MetalMat& src, MetalMat& dst, const Mat& rowKernel, Stream& stream);
    void applyColumnFilter(MetalMat& src, MetalMat& dst, const Mat& colKernel, Stream& stream);
    
    // Generic convolution engine
    void convolve2D(MetalMat& src, MetalMat& dst, const Mat& kernel, Point anchor, Stream& stream);
}
```

**Functions to Implement:**
1. **createBoxFilter** 
   - Simple uniform kernel convolution
   - Maps well to MPS MPSImageBox or custom kernel
   
2. **createLinearFilter**
   - Generic 2D convolution with arbitrary kernel
   - Custom Metal kernel required
   
3. **createSeparableLinearFilter** ⚠️ **Dependencies: Row/Column Filters**
   - Depends on `applyRowFilter` + `applyColumnFilter`
   - High performance optimization opportunity
   
4. **createGaussianFilter** 
   - Can use MPS MPSImageGaussianBlur or custom separable implementation
   - Dependency: Separable filter infrastructure

**Metal Implementation Strategy:**
- **Box Filter**: MPS MPSImageBox 
- **Linear Filter**: Custom Metal compute shader
- **Separable Filters**: Two-pass custom Metal kernels
- **Gaussian**: MPS MPSImageGaussianBlur (optimized) or separable custom

---

### **🟡 Group 1B: Derivative Filters (Priority 2)**
**Team:** Derivative Team  
**Duration:** 2-3 weeks  
**Dependencies:** Group 1A (separable filter infrastructure)

**Shared Helper Functions:**
```cpp
namespace metal::derivative_filters {
    // Kernel generation utilities
    Mat getSobelKernel(int ksize, int dx, int dy);
    Mat getScharrKernel(int dx, int dy);
    Mat getDerivKernel(int ksize, int dx, int dy, bool normalize);
    
    // Derivative-specific optimization
    void applyDerivativeKernel(MetalMat& src, MetalMat& dst, const Mat& kernel, double scale, Stream& stream);
}
```

**Functions to Implement:**
1. **createDerivFilter**
   - Generic derivative filter generator
   - Base for Sobel and Scharr
   
2. **createSobelFilter** ⚠️ **Dependencies: DerivFilter**
   - Uses `getDerivKernel` → separable filtering
   - Can leverage existing metal::Sobel as reference
   
3. **createScharrFilter** ⚠️ **Dependencies: DerivFilter**
   - Uses `getScharrKernel` → separable filtering
   - Similar structure to Sobel
   
4. **createLaplacianFilter**
   - Second-order derivative
   - Can use MPS or custom kernel

**Metal Implementation Strategy:**
- **Sobel/Scharr**: Optimize existing implementation to use separable approach
- **Laplacian**: MPS MPSImageLaplacian or custom kernel
- **Generic Deriv**: Custom Metal kernel with coefficient buffers

---

### **🟢 Group 1C: Morphological Filters (Priority 3)**
**Team:** Morphology Team  
**Duration:** 2-3 weeks  
**Dependencies:** Group 0A only

**Shared Helper Functions:**
```cpp
namespace metal::morphology {
    // Structuring element management
    void uploadStructuringElement(const Mat& kernel, id<MTLBuffer>& seBuffer);
    
    // Basic morphological operations (building blocks)
    void erode(MetalMat& src, MetalMat& dst, const Mat& kernel, Point anchor, int iterations, Stream& stream);
    void dilate(MetalMat& src, MetalMat& dst, const Mat& kernel, Point anchor, int iterations, Stream& stream);
    
    // Compound operations
    void morphologyEx(MetalMat& src, MetalMat& dst, int op, const Mat& kernel, Point anchor, int iterations, Stream& stream);
}
```

**Functions to Implement:**
1. **createMorphologyFilter**
   - All morphological operations: ERODE, DILATE, OPEN, CLOSE, GRADIENT, TOPHAT, BLACKHAT
   - Can use MPS morphological operations where available

**Metal Implementation Strategy:**
- **Basic Ops**: MPS MPSImageDilate, MPSImageErode
- **Compound Ops**: Sequence of basic operations with intermediate buffers
- **Custom Kernels**: For non-standard structuring elements

---

### **🔵 Group 1D: Rank & Statistical Filters (Priority 4)**
**Team:** Statistical Team  
**Duration:** 2-3 weeks  
**Dependencies:** Group 0A only

**Functions to Implement:**
1. **createBoxMaxFilter**
   - Local maximum in rectangular window
   
2. **createBoxMinFilter** 
   - Local minimum in rectangular window
   
3. **createMedianFilter**
   - Median filtering with histogram-based approach
   
4. **createRowSumFilter** 
   - 1D horizontal summation
   
5. **createColumnSumFilter**
   - 1D vertical summation

**Metal Implementation Strategy:**
- **Min/Max**: Custom Metal kernels with local memory optimization
- **Median**: Histogram-based Metal compute shader (complex)
- **Sum Filters**: Simple 1D convolution kernels

---

## 🖼️ **Phase 2: Advanced Image Processing (metalimgproc)**

### **🟣 Group 2A: Color & Gamma Processing (Priority 1)**
**Team:** Color Team  
**Duration:** 2-3 weeks  
**Dependencies:** Group 0A only

**Functions to Implement:**
1. **cvtColor** 
   - Advanced color space conversions beyond basic RGB↔BGR
   - RGB↔HSV, RGB↔LAB, RGB↔YUV, etc.
   
2. **demosaicing**
   - Bayer pattern demosaicing (COLOR_BayerBG2BGR_MHT, etc.)
   
3. **gammaCorrection**
   - Forward and inverse gamma correction
   
4. **alphaComp**
   - Alpha compositing operations

**Metal Implementation Strategy:**
- **cvtColor**: Custom Metal kernels for each color space
- **Demosaicing**: Advanced interpolation Metal kernels
- **Gamma**: LUT-based Metal kernels
- **Alpha Comp**: Blending Metal kernels

---

### **🟠 Group 2B: Histogram Operations (Priority 2)**
**Team:** Histogram Team  
**Duration:** 3-4 weeks  
**Dependencies:** Group 0A + parallel reduction infrastructure

**Shared Helper Functions:**
```cpp
namespace metal::histogram {
    // Parallel reduction utilities
    void parallelReduce(MetalMat& src, id<MTLBuffer>& result, std::function<void()> reduceOp, Stream& stream);
    
    // Histogram calculation core
    void computeHistogram(MetalMat& src, MetalMat& hist, int numBins, float minVal, float maxVal, Stream& stream);
    
    // Histogram manipulation
    void equalizeHistogram(MetalMat& hist, MetalMat& lut, Stream& stream);
}
```

**Functions to Implement:**
1. **calcHist** 
   - Basic histogram calculation
   - With and without mask support
   
2. **equalizeHist**
   - Histogram equalization
   
3. **createCLAHE** 
   - Contrast Limited Adaptive Histogram Equalization
   - Complex algorithm requiring tile processing
   
4. **evenLevels**
   - Generate evenly distributed histogram levels
   
5. **histEven** / **histRange**
   - Histogram calculation with custom bins

**Metal Implementation Strategy:**
- **Basic Histogram**: Atomic operations in Metal compute shaders
- **CLAHE**: Multi-pass algorithm with tile-based processing
- **Equalization**: LUT generation + application

---

### **🔶 Group 2C: Edge & Feature Detection (Priority 2)**
**Team:** Feature Team  
**Duration:** 4-5 weeks  
**Dependencies:** Group 1B (derivative filters) + Group 2B (histogram for thresholding)

**Shared Helper Functions:**
```cpp
namespace metal::features {
    // Edge detection utilities
    void nonMaximumSuppression(MetalMat& gradient, MetalMat& direction, MetalMat& suppressed, Stream& stream);
    void hysteresisThresholding(MetalMat& edges, double lowThresh, double highThresh, Stream& stream);
    
    // Corner detection utilities  
    void computeCornerResponse(MetalMat& Ixx, MetalMat& Iyy, MetalMat& Ixy, MetalMat& response, double k, Stream& stream);
    void localMaxima(MetalMat& response, std::vector<Point2f>& corners, double threshold, Stream& stream);
}
```

**Functions to Implement:**
1. **createCannyEdgeDetector** ⚠️ **Dependencies: Derivative filters + NMS + Hysteresis**
   - Multi-stage algorithm: Gaussian blur → Sobel → NMS → Hysteresis
   - Depends on Group 1B derivative filters
   
2. **cornerHarris** (if available in cudaimgproc)
   - Harris corner detection
   - Depends on derivative calculations
   
3. **goodFeaturesToTrack** (if available)
   - Shi-Tomasi corner detection

**Metal Implementation Strategy:**
- **Canny**: Multi-pass algorithm with intermediate textures
- **Corner Detection**: Custom Metal kernels for response calculation
- **Feature Tracking**: Optimized Metal kernels with local memory

---

### **🟤 Group 2D: Geometric Analysis (Priority 3)**
**Team:** Geometry Team  
**Duration:** 3-4 weeks  
**Dependencies:** Group 2C (edge detection for Hough transforms)

**Shared Helper Functions:**
```cpp
namespace metal::geometry {
    // Hough transform utilities
    void houghTransformAccumulator(MetalMat& edges, MetalMat& accumulator, float rho, float theta, Stream& stream);
    void findPeaks(MetalMat& accumulator, std::vector<Vec2f>& lines, int threshold, Stream& stream);
    
    // Template matching utilities
    void normalizedCrossCorrelation(MetalMat& src, MetalMat& templ, MetalMat& result, Stream& stream);
}
```

**Functions to Implement:**
1. **createHoughLinesDetector** ⚠️ **Dependencies: Edge detection**
   - Requires edge detection from Group 2C
   
2. **createHoughSegmentDetector** ⚠️ **Dependencies: HoughLines**
   - Probabilistic Hough transform
   
3. **createHoughCirclesDetector**
   - Circle detection via Hough transform
   
4. **matchTemplate**
   - Template matching with various correlation methods

**Metal Implementation Strategy:**
- **Hough Transforms**: Accumulator arrays with atomic operations
- **Template Matching**: Sliding window correlation Metal kernels
- **Circle Detection**: Circular Hough transform with 3D accumulator

---

### **🟫 Group 2E: Advanced Processing (Priority 4)**
**Team:** Advanced Team  
**Duration:** 4-5 weeks  
**Dependencies:** Multiple groups (color, statistical, features)

**Functions to Implement:**
1. **meanShiftFiltering** / **meanShiftProc** / **meanShiftSegmentation**
   - Iterative mean shift algorithm
   - Complex multi-pass algorithm
   
2. **bilateralFilter** 
   - Edge-preserving smoothing
   - Computationally intensive
   
3. **moments** / **spatialMoments**
   - Image moment calculation
   - Requires parallel reduction
   
4. **connectedComponents**
   - Connected component labeling
   - Complex graph-based algorithm

**Metal Implementation Strategy:**
- **Mean Shift**: Iterative Metal kernels with convergence checking
- **Bilateral**: Custom Metal kernels with spatial-range weighting
- **Moments**: Parallel reduction Metal kernels
- **Connected Components**: Union-find algorithm in Metal

---

## ⏱️ **Development Timeline & Parallelization**

### **Phase 0 (Weeks 1-3): Infrastructure**
```
Week 1-2: Base filter infrastructure, border handling
Week 3:   Testing framework, Metal helper utilities
```

### **Phase 1 (Weeks 4-10): Basic Filtering**
```
Parallel Development:
├── Group 1A: Linear Filters (Weeks 4-7)
├── Group 1B: Derivative Filters (Weeks 5-7) 
├── Group 1C: Morphological (Weeks 6-8)
└── Group 1D: Rank/Statistical (Weeks 7-9)

Week 10: Integration & testing
```

### **Phase 2 (Weeks 11-20): Advanced Processing**
```
Parallel Development:
├── Group 2A: Color Processing (Weeks 11-13)
├── Group 2B: Histogram Operations (Weeks 11-14)
├── Group 2C: Edge/Feature Detection (Weeks 12-16)  
├── Group 2D: Geometric Analysis (Weeks 15-18)
└── Group 2E: Advanced Processing (Weeks 16-20)

Week 21: Final integration & optimization
```

---

## 🎯 **Priority & Risk Assessment**

### **High Priority + Low Risk**
- ✅ Group 1A: Linear Filters (well-understood, MPS support)
- ✅ Group 1C: Morphological (MPS support available)
- ✅ Group 2A: Color Processing (straightforward kernels)

### **High Priority + Medium Risk**
- ⚠️ Group 1B: Derivative Filters (optimization complexity)
- ⚠️ Group 2B: Histogram Operations (parallel reduction challenges)
- ⚠️ Group 2C: Edge Detection (multi-stage algorithms)

### **Medium Priority + High Risk**
- 🔺 Group 1D: Rank Filters (median filter complexity)
- 🔺 Group 2D: Geometric Analysis (Hough transform complexity)
- 🔺 Group 2E: Advanced Processing (algorithm complexity)

---

## 📊 **Resource Allocation**

### **Team Structure (6 teams, 2-3 developers each)**
```
Infrastructure Team:  2 senior developers (Metal experts)
Linear Filters Team:  3 developers (1 senior, 2 mid)
Derivative Team:      2 developers (1 senior, 1 mid)  
Morphology Team:      2 developers (1 senior, 1 mid)
Statistical Team:     2 developers (1 senior, 1 mid)
Color Team:           2 developers (1 senior, 1 mid)
Histogram Team:       3 developers (1 senior, 2 mid)
Feature Team:         3 developers (2 senior, 1 mid) 
Geometry Team:        3 developers (2 senior, 1 mid)
Advanced Team:        3 developers (2 senior, 1 mid)
```

### **Testing Requirements**
Each group must provide:
- ✅ Unit tests comparing CPU vs Metal results (appropriate tolerances)
- ✅ Performance benchmarks vs CUDA equivalents  
- ✅ Edge case testing (various image sizes, formats)
- ✅ Memory leak testing
- ✅ Cross-platform testing (Intel + Apple Silicon)

---

## 🔄 **Development Workflow**

### **Parallel Development Process**
1. **Phase Start**: Infrastructure team completes prerequisites
2. **Parallel Implementation**: Multiple teams work independently on their groups
3. **Integration Points**: Weekly syncs to coordinate shared dependencies
4. **Testing Gates**: Each group must pass testing before integration
5. **Performance Validation**: Benchmarking against CUDA equivalents

### **Dependency Management**
```
Dependencies are carefully managed through:
├── Shared header files with forward declarations
├── Interface-based design (MetalFilter base class)
├── Helper function libraries (metal::internal namespace)
├── Build system coordination (CMake dependencies)
└── Integration testing at dependency boundaries
```

### **Success Metrics**
- ✅ **Functionality**: 100% API compatibility with CUDA equivalents
- ✅ **Performance**: 1.5x+ speedup vs CPU, competitive with CUDA  
- ✅ **Quality**: <2% test failure rate, zero memory leaks
- ✅ **Compatibility**: Works on all Apple Metal-capable devices

---

## 🎊 **Expected Outcomes**

Upon completion, this plan will deliver:

### **metalfilters Module**
- ✅ **14 filter types** with full CUDA API compatibility
- ✅ **High-performance Metal implementations** leveraging MPS where possible
- ✅ **Optimized separable filtering** infrastructure for maximum performance
- ✅ **Comprehensive testing suite** ensuring quality and compatibility

### **metalimgproc Module** 
- ✅ **25+ advanced algorithms** covering full CUDA imgproc functionality
- ✅ **Complex multi-stage algorithms** (Canny, CLAHE, Hough transforms)
- ✅ **Advanced image analysis** capabilities (moments, connected components)
- ✅ **Production-ready performance** competitive with CUDA implementations

### **Strategic Value**
- 🚀 **Complete Metal ecosystem** for computer vision on Apple platforms
- 🚀 **Performance leadership** in Apple GPU acceleration  
- 🚀 **Contributor adoption** through proven CUDA compatibility
- 🚀 **Future scalability** with extensible architecture for new algorithms

**Total Development Effort:** ~21 weeks with parallel teams  
**Expected Performance:** 1.5-3x speedup vs CPU, competitive with CUDA  
**API Compatibility:** 100% drop-in replacement for CUDA workflows

---

*This plan leverages the successful patterns established in our current Metal backend implementation, ensuring both quality and performance while enabling maximum parallel development efficiency.* 