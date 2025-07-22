# Metal Porting Guide for OpenCV CUDA Modules

## 1. Introduction

This document outlines the plan and framework for porting CUDA-based algorithms from the `cudafilters_reference` and `cudaimgproc_reference` modules to the Metal backend. The goal is to create a comprehensive set of hardware-accelerated functions for Apple platforms, enabling multiple engineers to work in parallel.

All new implementations should adhere to the principles and practices established in the `METAL_IMPLEMENTATION_GUIDE.md`. This includes the use of `cv::metal::MetalMat`, `cv::metal::Stream`, and the overall API design.

## 2. Proposed Framework and API Design

New Metal functions should be placed in the `cv::metal` namespace and exposed through new headers, analogous to the CUDA module structure.

### 2.1. New Modules and Headers

-   **`modules/metalfilters`**: A new module for filter objects.
    -   Public Header: `modules/metalfilters/include/opencv2/metalfilters.hpp`
    -   Implementation: `modules/metalfilters/src/metal/`
-   **`modules/imgproc`**: Standalone image processing functions will be added to the existing `imgproc` Metal integration.
    -   Public Header: `modules/imgproc/include/opencv2/imgproc/metal.hpp`
    -   Implementation: `modules/imgproc/src/metal/`

### 2.2. API Patterns

We will follow three main API patterns, consistent with existing OpenCV modules:

#### A. Filter Objects

For stateful or complex filters, a `cv::metal::Filter` base class will be used, with factory functions for creation.

**Base Class (`opencv2/metal.hpp` or `opencv2/metalfilters.hpp`):**
```cpp
namespace cv { namespace metal {
class CV_EXPORTS_W Filter : public cv::Algorithm
{
public:
    CV_WRAP virtual void apply(InputArray src, OutputArray dst, Stream& stream = Stream::Null()) = 0;
};
}} // cv::metal
```

**Factory Function Example (`opencv2/metalfilters.hpp`):**
```cpp
CV_EXPORTS_W Ptr<Filter> createBoxFilter(int srcType, int dstType, Size ksize, ...);
```

#### B. Standalone Functions

For common, stateless operations, standalone functions are preferred. This matches the existing Metal implementations for `GaussianBlur`, `Sobel`, and `resize`.

**Function Example (`opencv2/imgproc/metal.hpp`):**
```cpp
CV_EXPORTS_W void bilateralFilter(InputArray src, OutputArray dst, int kernel_size, ...);
```

#### C. Algorithm Classes

For complex algorithms with multiple steps or significant state (like Canny), a dedicated class inheriting from `cv::Algorithm` is appropriate.

**Class Example (`opencv2/imgproc/metal.hpp`):**
```cpp
class CV_EXPORTS_W CannyEdgeDetector : public cv::Algorithm { ... };
CV_EXPORTS_W Ptr<CannyEdgeDetector> createCannyEdgeDetector(double low_thresh, ...);
```

## 3. Shared Infrastructure and Dependencies

Several CUDA kernels are shared across multiple algorithms. Porting these components first is a priority as they are dependencies for other tasks.

| Component | CUDA Reference | Description | Metal Porting Priority |
| :--- | :--- | :--- | :--- |
| **Separable Convolution** | `row_filter.hpp`, `column_filter.hpp` | Core of many filters (Sobel, Gaussian, Scharr). | **High** |
| **Point List Generation** | `build_point_list.cu` | Generates a list of non-zero pixel coordinates from a binary image. Used by Hough transforms. | **High** |
| **Histogram Atomics** | `hist.cu` | Atomic operations for building histograms. | **Medium** |

## 4. Porting Tasks from `cudafilters_reference`

The following filters need to be ported. Many can be implemented using Metal Performance Shaders (MPS) for optimal performance. Others will require custom Metal Shading Language (MSL) kernels.

| Algorithm | Proposed Metal API | CUDA Reference Files | Dependencies | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Box Filter** | `Ptr<Filter> createBoxFilter(...)` | `filtering.cpp` (uses NPP) | None | Use `MPSImageBox`. |
| **Linear Filter (2D)** | `Ptr<Filter> createLinearFilter(...)` | `src/cuda/filter2d.cu` | None | Use `MPSImageConvolution`. |
| **Separable Filter** | `Ptr<Filter> createSeparableLinearFilter(...)` | `src/cuda/row_filter.hpp`, `src/cuda/column_filter.hpp` | Separable Conv. | High priority. Use `MPSImageSeparableConvolution`. |
| **Gaussian Filter** | `Ptr<Filter> createGaussianFilter(...)` | `filtering.cpp` (wraps separable) | Separable Filter | Use `MPSImageGaussianBlur`. |
| **Sobel/Scharr/Deriv** | `Ptr<Filter> create...Filter(...)` | `filtering.cpp` (wraps separable) | Separable Filter | Use `MPSImageSobel` or `MPSImageConvolution`. |
| **Morphology** | `Ptr<Filter> createMorphologyFilter(...)` | `filtering.cpp` (uses NPP) | None | Use `MPSImageAreaMax` (dilate), `MPSImageAreaMin` (erode), etc. |
| **Median Filter** | `Ptr<Filter> createMedianFilter(...)` | `src/cuda/median_filter.cu`, `wavelet_matrix_*.cuh` | None | Complex. Use `MPSImageMedian`. |

## 5. Porting Tasks from `cudaimgproc_reference`

This module contains a wider variety of image processing algorithms.

| Algorithm | Proposed Metal API | CUDA Reference Files | Dependencies | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Bilateral Filter** | `bilateralFilter(...)` | `src/cuda/bilateral_filter.cu` | None | Requires custom MSL kernel. MPS does not have a direct equivalent. |
| **Canny Detector** | `Ptr<CannyEdgeDetector> create...()` | `src/cuda/canny.cu` | Sobel Filter | Multi-stage algorithm. Port Sobel first, then implement non-maximum suppression and hysteresis. |
| **Hough Lines** | `Ptr<HoughLinesDetector> create...()` | `src/cuda/hough_lines.cu` | Point List Gen. | Requires accumulator with atomics. |
| **Hough Circles** | `Ptr<HoughCirclesDetector> create...()` | `src/cuda/hough_circles.cu` | Point List Gen. | Similar to Hough Lines. |
| **Template Matching** | `Ptr<TemplateMatching> create...()` | `src/cuda/match_template.cu` | Linear Filter (for CCORR) | Can be implemented with convolutions. `MPSImageConvolution` is a good starting point. |
| **Corner Detection** | `Ptr<CornernessCriteria> create...()` | `src/cuda/corners.cu` | Sobel Filter | Harris and MinEigenVal criteria depend on derivatives. |
| **Good Features to Track** | `Ptr<CornersDetector> create...()` | `src/cuda/gftt.cu` | Corner Detection | Builds upon cornerness criteria. |
| **Histogram** | `calcHist(...)`, `equalizeHist(...)` | `src/cuda/hist.cu` | Histogram Atomics | `MPSImageHistogram` can be used. `equalizeHist` uses `MPSImageHistogramEqualization`. |
| **CLAHE** | `Ptr<CLAHE> createCLAHE(...)` | `src/cuda/clahe.cu` | Histogram | Complex algorithm. Requires custom kernel for LUT generation and interpolation. |
| **Color Conversion** | `cvtColor(...)` | `src/cuda/color.cu` | None | Many conversions can be done with custom, simple MSL kernels. Some might be available in MPS. |
| **Demosaicing** | `demosaicing(...)` | `src/cuda/debayer.cu` | None | Requires custom MSL kernel. |

## 6. Development Workflow

1.  **Pick a Task**: Choose an algorithm from the lists above, preferably one whose dependencies are met.
2.  **Create New Files**: Add files to the appropriate module (`metalfilters` or `imgproc`).
3.  **Implement API**: Create the C++ function/class in Objective-C++ (`.mm` file).
4.  **Write Kernel**: Implement the core logic using MPS or a custom MSL kernel.
5.  **Add Tests**:
    -   Add a correctness test in `modules/<module>/test/test_metal.cpp` comparing against the CPU version.
    -   Add a performance test in `modules/<module>/perf/perf_metal.cpp`.
6.  **Update CMake**: Add new files to `CMakeLists.txt` in the module directory.
7.  **Submit Pull Request**: Ensure all tests pass.

This structured approach allows for clear task division and parallel development, accelerating the completion of the Metal backend.