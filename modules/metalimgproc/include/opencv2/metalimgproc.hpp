#ifndef OPENCV_METALIMGPROC_HPP
#define OPENCV_METALIMGPROC_HPP

#include <opencv2/core/metal.hpp>
#include <opencv2/imgproc.hpp>

namespace cv { namespace metal {

//! @addtogroup metalimgproc
//! @{

/** @brief Performs K-means clustering using Metal acceleration.

This function implements K-means clustering algorithm with Metal GPU acceleration, providing 
significant performance improvements over CPU implementation for large datasets.

@param data Input data for clustering. Each row represents a sample point. For images, this
is typically a N×3 matrix where N is the number of pixels and each row contains RGB/BGR values.
@param K Number of clusters to create.
@param bestLabels Output cluster labels for each input sample. Will be resized to match input data.
@param criteria Termination criteria combining maximum iterations and convergence threshold.
@param attempts Number of clustering attempts (algorithm will return best result).
@param flags Algorithm initialization method (currently supports KMEANS_RANDOM_CENTERS).
@param centers Output cluster centers. Will be K×3 matrix for RGB data.
@param stream Stream for asynchronous execution.

@note The Metal implementation follows the same algorithm as CPU cv::kmeans but accelerates
computations using Metal compute shaders. Results should be equivalent within numerical precision.

@sa cv::kmeans
*/
CV_EXPORTS double kmeans(const MetalMat& data, int K, MetalMat& bestLabels,
                        TermCriteria criteria, int attempts, int flags, MetalMat& centers, Stream& stream);

CV_EXPORTS double kmeans(const MetalMat& data, int K, MetalMat& bestLabels,
                        TermCriteria criteria, int attempts, int flags, MetalMat& centers);

/** @brief Runs the GrabCut algorithm using Metal acceleration.

The function implements the [GrabCut image segmentation algorithm](https://en.wikipedia.org/wiki/GrabCut)
with Metal GPU acceleration. This provides significant performance improvements over the CPU implementation
for iterative segmentation workflows.

@param img Input 8-bit 3-channel image (converted to 4-channel BGRA internally for Metal processing).
@param mask Input/output 8-bit single-channel mask. The mask is initialized by the function when
mode is set to #GC_INIT_WITH_RECT. Its elements may have one of the #GrabCutClasses.
@param rect ROI containing a segmented object. The pixels outside of the ROI are marked as
"obvious background". The parameter is only used when mode==#GC_INIT_WITH_RECT.
@param bgdModel Temporary array for the background model. Do not modify it while you are
processing the same image.
@param fgdModel Temporary array for the foreground model. Do not modify it while you are
processing the same image.
@param iterCount Number of iterations the algorithm should make before returning the result. Note
that the result can be refined with further calls with mode==#GC_INIT_WITH_MASK or
mode==GC_EVAL.
@param mode Operation mode that could be one of the #GrabCutModes.
@param useGpuGraphCut If true, uses Metal GPU graph cut solver; if false, uses CPU graph cut solver.
Default is true for best performance on supported hardware.
@param stream Stream for asynchronous execution.

@note The Metal implementation follows the same algorithm as the CPU version but accelerates the
Gaussian Mixture Model (GMM) computations using Metal Performance Shaders and custom kernels.
When useGpuGraphCut=true, the entire pipeline runs on GPU. When false, GMM runs on GPU but 
graph-cut runs on CPU for maximum compatibility.

@sa cv::grabCut
*/
CV_EXPORTS void grabCut(InputArray img, InputOutputArray mask, Rect rect,
                        InputOutputArray bgdModel, InputOutputArray fgdModel,
                        int iterCount, int mode = GC_EVAL, bool useGpuGraphCut = true);

CV_EXPORTS void grabCut(InputArray img, InputOutputArray mask, Rect rect,
                        InputOutputArray bgdModel, InputOutputArray fgdModel,
                        int iterCount, int mode, bool useGpuGraphCut, Stream& stream);

/** @brief Runs the GrabCut algorithm with shared deterministic K-means initialization.

This function uses deterministic K-means clustering for consistent results between CPU and Metal 
implementations. It provides identical segmentation results when both CPU and Metal use the same 
random seed.

@param img Input 8-bit 3-channel image (converted to 4-channel BGRA internally for Metal processing).
@param mask Input/output 8-bit single-channel mask. The mask is initialized by the function when
mode is set to #GC_INIT_WITH_RECT. Its elements may have one of the #GrabCutClasses.
@param rect ROI containing a segmented object. The pixels outside of the ROI are marked as
"obvious background". The parameter is only used when mode==#GC_INIT_WITH_RECT.
@param bgdModel Temporary array for the background model. Do not modify it while you are
processing the same image.
@param fgdModel Temporary array for the foreground model. Do not modify it while you are
processing the same image.
@param iterCount Number of iterations the algorithm should make before returning the result. Note
that the result can be refined with further calls with mode==#GC_INIT_WITH_MASK or
mode==GC_EVAL.
@param mode Operation mode that could be one of the #GrabCutModes.
@param randomSeed Fixed random seed for deterministic K-means initialization (default: 42).
@param useGpuGraphCut If true, uses Metal GPU graph cut solver; if false, uses CPU graph cut solver.
Default is true for best performance on supported hardware.
@param stream Stream for asynchronous execution.

@sa cv::grabCutWithSharedKMeans
*/
CV_EXPORTS void grabCutWithSharedKMeans(InputArray img, InputOutputArray mask, Rect rect,
                                       InputOutputArray bgdModel, InputOutputArray fgdModel,
                                       int iterCount, int mode, uint64_t randomSeed, 
                                       bool useGpuGraphCut, Stream& stream);

CV_EXPORTS void grabCutWithSharedKMeans(InputArray img, InputOutputArray mask, Rect rect,
                                       InputOutputArray bgdModel, InputOutputArray fgdModel,
                                       int iterCount, int mode = GC_EVAL, uint64_t randomSeed = 42,
                                       bool useGpuGraphCut = true);

/** @brief Internal mask-based K-means clustering for GrabCut initialization.
@note This is an internal function not intended for public use.
*/
void kmeansClusterByMask(const MetalMat& inImg, const MetalMat& mask, bool useBackground,
                        MetalMat& outLabels, cv::Mat& centroids, Stream& stream);

/** @brief GPU Component Assignment for GrabCut optimization.

This function replaces CPU pixel-by-pixel loops with GPU parallel processing for assigning
k-means cluster labels to component indices. This is part of Phase 1.3 of the GrabCut
performance optimization plan.

@param mask Input GrabCut mask (GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD values).
@param bgLabels Background k-means cluster labels from Metal k-means.
@param fgLabels Foreground k-means cluster labels from Metal k-means.
@param compIdxs Output component assignment indices (0-4 for both BG and FG).
@param stream Stream for asynchronous execution.

@note This eliminates data transfer overhead (~80MB per iteration) and CPU processing loops,
providing 2-3x speedup over the original CPU implementation.
*/
void assignComponents(const MetalMat& mask, const MetalMat& bgLabels, const MetalMat& fgLabels, 
                     MetalMat& compIdxs, Stream& stream);

//! @}
}} // cv::metal

#endif // OPENCV_METALIMGPROC_HPP 