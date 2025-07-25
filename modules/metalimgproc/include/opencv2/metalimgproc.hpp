#ifndef OPENCV_METALIMGPROC_HPP
#define OPENCV_METALIMGPROC_HPP

#include <opencv2/core/metal.hpp>
#include <opencv2/imgproc.hpp>

namespace cv { namespace metal {

//! @addtogroup metalimgproc
//! @{

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
@param stream Stream for asynchronous execution.

@note The Metal implementation follows the same algorithm as the CPU version but accelerates the
Gaussian Mixture Model (GMM) computations using Metal Performance Shaders and custom kernels.
The graph-cut portion remains on the CPU for maximum compatibility with existing OpenCV infrastructure.

@sa cv::grabCut
*/
CV_EXPORTS void grabCut(InputArray img, InputOutputArray mask, Rect rect,
                        InputOutputArray bgdModel, InputOutputArray fgdModel,
                        int iterCount, int mode = GC_EVAL);

CV_EXPORTS void grabCut(InputArray img, InputOutputArray mask, Rect rect,
                        InputOutputArray bgdModel, InputOutputArray fgdModel,
                        int iterCount, int mode, Stream& stream);

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
@param stream Stream for asynchronous execution.

@sa cv::grabCutWithSharedKMeans
*/
CV_EXPORTS void grabCutWithSharedKMeans(InputArray img, InputOutputArray mask, Rect rect,
                                       InputOutputArray bgdModel, InputOutputArray fgdModel,
                                       int iterCount, int mode, uint64_t randomSeed, Stream& stream);

CV_EXPORTS void grabCutWithSharedKMeans(InputArray img, InputOutputArray mask, Rect rect,
                                       InputOutputArray bgdModel, InputOutputArray fgdModel,
                                       int iterCount, int mode = GC_EVAL, uint64_t randomSeed = 42);

//! @}
}} // cv::metal

#endif // OPENCV_METALIMGPROC_HPP 