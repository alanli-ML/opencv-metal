#ifndef OPENCV_METALIMGPROC_GMM_KERNELS_HPP
#define OPENCV_METALIMGPROC_GMM_KERNELS_HPP

#include "precomp.hpp"

#ifdef HAVE_METAL

namespace cv { namespace metal {

// Initialize GMM pipelines
void initializeGMMPipelines();

// Pipeline accessors
id<MTLComputePipelineState> getGMMInitializePipeline();
id<MTLComputePipelineState> getGMMAssignPipeline();
id<MTLComputePipelineState> getGMMDataTermPipeline();
id<MTLComputePipelineState> getGMMReductionPipeline();
id<MTLComputePipelineState> getGMMAccumulateStatsPipeline();
id<MTLComputePipelineState> getGMMCountPixelsPipeline();
id<MTLComputePipelineState> getGMMFinalizeParametersPipeline();

}} // cv::metal

#endif // HAVE_METAL

#endif // OPENCV_METALIMGPROC_GMM_KERNELS_HPP 