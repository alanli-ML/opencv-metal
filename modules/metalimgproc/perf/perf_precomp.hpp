#ifndef OPENCV_METALIMGPROC_PERF_PRECOMP_HPP
#define OPENCV_METALIMGPROC_PERF_PRECOMP_HPP

#include "opencv2/ts.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/metal.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/metalimgproc.hpp"

#ifdef HAVE_METAL

namespace cv { namespace metal {

// Forward declarations for testing internal functions
double calcBeta(const MetalMat& image);
void calcNWeights(const MetalMat& image, MetalMat& leftW, MetalMat& topleftW, MetalMat& topW, MetalMat& toprightW, 
                  double beta, double gamma, Stream& stream);

}} // cv::metal

#endif // HAVE_METAL

#endif // OPENCV_METALIMGPROC_PERF_PRECOMP_HPP 