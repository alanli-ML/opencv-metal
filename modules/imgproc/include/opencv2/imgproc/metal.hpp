// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGPROC_METAL_HPP
#define OPENCV_IMGPROC_METAL_HPP

#include <opencv2/core/metal.hpp>
#include <opencv2/imgproc.hpp>

namespace cv { namespace metal {

//! @addtogroup imgproc_metal
//! @{

CV_EXPORTS void GaussianBlur(const MetalMat& src, MetalMat& dst, Size ksize, double sigmaX);
CV_EXPORTS void GaussianBlur(const MetalMat& src, MetalMat& dst, Size ksize, double sigmaX, Stream& stream);

CV_EXPORTS void Sobel(const MetalMat& src, MetalMat& dst, int ddepth, int dx, int dy, int ksize = 3);
CV_EXPORTS void Sobel(const MetalMat& src, MetalMat& dst, int ddepth, int dx, int dy, int ksize, Stream& stream);

CV_EXPORTS void resize(const MetalMat& src, MetalMat& dst, Size dsize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR);
CV_EXPORTS void resize(const MetalMat& src, MetalMat& dst, Size dsize, double fx, double fy, int interpolation, Stream& stream);

//! @}
}} // cv::metal

#endif // OPENCV_IMGPROC_METAL_HPP