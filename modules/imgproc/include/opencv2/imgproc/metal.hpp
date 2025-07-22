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

CV_EXPORTS void bilateralFilter(const MetalMat& src, MetalMat& dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode = BORDER_DEFAULT);
CV_EXPORTS void bilateralFilter(const MetalMat& src, MetalMat& dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode, Stream& stream);

CV_EXPORTS void boxFilter(const MetalMat& src, MetalMat& dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), bool normalize = true, int borderType = BORDER_DEFAULT);
CV_EXPORTS void boxFilter(const MetalMat& src, MetalMat& dst, int ddepth, Size ksize, Point anchor, bool normalize, int borderType, Stream& stream);

CV_EXPORTS void filter2D(const MetalMat& src, MetalMat& dst, int ddepth, InputArray kernel, Point anchor = Point(-1,-1), double delta = 0, int borderType = BORDER_DEFAULT);
CV_EXPORTS void filter2D(const MetalMat& src, MetalMat& dst, int ddepth, InputArray kernel, Point anchor, double delta, int borderType, Stream& stream);

CV_EXPORTS void erode(const MetalMat& src, MetalMat& dst, InputArray kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, const Scalar& borderValue = morphologyDefaultBorderValue());
CV_EXPORTS void erode(const MetalMat& src, MetalMat& dst, InputArray kernel, Point anchor, int iterations, int borderType, const Scalar& borderValue, Stream& stream);

CV_EXPORTS void dilate(const MetalMat& src, MetalMat& dst, InputArray kernel, Point anchor = Point(-1,-1), int iterations = 1, int borderType = BORDER_CONSTANT, const Scalar& borderValue = morphologyDefaultBorderValue());
CV_EXPORTS void dilate(const MetalMat& src, MetalMat& dst, InputArray kernel, Point anchor, int iterations, int borderType, const Scalar& borderValue, Stream& stream);

CV_EXPORTS void medianBlur(const MetalMat& src, MetalMat& dst, int ksize);
CV_EXPORTS void medianBlur(const MetalMat& src, MetalMat& dst, int ksize, Stream& stream);

CV_EXPORTS void matchTemplate(const MetalMat& image, const MetalMat& templ, MetalMat& result, int method, InputArray mask = noArray());
CV_EXPORTS void matchTemplate(const MetalMat& image, const MetalMat& templ, MetalMat& result, int method, InputArray mask, Stream& stream);

//! @}
}} // cv::metal

#endif // OPENCV_IMGPROC_METAL_HPP