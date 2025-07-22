// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_METALFILTERS_HPP
#define OPENCV_METALFILTERS_HPP

#include <opencv2/core/metal.hpp>
#include <opencv2/imgproc.hpp>

namespace cv { namespace metal {

//! @addtogroup metalfilters
//! @{

/** @brief Common interface for all Metal filters.
 */
class CV_EXPORTS Filter : public cv::Algorithm
{
public:
    /** @brief Applies the specified filter to the image.

    @param src Input image.
    @param dst Output image.
    @param stream Stream for the asynchronous version.
     */
    virtual void apply(const MetalMat& src, MetalMat& dst, Stream& stream) = 0;
    void apply(const MetalMat& src, MetalMat& dst) {
        Stream defaultStream;
        apply(src, dst, defaultStream);
    }
};

/** @brief Creates a normalized 2D box filter.

@param srcType Input image type. Supported types are CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4.
@param dstType Output image type. Must be the same as srcType.
@param ksize Kernel size.
@param anchor Anchor point. The default value Point(-1, -1) means that the anchor is at the kernel
center. MPSImageBox only supports a centered anchor.
@param borderMode Pixel extrapolation method. For details, see borderInterpolate. BORDER_CONSTANT with non-zero border is not supported.

@sa boxFilter
 */
CV_EXPORTS Ptr<Filter> createBoxFilter(int srcType, int dstType, Size ksize,
                                       Point anchor = Point(-1,-1),
                                       int borderMode = BORDER_DEFAULT,
                                       Scalar borderVal = Scalar::all(0));

//! @}
}} // cv::metal

#endif // OPENCV_METALFILTERS_HPP