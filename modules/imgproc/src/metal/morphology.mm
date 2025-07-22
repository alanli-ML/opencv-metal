// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_precomp.hpp"

namespace cv { namespace metal {

//==============================================================================
// Erode Implementation
//==============================================================================

void erode(const MetalMat& src, MetalMat& dst, InputArray kernel, Point anchor, int iterations, int borderType, const Scalar& borderValue, Stream& stream)
{
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    CV_Assert(iterations > 0);
    CV_Assert(borderType == BORDER_DEFAULT || borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE);

    Mat kernelMat = kernel.getMat();
    CV_Assert(kernelMat.rows % 2 == 1 && kernelMat.cols % 2 == 1); // Odd kernel size required

    dst.create(src.size(), src.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MPSImageAreaMin *erosion = [[MPSImageAreaMin alloc] initWithDevice:MetalContext::getInstance().device
                                                               kernelWidth:kernelMat.cols
                                                              kernelHeight:kernelMat.rows];
        CV_Assert(erosion != nil);
        erosion.edgeMode = (borderType == BORDER_REPLICATE) ? MPSImageEdgeModeClamp : MPSImageEdgeModeZero;

        // For multiple iterations, we need to ping-pong between textures
        MetalMat temp_src = src;
        MetalMat temp_dst = dst;
        
        for (int i = 0; i < iterations; ++i) {
            if (i > 0) {
                // Swap source and destination for next iteration
                temp_src = temp_dst;
                if (i < iterations - 1) {
                    // Create temporary texture for intermediate results
                    temp_dst.create(src.size(), src.type());
                } else {
                    temp_dst = dst; // Final iteration writes to output
                }
            }
            
            [erosion encodeToCommandBuffer:commandBuffer
                             sourceTexture:temp_src.texture()
                        destinationTexture:temp_dst.texture()];
        }
    }
}

void erode(const MetalMat& src, MetalMat& dst, InputArray kernel, Point anchor, int iterations, int borderType, const Scalar& borderValue)
{
    Stream stream;
    erode(src, dst, kernel, anchor, iterations, borderType, borderValue, stream);
    stream.commitAndWait();
}

//==============================================================================
// Dilate Implementation
//==============================================================================

void dilate(const MetalMat& src, MetalMat& dst, InputArray kernel, Point anchor, int iterations, int borderType, const Scalar& borderValue, Stream& stream)
{
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);
    CV_Assert(iterations > 0);
    CV_Assert(borderType == BORDER_DEFAULT || borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE);

    Mat kernelMat = kernel.getMat();
    CV_Assert(kernelMat.rows % 2 == 1 && kernelMat.cols % 2 == 1); // Odd kernel size required

    dst.create(src.size(), src.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MPSImageAreaMax *dilation = [[MPSImageAreaMax alloc] initWithDevice:MetalContext::getInstance().device
                                                                kernelWidth:kernelMat.cols
                                                               kernelHeight:kernelMat.rows];
        CV_Assert(dilation != nil);
        dilation.edgeMode = (borderType == BORDER_REPLICATE) ? MPSImageEdgeModeClamp : MPSImageEdgeModeZero;

        // For multiple iterations, we need to ping-pong between textures
        MetalMat temp_src = src;
        MetalMat temp_dst = dst;
        
        for (int i = 0; i < iterations; ++i) {
            if (i > 0) {
                // Swap source and destination for next iteration
                temp_src = temp_dst;
                if (i < iterations - 1) {
                    // Create temporary texture for intermediate results
                    temp_dst.create(src.size(), src.type());
                } else {
                    temp_dst = dst; // Final iteration writes to output
                }
            }
            
            [dilation encodeToCommandBuffer:commandBuffer
                              sourceTexture:temp_src.texture()
                         destinationTexture:temp_dst.texture()];
        }
    }
}

void dilate(const MetalMat& src, MetalMat& dst, InputArray kernel, Point anchor, int iterations, int borderType, const Scalar& borderValue)
{
    Stream stream;
    dilate(src, dst, kernel, anchor, iterations, borderType, borderValue, stream);
    stream.commitAndWait();
}

}} // cv::metal 