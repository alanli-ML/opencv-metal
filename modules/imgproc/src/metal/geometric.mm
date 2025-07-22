// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_precomp.hpp"

namespace cv { namespace metal {

//==============================================================================
// Resize Implementation
//==============================================================================

void resize(const MetalMat& src, MetalMat& dst, Size dsize, double fx, double fy, int interpolation, Stream& stream)
{
    CV_Assert(!src.empty());
    Size ssize = src.size();

    if (dsize.empty())
    {
        CV_Assert(fx > 0 && fy > 0);
        dsize = Size(saturate_cast<int>(ssize.width * fx), saturate_cast<int>(ssize.height * fy));
    }
    else
    {
        CV_Assert(dsize.width > 0 && dsize.height > 0);
    }
    CV_Assert(dst.empty() || dst.type() == src.type());

    dst.create(dsize, src.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MPSImageScale *scaler = nil;

        switch (interpolation)
        {
            case INTER_NEAREST:
                // Use bilinear scale with filter set to nearest neighbor
                scaler = [[MPSImageBilinearScale alloc] initWithDevice:MetalContext::getInstance().device];
                break;
            case INTER_LINEAR:
                scaler = [[MPSImageBilinearScale alloc] initWithDevice:MetalContext::getInstance().device];
                break;
            case INTER_LANCZOS4:
                scaler = [[MPSImageLanczosScale alloc] initWithDevice:MetalContext::getInstance().device];
                break;
            default:
                CV_Error(Error::StsBadArg, "Unsupported interpolation type for Metal backend");
        }
        CV_Assert(scaler != nil);

        [scaler encodeToCommandBuffer:commandBuffer
                        sourceTexture:src.texture()
                   destinationTexture:dst.texture()];
    }
}

void resize(const MetalMat& src, MetalMat& dst, Size dsize, double fx, double fy, int interpolation)
{
    Stream stream;
    resize(src, dst, dsize, fx, fy, interpolation, stream);
    stream.commitAndWait();
}

}} // cv::metal 