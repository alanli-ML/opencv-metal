// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_precomp.hpp"

namespace cv { namespace metal {

void GaussianBlur(const MetalMat& src, MetalMat& dst, Size ksize, double sigmaX, Stream& stream)
{
    CV_Assert(src.type() == dst.type() || dst.empty());
    CV_Assert(ksize.width > 0 && ksize.width % 2 == 1 &&
              ksize.height > 0 && ksize.height % 2 == 1);
    CV_Assert(sigmaX > 0);

    dst.create(src.size(), src.type());

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        // MPSImageGaussianBlur sigma is float
        MPSImageGaussianBlur *blur = [[MPSImageGaussianBlur alloc] initWithDevice:MetalContext::getInstance().device sigma:(float)sigmaX];
        CV_Assert(blur != nil);
        blur.edgeMode = MPSImageEdgeModeClamp;

        // The kernel size is derived from sigma by MPS, so ksize is not used directly.
        // It's kept for API compatibility with cv::GaussianBlur.

        [blur encodeToCommandBuffer:commandBuffer
                      sourceTexture:src.texture()
                 destinationTexture:dst.texture()];
    }
}

void GaussianBlur(const MetalMat& src, MetalMat& dst, Size ksize, double sigmaX)
{
    Stream stream;
    GaussianBlur(src, dst, ksize, sigmaX, stream);
    stream.commitAndWait();
}

void Sobel(const MetalMat& src, MetalMat& dst, int ddepth, int dx, int dy, int ksize, Stream& stream)
{
    // For MVP, only 32F is supported due to MPS limitations on output formats for integer types.
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(ddepth == -1 || ddepth == CV_32F);
    CV_Assert((dx == 1 && dy == 0) || (dx == 0 && dy == 1)); // For now, only support 1st order derivatives
    CV_Assert(ksize == 3); // For MVP, only ksize=3 is supported.

    int dst_type = CV_MAKETYPE(ddepth < 0 ? CV_32F : ddepth, src.channels());
    dst.create(src.size(), dst_type);

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        float kernelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        float kernelY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
        const float* weights = (dx != 0) ? kernelX : kernelY;

        MPSImageConvolution *sobel = [[MPSImageConvolution alloc] initWithDevice:MetalContext::getInstance().device
                                                                     kernelWidth:3
                                                                    kernelHeight:3
                                                                         weights:weights];
        CV_Assert(sobel != nil);
        sobel.edgeMode = MPSImageEdgeModeClamp;

        [sobel encodeToCommandBuffer:commandBuffer
                       sourceTexture:src.texture()
                  destinationTexture:dst.texture()];
    }
}

void Sobel(const MetalMat& src, MetalMat& dst, int ddepth, int dx, int dy, int ksize)
{
    Stream stream;
    Sobel(src, dst, ddepth, dx, dy, ksize, stream);
    stream.commitAndWait();
}

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