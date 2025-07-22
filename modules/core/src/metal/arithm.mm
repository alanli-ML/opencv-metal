// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "metal_precomp.hpp"
#include "metal_wrapper.hpp"

namespace cv { namespace metal {

void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(dst.empty() || (dst.size() == src1.size() && dst.type() == src1.type()));

    dst.create(src1.size(), src1.type());

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
        CV_Assert(commandBuffer != nil);

        MPSImageAdd *adder = [[MPSImageAdd alloc] initWithDevice:MetalContext::getInstance().device];
        CV_Assert(adder != nil);

        [adder encodeToCommandBuffer:commandBuffer
                      primaryTexture:src1.texture()
                    secondaryTexture:src2.texture()
                  destinationTexture:dst.texture()];
    }
}

void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    Stream stream;
    add(src1, src2, dst, stream);
    stream.commitAndWait();
}

void subtract(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(dst.empty() || (dst.size() == src1.size() && dst.type() == src1.type()));

    dst.create(src1.size(), src1.type());

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
        CV_Assert(commandBuffer != nil);

        MPSImageSubtract *subtractor = [[MPSImageSubtract alloc] initWithDevice:MetalContext::getInstance().device];
        CV_Assert(subtractor != nil);

        [subtractor encodeToCommandBuffer:commandBuffer
                           primaryTexture:src1.texture()
                         secondaryTexture:src2.texture()
                       destinationTexture:dst.texture()];
    }
}

void subtract(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    Stream stream;
    subtract(src1, src2, dst, stream);
    stream.commitAndWait();
}

void multiply(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(dst.empty() || (dst.size() == src1.size() && dst.type() == src1.type()));

    dst.create(src1.size(), src1.type());

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
        CV_Assert(commandBuffer != nil);

        MPSImageMultiply *multiplier = [[MPSImageMultiply alloc] initWithDevice:MetalContext::getInstance().device];
        CV_Assert(multiplier != nil);

        [multiplier encodeToCommandBuffer:commandBuffer
                           primaryTexture:src1.texture()
                         secondaryTexture:src2.texture()
                       destinationTexture:dst.texture()];
    }
}

void multiply(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    Stream stream;
    multiply(src1, src2, dst, stream);
    stream.commitAndWait();
}

void divide(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream)
{
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(dst.empty() || (dst.size() == src1.size() && dst.type() == src1.type()));

    dst.create(src1.size(), src1.type());

    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
        CV_Assert(commandBuffer != nil);

        MPSImageDivide *divider = [[MPSImageDivide alloc] initWithDevice:MetalContext::getInstance().device];
        CV_Assert(divider != nil);

        [divider encodeToCommandBuffer:commandBuffer
                        primaryTexture:src1.texture()
                      secondaryTexture:src2.texture()
                    destinationTexture:dst.texture()];
    }
}

void divide(const MetalMat& src1, const MetalMat& src2, MetalMat& dst)
{
    Stream stream;
    divide(src1, src2, dst, stream);
    stream.commitAndWait();
}

}} // cv::metal