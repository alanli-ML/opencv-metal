// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv { namespace metal {

namespace {

class BoxFilter_Metal : public Filter
{
public:
    BoxFilter_Metal(int srcType, int dstType, Size ksize, Point anchor, int borderMode);
    void apply(const MetalMat& src, MetalMat& dst, Stream& stream);

private:
    int srcType_, dstType_;
    Size ksize_;
    Point anchor_;
    int borderMode_;
};

BoxFilter_Metal::BoxFilter_Metal(int srcType, int dstType, Size ksize, Point anchor, int borderMode)
    : srcType_(srcType), dstType_(dstType), ksize_(ksize), anchor_(anchor), borderMode_(borderMode)
{
    CV_Assert(srcType == dstType);
    CV_Assert(srcType == CV_8UC1 || srcType == CV_8UC4 || srcType == CV_32FC1 || srcType == CV_32FC4);
    CV_Assert(ksize.width > 0 && ksize.height > 0);
    // MPSImageBox does not support a non-centered anchor.
    CV_Assert(anchor == Point(-1, -1) || (anchor.x == ksize.width / 2 && anchor.y == ksize.height / 2));
    CV_Assert(borderMode == BORDER_CONSTANT || borderMode == BORDER_REPLICATE || borderMode == BORDER_REFLECT || borderMode == BORDER_REFLECT_101 || borderMode == BORDER_DEFAULT);
}

void BoxFilter_Metal::apply(const MetalMat& src, MetalMat& dst, Stream& stream)
{
    CV_Assert(src.type() == srcType_);

    dst.create(src.size(), dstType_);

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        MPSImageBox *box = [[MPSImageBox alloc] initWithDevice:MetalContext::getInstance().device
                                                   kernelWidth:ksize_.width
                                                  kernelHeight:ksize_.height];
        CV_Assert(box != nil);

        switch (borderMode_)
        {
            case BORDER_REPLICATE:
                box.edgeMode = MPSImageEdgeModeClamp;
                break;
            case BORDER_REFLECT:
                box.edgeMode = MPSImageEdgeModeMirror;
                break;
            case BORDER_CONSTANT:
                box.edgeMode = MPSImageEdgeModeZero;
                break;
            case BORDER_REFLECT_101: // BORDER_DEFAULT has same value as BORDER_REFLECT_101
                box.edgeMode = MPSImageEdgeModeMirror;
                break;
            default:
                CV_Error(Error::StsBadArg, "Unsupported border mode for Metal backend");
        }

        [box encodeToCommandBuffer:commandBuffer
                      sourceTexture:src.texture()
                 destinationTexture:dst.texture()];
    }
}

} // anonymous namespace

Ptr<Filter> createBoxFilter(int srcType, int dstType, Size ksize, Point anchor, int borderMode, Scalar borderVal)
{
    if (borderMode == BORDER_CONSTANT && borderVal != Scalar::all(0))
        CV_Error(Error::StsNotImplemented, "Metal boxFilter with non-zero border value is not supported");

    return makePtr<BoxFilter_Metal>(srcType, dstType, ksize, anchor, borderMode);
}

}} // namespace cv::metal