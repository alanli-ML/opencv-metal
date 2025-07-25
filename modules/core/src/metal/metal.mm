// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "metal_precomp.hpp"
#include "metal_wrapper.hpp"
// Removed imgproc dependency to avoid circular dependency

// This file will be populated with Objective-C++ code.
// The .mm extension allows mixing C++ and Objective-C.

namespace cv { namespace metal {

class MetalMatData
{
public:
    MetalMatData();
    ~MetalMatData();

private:
    friend class MetalMat;
    id<MTLTexture> texture;
};

MetalMatData::MetalMatData() : texture(nil) {}
MetalMatData::~MetalMatData()
{
    // ARC will release the texture when this object is destroyed.
}

MetalContext::MetalContext()
{
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device)
        {
            CV_Error(Error::StsError, "Metal is not supported on this device");
        }
        commandQueue = [device newCommandQueue];
    }
}

MetalContext& MetalContext::getInstance()
{
    static MetalContext instance;
    return instance;
}

id<MTLFunction> MetalContext::getMetalFunction(const std::string& kernelSource, const std::string& functionName, bool isMpsAvailable)
{
    std::lock_guard<Mutex> lock(cacheMutex);

    // Check function cache first
    auto funcIt = functionCache.find(functionName);
    if (funcIt != functionCache.end()) {
        return funcIt->second;
    }

    // Check library cache
    id<MTLLibrary> library = nil;
    auto libIt = libraryCache.find(kernelSource);
    if (libIt != libraryCache.end()) {
        library = libIt->second;
    } else {
        // Compile new library
        NSError *error = nil;
        NSString *sourceString = [NSString stringWithUTF8String:kernelSource.c_str()];
        MTLCompileOptions *options = [MTLCompileOptions new];

        if (isMpsAvailable)
        {
            // Allow MPS data types if needed
            options.preprocessorMacros = @{ @"MPS_TYPES_IMPL" : @"" };
        }

        library = [device newLibraryWithSource:sourceString options:options error:&error];
        if (!library) {
            CV_Error(Error::StsError, [error.localizedDescription cStringUsingEncoding:NSUTF8StringEncoding]);
        }
        libraryCache[kernelSource] = library;
    }

    // Get function from library
    NSString *functionNameString = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:functionNameString];
    if (!function) {
        CV_Error(Error::StsError, "Failed to find Metal function: " + functionName);
    }

    // Cache and return function
    functionCache[functionName] = function;
    return function;
}

static int getCVPixelFormatFromMetal(MTLPixelFormat pixelFormat)
{
    switch(pixelFormat)
    {
        case MTLPixelFormatR8Unorm:
        case MTLPixelFormatR8Uint:
            return CV_8UC1;
        case MTLPixelFormatBGRA8Unorm: return CV_8UC4;
        case MTLPixelFormatR32Float: return CV_32FC1;
        case MTLPixelFormatRGBA32Float: return CV_32FC4;
        case MTLPixelFormatR32Sint: return CV_32SC1;
        case MTLPixelFormatRGBA32Sint: return CV_32SC4;
        default: return -1;
    }
}

static MTLPixelFormat getMetalPixelFormat(int type)
{
    int depth = CV_MAT_DEPTH(type);
    int cn = CV_MAT_CN(type);
    switch(depth)
    {
        case CV_8U:
            switch(cn)
            {
                case 1: return MTLPixelFormatR8Uint;   // Use integer format to avoid normalization for masks
                case 4: return MTLPixelFormatBGRA8Unorm; // OpenCV's default for 4-channel 8U is BGRA
            }
            break;
        case CV_32F:
            switch(cn)
            {
                case 1: return MTLPixelFormatR32Float;
                case 4: return MTLPixelFormatRGBA32Float;
            }
            break;
        case CV_32S:
            switch(cn)
            {
                case 1: return MTLPixelFormatR32Sint;
                case 4: return MTLPixelFormatRGBA32Sint;
            }
            break;
    }
    return MTLPixelFormatInvalid;
}

MetalMat::MetalMat()
: flags(0), rows_(0), cols_(0), u(0), offset(0), texture_not_owned_(nil), original_channels_(0)
{
    step[0] = step[1] = 0;
}

MetalMat::MetalMat(int _rows, int _cols, int _type)
    : flags(0), rows_(0), cols_(0), u(0), offset(0), texture_not_owned_(nil), original_channels_(CV_MAT_CN(_type))
{
    step[0] = step[1] = 0;
    create(_rows, _cols, _type);
}

MetalMat::MetalMat(const Mat& m)
    : flags(0), rows_(0), cols_(0), u(0), offset(0), texture_not_owned_(nil), original_channels_(m.channels())
{
    step[0] = step[1] = 0;
    upload(m);
}

MetalMat::MetalMat(Size size, int type)
    : flags(0), rows_(0), cols_(0), u(0), offset(0), texture_not_owned_(nil), original_channels_(CV_MAT_CN(type))
{
    step[0] = step[1] = 0;
    create(size.height, size.width, type);
}

MetalMat::MetalMat(id texture)
    : flags(0), rows_(0), cols_(0), u(), offset(0), texture_not_owned_(texture), original_channels_(0)
{
    step[0] = step[1] = 0;
    if (texture == nil) {
        texture_not_owned_ = nil;
        return;
    }

    @autoreleasepool {
        id<MTLTexture> tex = (id<MTLTexture>)texture;
        int cv_type = getCVPixelFormatFromMetal([tex pixelFormat]);
        if (cv_type == -1)
            CV_Error(Error::StsUnsupportedFormat, "Unsupported MTLTexture pixel format for MetalMat wrapping");

        flags = Mat::MAGIC_VAL + cv_type;
        rows_ = (int)[tex height];
        cols_ = (int)[tex width];
        step[0] = cols_ * elemSize();
        step[1] = elemSize();
    }
}


MetalMat::~MetalMat()
{
    release();
}

MetalMat::MetalMat(const MetalMat& m)
    : flags(m.flags), rows_(m.rows_), cols_(m.cols_), u(m.u), offset(m.offset), texture_not_owned_(m.texture_not_owned_), original_channels_(m.original_channels_)
{
    step[0] = m.step[0];
    step[1] = m.step[1];
}

MetalMat& MetalMat::operator=(const MetalMat& m)
{
    if (this != &m)
    {
        u = m.u;
        texture_not_owned_ = m.texture_not_owned_;
        flags = m.flags;
        rows_ = m.rows_;
        cols_ = m.cols_;
        offset = m.offset;
        step[0] = m.step[0];
        step[1] = m.step[1];
        original_channels_ = m.original_channels_;
    }
    return *this;
}

MetalMat MetalMat::clone() const
{
    MetalMat m;
    if (empty())
        return m;

    m.create(rows_, cols_, type());

    @autoreleasepool {
        Stream stream;
        id<MTLBlitCommandEncoder> blitEncoder = StreamAccessor::createBlitEncoder(stream);

        [blitEncoder copyFromTexture:texture()
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:{0, 0, 0}
                          sourceSize:{(NSUInteger)cols_, (NSUInteger)rows_, 1}
                           toTexture:m.texture()
                    destinationSlice:0
                    destinationLevel:0
                   destinationOrigin:{0, 0, 0}];
        [blitEncoder endEncoding];
        stream.commitAndWait();
    }
    return m;
}

MetalMat::MetalMat(const MetalMat& m, const Rect& roi)
    : flags(0), rows_(0), cols_(0), u(0), offset(0), texture_not_owned_(nil)
{
    step[0] = step[1] = 0;
    CV_Assert(!m.empty());
    CV_Assert(roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 &&
              roi.x + roi.width <= m.cols_ && roi.y + roi.height <= m.rows_);
    if (roi.width == 0 || roi.height == 0)
        return;

    create(roi.height, roi.width, m.type());

    @autoreleasepool {
        Stream stream;
        id<MTLBlitCommandEncoder> blitEncoder = StreamAccessor::createBlitEncoder(stream);

        MTLOrigin srcOrigin = {(NSUInteger)roi.x, (NSUInteger)roi.y, 0};
        MTLSize srcSize = {(NSUInteger)roi.width, (NSUInteger)roi.height, 1};
        MTLOrigin dstOrigin = {0, 0, 0};

        [blitEncoder copyFromTexture:m.texture()
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:srcOrigin
                          sourceSize:srcSize
                           toTexture:texture()
                    destinationSlice:0
                    destinationLevel:0
                   destinationOrigin:dstOrigin];

        [blitEncoder endEncoding];
        stream.commitAndWait();
    }
}

MetalMat::MetalMat(const MetalMat& m, const Range& rowRange, const Range& colRange)
{
    *this = MetalMat(m, Rect(colRange.start, rowRange.start, colRange.size(), rowRange.size()));
}


MetalMat MetalMat::operator()(const Range& rowRange, const Range& colRange) const
{
    return MetalMat(*this, rowRange, colRange);
}

MetalMat MetalMat::operator()(const Rect& roi) const
{
    return MetalMat(*this, roi);
}

void MetalMat::create(int _rows, int _cols, int _type)
{
    _type = CV_MAT_TYPE(_type);
    if (u && rows_ == _rows && cols_ == _cols && type() == _type)
        return;

    release();
    texture_not_owned_ = nil;

    if (_rows <= 0 || _cols <= 0)
        return;

    @autoreleasepool {
        flags = Mat::MAGIC_VAL + _type;
        rows_ = _rows;
        cols_ = _cols;
        original_channels_ = CV_MAT_CN(_type);
        offset = 0;
        step[0] = cols_ * elemSize();
        step[1] = elemSize();

        u = makePtr<MetalMatData>();

        MetalContext& ctx = MetalContext::getInstance();
        MTLPixelFormat pixelFormat = getMetalPixelFormat(type());
        if (pixelFormat == MTLPixelFormatInvalid)
        {
            CV_Error(Error::StsUnsupportedFormat, "Unsupported MetalMat format");
        }

        MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:pixelFormat
            width:_cols
            height:_rows
            mipmapped:NO];

        textureDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        u->texture = [ctx.device newTextureWithDescriptor:textureDescriptor];

        if (!u->texture)
        {
            CV_Error(Error::StsError, "Failed to create MTLTexture");
        }
    }
}

void MetalMat::create(Size _size, int _type)
{
    create(_size.height, _size.width, _type);
}

void MetalMat::upload(const Mat& m)
{
    CV_Assert(!m.empty());
    int _type = m.type();
    int _cn = m.channels();

    // Preserve original channels for correct download later
    original_channels_ = _cn;

    if (_cn == 3)
    {
        // Store as 4-channel BGRA internally
        _type = CV_MAKETYPE(m.depth(), 4);
    }

    create(m.rows, m.cols, _type);
    CV_Assert(u && u->texture);

    @autoreleasepool {
        Mat src_to_upload;
        if (_cn == 3)
        {
            int depth = m.depth();
            int dstType = (depth == CV_8U) ? CV_8UC4 : CV_32FC4;
            src_to_upload.create(m.size(), dstType);

            int total_pixels = m.rows * m.cols;
            if (depth == CV_8U)
            {
                const uchar* src_ptr = m.ptr<uchar>();
                uchar* dst_ptr = src_to_upload.ptr<uchar>();
                for (int i = 0; i < total_pixels; i++)
                {
                    dst_ptr[i*4 + 0] = src_ptr[i*3 + 0]; // B
                    dst_ptr[i*4 + 1] = src_ptr[i*3 + 1]; // G
                    dst_ptr[i*4 + 2] = src_ptr[i*3 + 2]; // R
                    dst_ptr[i*4 + 3] = 255;              // A
                }
            }
            else if (depth == CV_32F)
            {
                const float* src_ptr = m.ptr<float>();
                float* dst_ptr = src_to_upload.ptr<float>();
                for (int i = 0; i < total_pixels; i++)
                {
                    dst_ptr[i*4 + 0] = src_ptr[i*3 + 0];
                    dst_ptr[i*4 + 1] = src_ptr[i*3 + 1];
                    dst_ptr[i*4 + 2] = src_ptr[i*3 + 2];
                    dst_ptr[i*4 + 3] = 1.0f;
                }
            }
            else
            {
                CV_Error(Error::StsUnsupportedFormat, "Unsupported depth for 3-channel upload");
            }

            // original_channels_ was overwritten by create(); restore to 3 so download converts back
            original_channels_ = 3;
        }
        else
        {
            src_to_upload = m;
        }

        // Write into Metal texture
        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)cols_, (NSUInteger)rows_);
        [u->texture replaceRegion:region
                        mipmapLevel:0
                        withBytes:src_to_upload.data
                        bytesPerRow:src_to_upload.step];
    }
    // Note: Non 3-channel case handled above; no additional else needed.
}


void MetalMat::download(Mat& m, Stream& stream, bool sync) const
{
    CV_Assert(!empty());
    if (sync)
    {
        // Stream-aware sync download - commit any pending operations first
        if (stream.hasEnqueuedCommands()) {
            stream.syncCPU();
        }
        downloadImpl(m);
    }
    else
    {
        // Asynchronous download - allocate Mat immediately and queue texture read operation
        @autoreleasepool {
            // Allocate the output Mat structure immediately so caller can use it
            if (original_channels_ == 3 && CV_MAT_CN(type()) == 4)
            {
                int depth = CV_MAT_DEPTH(type());
                int outType = CV_MAKETYPE(depth, 3);
                m.create(rows_, cols_, outType);
            }
            else
            {
                m.create(rows_, cols_, type());
            }
            
            // Create temporary buffer for texture data
            MetalContext& ctx = MetalContext::getInstance();
            
            // Calculate buffer size for internal texture format (before channel conversion)
            size_t bytesPerRow = cols_ * elemSize();
            size_t totalBytes = rows_ * bytesPerRow;
            
            id<MTLBuffer> tempBuffer = [ctx.device newBufferWithLength:totalBytes 
                                                               options:MTLResourceStorageModeShared];
            
            if (!tempBuffer) {
                CV_Error(Error::StsError, "Failed to create temporary buffer for async download");
            }
            
            // For async download, we need command buffer reference for completion handler
            id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
            
            // Queue texture-to-buffer copy operation
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            [blitEncoder copyFromTexture:texture()
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(cols_, rows_, 1)
                                toBuffer:tempBuffer
                       destinationOffset:0
                    destinationBytesPerRow:bytesPerRow
                  destinationBytesPerImage:totalBytes];
            [blitEncoder endEncoding];
            
            // Capture necessary values for completion handler
            int capturedRows = rows_;
            int capturedCols = cols_;
            int capturedOriginalChannels = original_channels_;
            int capturedType = type();
            size_t capturedBytesPerRow = bytesPerRow;
            
            // Add completion handler to copy buffer data to Mat when GPU operation finishes
            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
                @autoreleasepool {
                    // Copy buffer data to Mat with proper channel conversion
                    if (capturedOriginalChannels == 3 && CV_MAT_CN(capturedType) == 4)
                    {
                        // Handle 3-channel case: convert from internal BGRA to BGR
                        int depth = CV_MAT_DEPTH(capturedType);
                        
                        // Create temporary Mat for 4-channel data
                        Mat tmp(capturedRows, capturedCols, capturedType);
                        memcpy(tmp.data, [tempBuffer contents], totalBytes);
                        
                        // Convert to 3-channel output
                        size_t nPixels = (size_t)capturedRows * (size_t)capturedCols;
                        if (depth == CV_8U)
                        {
                            const uchar* srcPtr = tmp.ptr<uchar>();
                            uchar* dstPtr = m.ptr<uchar>();
                            for (size_t i = 0; i < nPixels; ++i)
                            {
                                dstPtr[i*3 + 0] = srcPtr[i*4 + 0]; // B
                                dstPtr[i*3 + 1] = srcPtr[i*4 + 1]; // G
                                dstPtr[i*3 + 2] = srcPtr[i*4 + 2]; // R
                            }
                        }
                        else if (depth == CV_32F)
                        {
                            const float* srcPtr = tmp.ptr<float>();
                            float* dstPtr = m.ptr<float>();
                            for (size_t i = 0; i < nPixels; ++i)
                            {
                                dstPtr[i*3 + 0] = srcPtr[i*4 + 0];
                                dstPtr[i*3 + 1] = srcPtr[i*4 + 1];
                                dstPtr[i*3 + 2] = srcPtr[i*4 + 2];
                            }
                        }
                    }
                    else
                    {
                        // Direct copy for matching channel formats
                        memcpy(m.data, [tempBuffer contents], totalBytes);
                    }
                }
            }];
        }
    }
}

void MetalMat::download(Mat& m) const
{
    Stream defaultStream;
    download(m, defaultStream, true);  // Always blocking with own stream
    defaultStream.commitAndWait();     // Ensure completion
}

void MetalMat::downloadImpl(Mat& m) const
{
    // Handle 3-channel matrices stored internally as 4-channel BGRA/RGBA
    if (original_channels_ == 3 && CV_MAT_CN(type()) == 4)
    {
        int depth = CV_MAT_DEPTH(type());
        int internalType = type();
        Mat tmp(rows_, cols_, internalType);

        @autoreleasepool {
            MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)cols_, (NSUInteger)rows_);
            [u->texture getBytes:tmp.data
                     bytesPerRow:tmp.step
                     fromRegion:region
                     mipmapLevel:0];
        }

        int outType = CV_MAKETYPE(depth, 3);
        m.create(rows_, cols_, outType);

        size_t nPixels = (size_t)rows_ * (size_t)cols_;
        if (depth == CV_8U)
        {
            const uchar* srcPtr = tmp.ptr<uchar>();
            uchar* dstPtr = m.ptr<uchar>();
            for (size_t i = 0; i < nPixels; ++i)
            {
                dstPtr[i*3 + 0] = srcPtr[i*4 + 0];
                dstPtr[i*3 + 1] = srcPtr[i*4 + 1];
                dstPtr[i*3 + 2] = srcPtr[i*4 + 2];
            }
        }
        else if (depth == CV_32F)
        {
            const float* srcPtr = tmp.ptr<float>();
            float* dstPtr = m.ptr<float>();
            for (size_t i = 0; i < nPixels; ++i)
            {
                dstPtr[i*3 + 0] = srcPtr[i*4 + 0];
                dstPtr[i*3 + 1] = srcPtr[i*4 + 1];
                dstPtr[i*3 + 2] = srcPtr[i*4 + 2];
            }
        }
        else
        {
            CV_Error(Error::StsUnsupportedFormat, "Unsupported depth for 3-channel download");
        }
    }
    else
    {
        m.create(rows_, cols_, type());
        @autoreleasepool {
            MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)cols_, (NSUInteger)rows_);
            [u->texture getBytes:m.data
                     bytesPerRow:m.step
                     fromRegion:region
                     mipmapLevel:0];
        }
    }
}

bool MetalMat::empty() const
{
    return !u && !texture_not_owned_;
}

id MetalMat::texture() const
{
    return u ? u->texture : (id<MTLTexture>)texture_not_owned_;
}

void MetalMat::release()
{
    u.release();
    texture_not_owned_ = nil;
    rows_ = cols_ = 0;
    offset = 0;
    step[0] = step[1] = 0;
    flags = 0;
}

void MetalMat::setOriginalChannels(int cn)
{
    original_channels_ = cn;
}

// Stream Implementation
Stream::Impl::Impl() {
    MetalContext& ctx = MetalContext::getInstance();
    commandBuffer = [ctx.commandQueue commandBuffer];
}
Stream::Impl::~Impl() {}

id<MTLCommandBuffer> Stream::Impl::getCommandBuffer()
{
    if (commandBuffer == nil || [commandBuffer status] >= MTLCommandBufferStatusCommitted)
    {
        MetalContext& ctx = MetalContext::getInstance();
        commandBuffer = [ctx.commandQueue commandBuffer];
    }
    
    return commandBuffer;
}

Stream::Stream() { impl = makePtr<Impl>(); }
Stream::~Stream() {}
Stream::Stream(const Stream& s) : impl(s.impl) {}
Stream& Stream::operator=(const Stream& s) { impl = s.impl; return *this; }

void Stream::Impl::commit()
{
    if (commandBuffer)
    {
        [commandBuffer commit];
    }
}

void Stream::Impl::waitUntilCompleted()
{
    if (commandBuffer)
    {
        [commandBuffer waitUntilCompleted];
    }
}

void Stream::Impl::commitAndWait()
{
    if (commandBuffer && hasEnqueuedCommands())
    {
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void Stream::Impl::syncCPU()
{
    if (commandBuffer && hasEnqueuedCommands())
    {
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Always create replacement buffer after sync for continued usage
        MetalContext& ctx = MetalContext::getInstance();
        commandBuffer = [ctx.commandQueue commandBuffer];
    }
}

bool Stream::Impl::hasEnqueuedCommands() const
{
    return commandBuffer && ([commandBuffer status] == MTLCommandBufferStatusEnqueued || [commandBuffer status] == MTLCommandBufferStatusNotEnqueued);
}

void Stream::commit()
{
    if (impl) impl->commit();
}

void Stream::waitUntilCompleted()
{
    if (impl) impl->waitUntilCompleted();
}

void Stream::commitAndWait()
{
    if (impl) impl->commitAndWait();
}

void Stream::syncCPU()
{
    if (impl) impl->syncCPU();
}

bool Stream::hasEnqueuedCommands() const
{
    return impl && impl->hasEnqueuedCommands();
}

id StreamAccessor::createComputeEncoder(const Stream& stream)
{
    if (!stream.impl)
        return nil;
    return [stream.impl->commandBuffer computeCommandEncoder];
}

id StreamAccessor::createBlitEncoder(const Stream& stream)
{
    if (!stream.impl)
        return nil;
    id<MTLCommandBuffer> commandBuffer = stream.impl->getCommandBuffer();
    return [commandBuffer blitCommandEncoder];
}

id StreamAccessor::getCommandBuffer(const Stream& stream)
{
    if (!stream.impl)
        return nil;
    return stream.impl->getCommandBuffer();
}

}} // cv::metal