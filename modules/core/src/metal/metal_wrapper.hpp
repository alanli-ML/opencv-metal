// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_METAL_WRAPPER_HPP
#define OPENCV_CORE_METAL_WRAPPER_HPP

#include "opencv2/core/metal.hpp"
#include <map>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
typedef void* id;
#endif

namespace cv { namespace metal {

// This is an internal class that is not exposed in public headers
class MetalContext
{
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;

    static MetalContext& getInstance();

    id<MTLFunction> getMetalFunction(const std::string& kernelSource, const std::string& functionName, bool isMpsAvailable = true);

private:
    MetalContext();
    // disable copy/assignment
    MetalContext(const MetalContext&);
    MetalContext& operator=(const MetalContext&);

    std::map<std::string, id<MTLLibrary>> libraryCache;
    std::map<std::string, id<MTLFunction>> functionCache;
    Mutex cacheMutex;
};

class Stream::Impl
{
public:
    Impl();
    ~Impl();

    id<MTLCommandBuffer> getCommandBuffer();
    void commit();
    void waitUntilCompleted();
    void commitAndWait();
    void syncCPU();
    bool hasEnqueuedCommands() const;

    id<MTLCommandBuffer> commandBuffer;
};

}} // cv::metal

#endif // OPENCV_CORE_METAL_WRAPPER_HPP