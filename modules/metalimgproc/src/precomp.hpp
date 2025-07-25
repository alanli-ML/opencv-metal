#ifndef OPENCV_METALIMGPROC_PRECOMP_HPP
#define OPENCV_METALIMGPROC_PRECOMP_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/metal.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/metalimgproc.hpp"

#ifdef HAVE_METAL

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#else
typedef void* id;
#endif

// Include internal Metal wrapper
#include "../../core/src/metal/metal_wrapper.hpp"

#endif // HAVE_METAL

#endif // OPENCV_METALIMGPROC_PRECOMP_HPP 