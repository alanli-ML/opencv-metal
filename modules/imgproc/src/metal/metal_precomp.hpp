// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMGPROC_METAL_PRECOMP_HPP
#define OPENCV_IMGPROC_METAL_PRECOMP_HPP

#include "opencv2/core.hpp"
#include "opencv2/core/metal.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/metal.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// This is an internal header which is not installed.
// We can get direct access to MetalMat internals here.
#include "../../../core/src/metal/metal_wrapper.hpp"
#endif

#endif // OPENCV_IMGPROC_METAL_PRECOMP_HPP