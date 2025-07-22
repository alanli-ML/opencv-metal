#ifndef OPENCV_TEST_PRECOMP_HPP
#define OPENCV_TEST_PRECOMP_HPP

#include "opencv2/ts.hpp"
#include "opencv2/ts/cuda_test.hpp"

#ifdef HAVE_METAL
#include "opencv2/core/metal.hpp"
#endif

namespace opencv_test {
using namespace testing;
using namespace cv;
}

#endif