// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_METAL_HPP
#define OPENCV_CORE_METAL_HPP

#ifdef __cplusplus

#include "opencv2/core.hpp"

// Forward declaration for Objective-C types
#ifdef __OBJC__
@protocol MTLTexture;
@protocol MTLCommandBuffer;
#else
typedef void* id;
#endif

namespace cv { namespace metal {

//! @addtogroup core_metal
//! @{

class MetalMatData;

class CV_EXPORTS MetalMat
{
public:
    MetalMat();
    MetalMat(int rows, int cols, int type);
    explicit MetalMat(const Mat& m);
    MetalMat(Size size, int type);
    explicit MetalMat(id texture);

    ~MetalMat();
    MetalMat(const MetalMat& m);
    MetalMat(const MetalMat& m, const Range& rowRange, const Range& colRange);
    MetalMat(const MetalMat& m, const Rect& roi);
    MetalMat& operator=(const MetalMat& m);

    void upload(const Mat& m);
    void download(Mat& m) const;

    //! returns a deep copy of the MetalMat
    MetalMat clone() const;

    //! creates a new MetalMat header for the specified sub-array
    MetalMat operator()(const Range& rowRange, const Range& colRange) const;
    MetalMat operator()(const Rect& roi) const;

    // Accessors
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int type() const { return CV_MAT_TYPE(flags); }
    int depth() const { return CV_MAT_DEPTH(flags); }
    int channels() const { return CV_MAT_CN(flags); }
    bool empty() const;
    Size size() const { return Size(cols(), rows()); }
    size_t elemSize() const { return CV_ELEM_SIZE(type()); }
    size_t elemSize1() const { return CV_ELEM_SIZE1(type()); }

    id texture() const;

    void create(int _rows, int _cols, int _type);
    void create(Size _size, int _type);

private:
    void release();

    int flags;
    int rows_, cols_;
    Ptr<MetalMatData> u;
    size_t offset;
    size_t step[2];
    id texture_not_owned_;
};

class CV_EXPORTS_W Stream
{
public:
    CV_WRAP Stream();
    explicit Stream(id commandBuffer);
    ~Stream();

    Stream(const Stream&);
    Stream& operator=(const Stream&);

    CV_WRAP void commit();
    CV_WRAP void waitUntilCompleted();
    CV_WRAP void commitAndWait();
    CV_WRAP bool hasEnqueuedCommands() const;

    class Impl;
    Ptr<Impl> impl;
};

struct CV_EXPORTS StreamAccessor
{
    static id getCommandBuffer(const Stream& stream);
};

CV_EXPORTS void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst);
CV_EXPORTS void add(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream);

CV_EXPORTS void subtract(const MetalMat& src1, const MetalMat& src2, MetalMat& dst);
CV_EXPORTS void subtract(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream);

CV_EXPORTS void multiply(const MetalMat& src1, const MetalMat& src2, MetalMat& dst);
CV_EXPORTS void multiply(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream);

CV_EXPORTS void divide(const MetalMat& src1, const MetalMat& src2, MetalMat& dst);
CV_EXPORTS void divide(const MetalMat& src1, const MetalMat& src2, MetalMat& dst, Stream& stream);


//! @}
}} // cv::metal

#endif // __cplusplus

#endif // OPENCV_CORE_METAL_HPP