@cond HAVE_METAL
Using the Metal Backend {#tutorial_metal_basics}
===========================================

@tableofcontents

Goal
----

This tutorial will give a good grasp on how to approach coding by using the Metal backend of OpenCV. As
a prerequisite you should already know how to handle the core, highgui and imgproc modules. So, our
main goals are:

-   What's different compared to the CPU?
-   How to use `cv::metal::MetalMat` for GPU data.
-   How to use `cv::metal::Stream` for efficient, chained operations.
-   Create a simple GPU-accelerated image processing pipeline.
-   How to interoperate with an existing Swift/Objective-C Metal application.

The Metal Backend
---------------

If you have an Apple device with a Metal-capable GPU, you can leverage it for hardware-accelerated computer vision. The Metal backend is designed to be consistent with other hardware acceleration modules in OpenCV, like the CUDA backend.

All functions and data structures for Metal are in the `cv::metal` namespace. You may add this to the default one via the `use namespace` keyword, or mark it everywhere explicitly via `cv::metal::` to avoid confusion.
@code{.cpp}
#include <opencv2/core/metal.hpp>        // Metal data structures
#include <opencv2/imgproc/metal.hpp>     // Metal-accelerated imgproc functions
@endcode

The GPU has its own memory. When you read data from a file with OpenCV into a `cv::Mat` object, that data resides in your system's memory. The CPU works directly on this memory (via its cache), however the GPU cannot. It has to transfer the information required for calculations from the system memory to its own. This is done via an upload process and is time consuming. In the end, the result will have to be downloaded back to your system memory for your CPU to use it. Porting small, single functions to the GPU is not recommended, as the upload/download overhead will likely be larger than the performance gain from parallel execution.

The key to good performance is to keep the data on the GPU as long as possible, chaining multiple operations together without returning to the CPU.

`cv::metal::MetalMat`
---------------------

`cv::Mat` objects are stored only in system memory. For getting an OpenCV matrix to the GPU you'll need to use its Metal counterpart: `cv::metal::MetalMat`. It works similarly to `cv::Mat` but is limited to 2D data. To upload a `cv::Mat` object to the GPU, you create a `MetalMat` from it. To download, you call the `download` method.
@code{.cpp}
// Assume src_host is a cv::Mat loaded from a file
cv::Mat src_host = cv::imread("my_image.png");

// Create a MetalMat and upload the data
cv::metal::MetalMat src_device(src_host);

// ... perform GPU operations ...
cv::metal::MetalMat dst_device;
cv::metal::GaussianBlur(src_device, dst_device, cv::Size(7, 7), 2.0);

// Download the result back to a cv::Mat
cv::Mat dst_host;
dst_device.download(dst_host);

// Now dst_host can be displayed or saved
cv::imshow("Result", dst_host);
@endcode

`cv::metal::Stream` for Chained Operations
------------------------------------------

Every call to a `cv::metal` function like `GaussianBlur` involves creating a command buffer, encoding the command, committing it to the GPU, and waiting for it to finish. This can be inefficient if you are performing many operations in a sequence.

To avoid this overhead, you can use a `cv::metal::Stream`. A `Stream` object encapsulates a `MTLCommandBuffer`. When you pass a `Stream` to a `cv::metal` function, the function will encode its work onto the stream's command buffer but will *not* commit it. This allows you to chain multiple operations together. You are then responsible for committing the stream and waiting for completion.

Here's how to build a simple pipeline:
@code{.cpp}
// Assume src_host is a cv::Mat
cv::Mat src_host = cv::imread("my_image.png");

// Upload to the GPU
cv::metal::MetalMat src_device(src_host);

// Create intermediate and final MetalMats
cv::metal::MetalMat resized_device;
cv::metal::MetalMat blurred_device;
cv::metal::MetalMat sobel_device;

// Create a stream
cv::metal::Stream stream;

// Chain operations on the stream
cv::metal::resize(src_device, resized_device, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR, stream);
cv::metal::GaussianBlur(resized_device, blurred_device, cv::Size(5, 5), 1.5, stream);
cv::metal::Sobel(blurred_device, sobel_device, CV_32F, 1, 0, 3, stream);

// Commit the command buffer and wait for the GPU to finish
stream.commitAndWait();

// Download the final result
cv::Mat sobel_host;
sobel_device.download(sobel_host);
@endcode

By using a `Stream`, all three operations (`resize`, `GaussianBlur`, `Sobel`) are sent to the GPU as a single batch of work, which is much more efficient than sending them one by one.

Interoperability with Swift/Objective-C
---------------------------------------

The Metal backend is designed to integrate with existing Metal applications written in Swift or Objective-C. This is achieved by allowing `MetalMat` and `Stream` to wrap existing Metal objects.

- **Wrapping an `id<MTLTexture>`**: You can create a `MetalMat` from an existing `id<MTLTexture>` without copying the underlying data. The `MetalMat` will not take ownership of the texture.
@code{.cpp}
// In an Objective-C++ file (.mm)
// Assume 'myAppTexture' is an id<MTLTexture> from your Swift/Obj-C code
id<MTLTexture> myAppTexture = ...;

// Create a non-owning MetalMat wrapper
// Note: A __bridge cast is needed to convert the Obj-C pointer to the 'id' (void*) type
cv::metal::MetalMat mat_wrapper((__bridge id)myAppTexture);

// Now 'mat_wrapper' can be used in OpenCV functions
cv::metal::MetalMat dst_mat;
cv::metal::GaussianBlur(mat_wrapper, dst_mat, ...);
@endcode

- **Wrapping an `id<MTLCommandBuffer>`**: Similarly, you can wrap an existing command buffer in a `cv::metal::Stream`. This allows OpenCV to encode its commands into your application's existing render or compute pass.
@code{.cpp}
// Assume 'myAppCommandBuffer' is an id<MTLCommandBuffer>
id<MTLCommandBuffer> myAppCommandBuffer = ...;

// Create a non-owning Stream wrapper
cv::metal::Stream stream_wrapper((__bridge id)myAppCommandBuffer);

// Pass the stream to OpenCV functions
cv::metal::GaussianBlur(src_mat, dst_mat, ..., stream_wrapper);

// Your application is responsible for committing the command buffer.
// stream_wrapper.commitAndWait() will have no effect.
[myAppCommandBuffer commit];
[myAppCommandBuffer waitUntilCompleted];
@endcode

This interoperability requires an Objective-C++ bridging layer in your application, as Swift cannot call C++ directly.

Example Result
--------------

Here's an example of an image processing pipeline (resize, blur, Sobel) running on the Metal backend.

![](images/metal-basics.png)

Conclusion
----------

By using `cv::metal::MetalMat` to manage GPU memory and `cv::metal::Stream` to chain operations, you can build efficient, hardware-accelerated image processing pipelines on Apple platforms. The ability to wrap existing `MTLTexture` and `MTLCommandBuffer` objects allows for seamless integration into larger Swift or Objective-C applications, combining the power of OpenCV with your custom Metal code.

@endcond