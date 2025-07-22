Metal Module Introduction {#metal_intro}
========================

General Information
-------------------

The OpenCV Metal module is a set of classes and functions to utilize Apple's Metal framework for hardware-accelerated computation. It is implemented using Metal and Metal Performance Shaders (MPS) and supports Apple GPUs on macOS and iOS. The module includes utility functions, low-level vision primitives, and high-level algorithms.

The Metal module is designed as a host-level API, similar to the CUDA backend. If you have a pre-compiled OpenCV with Metal support, you are not required to have any Metal development tools installed. The module is intended for ease of use and does not require deep knowledge of Metal, though such knowledge is beneficial for advanced use cases and achieving maximum performance.

To enable Metal support, configure OpenCV using CMake with `WITH_METAL=ON`. This flag is enabled by default on Apple platforms.

`cv::metal::MetalMat`
---------------------

The primary data structure for GPU memory is `cv::metal::MetalMat`. It is analogous to `cv::Mat` and `cv::cuda::GpuMat` but is backed by an `id<MTLTexture>` object. It uses a reference-counted implementation for efficient shallow copies when managing its own memory.

Data transfer between CPU (`cv::Mat`) and GPU (`cv::metal::MetalMat`) is explicit:
@code{.cpp}
// Create a cv::Mat on the CPU
cv::Mat src_host = cv::imread("image.png", cv::IMREAD_COLOR);

// Upload to the GPU
// Note: 3-channel BGR images are automatically converted to 4-channel BGRA
// as this is more commonly supported by Metal textures.
cv::metal::MetalMat src_device(src_host);

// ... perform GPU operations ...
cv::metal::MetalMat dst_device;
// (e.g., cv::metal::GaussianBlur(src_device, dst_device, ...))

// Download the result back to the CPU
cv::Mat dst_host;
dst_device.download(dst_host);
@endcode

A note on Unified Memory: On Apple Silicon devices featuring a Unified Memory Architecture (UMA), the `MetalMat(const Mat&)` constructor and `upload()` method still perform a data copy. They use Metal's `[MTLTexture replaceRegion:withBytes:bytesPerRow:]` method, which copies data from the source `cv::Mat` buffer into the texture's backing store. While a zero-copy approach is possible on UMA platforms using `MTLStorageModeShared`, the current implementation prioritizes a consistent API that works across both integrated (Apple Silicon) and discrete (Intel-based Macs) graphics architectures.

For interoperability with external Metal pipelines (e.g., in a Swift application), `MetalMat` can also wrap an existing `id<MTLTexture>` without taking ownership or copying data.

@code{.cpp}
// Assume 'myTexture' is an id<MTLTexture> from a Swift application
// This can be passed to an Objective-C++ wrapper.
id<MTLTexture> myTexture = ...;

// Wrap the existing texture in a MetalMat (non-owning)
cv::metal::MetalMat mat_from_swift((__bridge id)myTexture);

// Now use it in OpenCV Metal functions
cv::metal::MetalMat dst_device;
cv::metal::GaussianBlur(mat_from_swift, dst_device, ...);
@endcode

For interoperability with external Metal pipelines (e.g., in a Swift application), `MetalMat` can also wrap an existing `id<MTLTexture>` without taking ownership or copying data.

@code{.cpp}
// Assume 'myTexture' is an id<MTLTexture> from a Swift application
// This can be passed to an Objective-C++ wrapper.
id<MTLTexture> myTexture = ...;

// Wrap the existing texture in a MetalMat (non-owning)
cv::metal::MetalMat mat_from_swift((__bridge id)myTexture);

// Now use it in OpenCV Metal functions
cv::metal::MetalMat dst_device;
cv::metal::GaussianBlur(mat_from_swift, dst_device, ...);
@endcode

`cv::metal::Stream`
-------------------

By default, all `cv::metal` functions are synchronous. They create a command buffer, encode a command, commit it, and wait for the GPU to complete the operation. This is simple but can be inefficient for pipelines with multiple sequential operations.

To improve performance, `cv::metal::Stream` allows for asynchronous, chained execution. A `Stream` object encapsulates a `MTLCommandBuffer`. When a stream is passed to a `cv::metal` function, the function encodes its work onto the stream's command buffer but does not commit it. This allows multiple operations to be batched into a single submission to the GPU.

@code{.cpp}
cv::metal::MetalMat src_device(...);
cv::metal::MetalMat resized_device, blurred_device, sobel_device;

// Create a stream
cv::metal::Stream stream;

// Enqueue a sequence of operations on the same command buffer
cv::metal::resize(src_device, resized_device, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR, stream);
cv::metal::GaussianBlur(resized_device, blurred_device, cv::Size(5, 5), 1.5, stream);
cv::metal::Sobel(blurred_device, sobel_device, CV_32F, 1, 0, 3, stream);

// Commit the batched commands and wait for completion
stream.commitAndWait();
@endcode

To integrate with an external Metal pipeline, `Stream` can also wrap an existing `id<MTLCommandBuffer>`. When a `Stream` is created this way, it does not own the command buffer, and methods like `commitAndWait()` will have no effect. The responsibility for committing the buffer remains with the calling application.

@code{.cpp}
// Assume 'myCommandBuffer' is an id<MTLCommandBuffer> from a Swift application
id<MTLCommandBuffer> myCommandBuffer = ...;

// Wrap the command buffer in a Stream
cv::metal::Stream stream((__bridge id)myCommandBuffer);

// Use this stream to encode OpenCV operations into the external pipeline
cv::metal::GaussianBlur(src, dst, ..., stream);
// The Swift app is responsible for committing myCommandBuffer
@endcode

Using a stream significantly reduces the overhead of interacting with the GPU, leading to better performance for complex pipelines.

@code{.cpp}
// Assume 'myCommandBuffer' is an id<MTLCommandBuffer> from a Swift application
id<MTLCommandBuffer> myCommandBuffer = ...;

// Wrap the command buffer in a Stream
cv::metal::Stream stream((__bridge id)myCommandBuffer);

// Use this stream to encode OpenCV operations into the external pipeline
cv::metal::GaussianBlur(src, dst, ..., stream);
// The Swift app is responsible for committing myCommandBuffer
@endcode

Using a stream significantly reduces the overhead of interacting with the GPU, leading to better performance for complex pipelines.

Implemented Functions (MVP)
---------------------------

The initial version of the Metal backend includes the following functions, primarily leveraging the Metal Performance Shaders framework:

### Core Module (`opencv2/core/metal.hpp`)
-   `cv::metal::add`
-   `cv::metal::subtract`
-   `cv::metal::multiply`
-   `cv::metal::divide`

### Imgproc Module (`opencv2/imgproc/metal.hpp`)
-   `cv::metal::GaussianBlur`
-   `cv::metal::Sobel`
-   `cv::metal::resize`

Future Work
-----------

The Metal backend is an ongoing effort. Future work will focus on expanding its capabilities:

1.  **More `imgproc` Functions**: Implement other common image processing functions like `morphologyEx`, `Canny`, `HoughLines`, and a wider range of color conversions.
2.  **Custom Kernels**: For operations not available in the Metal Performance Shaders framework (e.g., bitwise operations, custom filters), custom kernels will be written in the Metal Shading Language (MSL).
3.  **Expanded Module Support**: Add Metal-accelerated implementations to other modules, such as `features2d` and `dnn`.
4.  **Efficient ROI Support**: The current ROI implementation in `MetalMat` creates a copy. A more efficient, zero-copy implementation using texture views or advanced MPS features will be investigated.
5.  **`UMat` Integration**: Improve the integration with `UMat` to allow for more seamless interoperability between different OpenCV backends (CPU, OpenCL, Metal).