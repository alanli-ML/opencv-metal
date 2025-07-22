@cond CUDA_MODULES
GPU-Accelerated Computer Vision {#tutorial_table_of_content_gpu}
=============================================

Squeeze out every little computation power from your system by using the power of your video card to
run the OpenCV algorithms.

CUDA Backend (cuda module)
--------------------------

-   @subpage tutorial_gpu_basics_similarity

    *Languages:* C++

    *Compatibility:* > OpenCV 2.0

    *Author:* Bernát Gábor

    This will give a good grasp on how to approach coding on the GPU module, once you already know
    how to handle the other modules. As a test case it will port the similarity methods from the
    tutorial @ref tutorial_video_input_psnr_ssim to the GPU.

-   @subpage tutorial_gpu_thrust_interop

    *Languages:* C++

    *Compatibility:* >= OpenCV 3.0

    This tutorial will show you how to wrap a GpuMat into a thrust iterator in order to be able to
    use the functions in the thrust library.
@endcond

@cond HAVE_METAL
Metal Backend
-------------

-   @subpage tutorial_metal_basics

    *Languages:* C++

    *Compatibility:* > OpenCV 4.x

    This tutorial gives an introduction to the Metal backend, explaining how to use `cv::metal::MetalMat`
    and `cv::metal::Stream` to build efficient, hardware-accelerated image processing pipelines on
    Apple platforms.
@endcond
