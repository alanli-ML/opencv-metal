#include <metal_stdlib>
using namespace metal;

// OpenCV-compatible multiply kernel for 8UC4
// Implements: min(255, a * b) for each channel
// Note: 8UC4 uses MTLPixelFormatBGRA8Unorm with normalized values [0.0, 1.0]
kernel void multiply_8UC4_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                 texture2d<float, access::read> src2 [[texture(1)]],
                                 texture2d<float, access::write> dst [[texture(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    
    float4 pixel1 = src1.read(gid);
    float4 pixel2 = src2.read(gid);
    
    // Convert from normalized [0.0, 1.0] to integer [0, 255] range
    uint4 int_pixel1 = uint4(pixel1 * 255.0f + 0.5f);
    uint4 int_pixel2 = uint4(pixel2 * 255.0f + 0.5f);
    
    // OpenCV multiply: min(255, a * b) for each channel 
    uint4 int_result;
    int_result.r = min(255u, int_pixel1.r * int_pixel2.r);
    int_result.g = min(255u, int_pixel1.g * int_pixel2.g);
    int_result.b = min(255u, int_pixel1.b * int_pixel2.b);
    int_result.a = min(255u, int_pixel1.a * int_pixel2.a);
    
    // Convert back to normalized [0.0, 1.0] range
    float4 result = float4(int_result) / 255.0f;
    
    dst.write(result, gid);
}

// OpenCV-compatible divide kernel for 8UC4  
// Implements: a / b (integer division) for each channel
// Note: 8UC4 uses MTLPixelFormatBGRA8Unorm with normalized values [0.0, 1.0]
kernel void divide_8UC4_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                               texture2d<float, access::read> src2 [[texture(1)]],
                               texture2d<float, access::write> dst [[texture(2)]],
                               uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    
    float4 pixel1 = src1.read(gid);
    float4 pixel2 = src2.read(gid);
    
    // Convert from normalized [0.0, 1.0] to integer [0, 255] range
    uint4 int_pixel1 = uint4(pixel1 * 255.0f + 0.5f);
    uint4 int_pixel2 = uint4(pixel2 * 255.0f + 0.5f);
    
    // OpenCV divide: a / b (integer division) for each channel, handle divide by zero
    uint4 int_result;
    int_result.r = (int_pixel2.r == 0) ? 0 : int_pixel1.r / int_pixel2.r;
    int_result.g = (int_pixel2.g == 0) ? 0 : int_pixel1.g / int_pixel2.g;
    int_result.b = (int_pixel2.b == 0) ? 0 : int_pixel1.b / int_pixel2.b;
    int_result.a = (int_pixel2.a == 0) ? 0 : int_pixel1.a / int_pixel2.a;
    
    // Convert back to normalized [0.0, 1.0] range
    float4 result = float4(int_result) / 255.0f;
    
    dst.write(result, gid);
}

// OpenCV-compatible multiply kernel for 32FC1
kernel void multiply_32FC1_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                  texture2d<float, access::read> src2 [[texture(1)]],
                                  texture2d<float, access::write> dst [[texture(2)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    
    float4 pixel1 = src1.read(gid);
    float4 pixel2 = src2.read(gid);
    
    // For float, simple multiplication (no saturation needed)
    float4 result = pixel1 * pixel2;
    
    dst.write(result, gid);
}

// OpenCV-compatible divide kernel for 32FC1
kernel void divide_32FC1_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                texture2d<float, access::read> src2 [[texture(1)]],
                                texture2d<float, access::write> dst [[texture(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    
    float4 pixel1 = src1.read(gid);
    float4 pixel2 = src2.read(gid);
    
    // For float division, handle divide by zero appropriately
    float4 result;
    result.r = (pixel2.r == 0.0f) ? 0.0f : pixel1.r / pixel2.r;
    result.g = (pixel2.g == 0.0f) ? 0.0f : pixel1.g / pixel2.g; 
    result.b = (pixel2.b == 0.0f) ? 0.0f : pixel1.b / pixel2.b;
    result.a = (pixel2.a == 0.0f) ? 0.0f : pixel1.a / pixel2.a;
    
    dst.write(result, gid);
}

// OpenCV-compatible multiply kernel for 32FC4
kernel void multiply_32FC4_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                  texture2d<float, access::read> src2 [[texture(1)]],
                                  texture2d<float, access::write> dst [[texture(2)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    
    float4 pixel1 = src1.read(gid);
    float4 pixel2 = src2.read(gid);
    
    // For float, simple multiplication
    float4 result = pixel1 * pixel2;
    
    dst.write(result, gid);
}

// OpenCV-compatible divide kernel for 32FC4
kernel void divide_32FC4_opencv(texture2d<float, access::read> src1 [[texture(0)]],
                                texture2d<float, access::read> src2 [[texture(1)]],
                                texture2d<float, access::write> dst [[texture(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
        return;
    }
    
    float4 pixel1 = src1.read(gid);
    float4 pixel2 = src2.read(gid);
    
    // For float division, handle divide by zero
    float4 result;
    result.r = (pixel2.r == 0.0f) ? 0.0f : pixel1.r / pixel2.r;
    result.g = (pixel2.g == 0.0f) ? 0.0f : pixel1.g / pixel2.g;
    result.b = (pixel2.b == 0.0f) ? 0.0f : pixel1.b / pixel2.b;
    result.a = (pixel2.a == 0.0f) ? 0.0f : pixel1.a / pixel2.a;
    
    dst.write(result, gid);
} 