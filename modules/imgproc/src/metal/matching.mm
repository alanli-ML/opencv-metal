// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_precomp.hpp"

namespace cv { namespace metal {

//==============================================================================
// Template Matching Shader Sources
//==============================================================================

static const char* matchTemplateShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

// OpenCV-compatible template matching kernels
kernel void matchTemplate_8UC1_CCORR_opencv(texture2d<float, access::sample> imageTexture [[texture(0)]],
                                            texture2d<float, access::sample> templateTexture [[texture(1)]],
                                            texture2d<float, access::write> resultTexture [[texture(2)]],
                                            constant int& template_width [[buffer(0)]],
                                            constant int& template_height [[buffer(1)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= resultTexture.get_width() || gid.y >= resultTexture.get_height()) return;

    float sum = 0.0f;

    for (int j = 0; j < template_height; ++j) {
        for (int i = 0; i < template_width; ++i) {
            // Read from image at offset position
            float2 image_coord = float2(gid) + float2(i, j);
            float image_pixel = imageTexture.sample(s, image_coord).r;
            
            // Read from template
            float2 template_coord = float2(i, j);
            float template_pixel = templateTexture.sample(s, template_coord).r;
            
            // Convert to integer range for OpenCV semantics
            uint image_int = uint(image_pixel * 255.0f + 0.5f);
            uint template_int = uint(template_pixel * 255.0f + 0.5f);
            
            // Cross-correlation (not convolution)
            sum += float(image_int) * float(template_int);
        }
    }

    // Normalize and write result
    float result = sum / (255.0f * 255.0f); // Normalize to reasonable range
    resultTexture.write(float4(result, 0, 0, 1), gid);
}

kernel void matchTemplate_8UC1_CCORR_NORMED_opencv(texture2d<float, access::sample> imageTexture [[texture(0)]],
                                                   texture2d<float, access::sample> templateTexture [[texture(1)]],
                                                   texture2d<float, access::write> resultTexture [[texture(2)]],
                                                   constant int& template_width [[buffer(0)]],
                                                   constant int& template_height [[buffer(1)]],
                                                   uint2 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    if (gid.x >= resultTexture.get_width() || gid.y >= resultTexture.get_height()) return;

    float sum = 0.0f;
    float image_sum2 = 0.0f;
    float template_sum2 = 0.0f;

    for (int j = 0; j < template_height; ++j) {
        for (int i = 0; i < template_width; ++i) {
            // Read from image at offset position
            float2 image_coord = float2(gid) + float2(i, j);
            float image_pixel = imageTexture.sample(s, image_coord).r;
            
            // Read from template
            float2 template_coord = float2(i, j);
            float template_pixel = templateTexture.sample(s, template_coord).r;
            
            // Convert to integer range for OpenCV semantics
            uint image_int = uint(image_pixel * 255.0f + 0.5f);
            uint template_int = uint(template_pixel * 255.0f + 0.5f);
            
            float image_f = float(image_int);
            float template_f = float(template_int);
            
            // Cross-correlation
            sum += image_f * template_f;
            image_sum2 += image_f * image_f;
            template_sum2 += template_f * template_f;
        }
    }

    // Normalized correlation
    float norm = sqrt(image_sum2 * template_sum2);
    float result = (norm > 0.0f) ? (sum / norm) : 0.0f;
    
    resultTexture.write(float4(result, 0, 0, 1), gid);
}
)";

//==============================================================================
// MatchTemplate Implementation
//==============================================================================

void matchTemplate(const MetalMat& image, const MetalMat& templ, MetalMat& result, int method, InputArray mask, Stream& stream)
{
    CV_Assert(image.depth() == CV_8U || image.depth() == CV_32F);
    CV_Assert(image.type() == templ.type());
    CV_Assert(mask.empty()); // Masks not supported in initial implementation
    CV_Assert(image.size().width >= templ.size().width && image.size().height >= templ.size().height);
    CV_Assert(templ.cols() % 2 == 1 && templ.rows() % 2 == 1); // MPS requires odd dimensions
    
    // Calculate result size
    cv::Size result_size(image.cols() - templ.cols() + 1, image.rows() - templ.rows() + 1);
    result.create(result_size, CV_32FC1);

    id<MTLCommandBuffer> commandBuffer = StreamAccessor::getCommandBuffer(stream);
    CV_Assert(commandBuffer != nil);

    @autoreleasepool {
        switch (method) {
            case TM_CCORR:
            case TM_CCORR_NORMED: {
                // For template matching, we need to flip the template for correlation
                // This is a simplified implementation - full template matching requires
                // more sophisticated correlation and normalization
                Mat templ_cpu;
                templ.download(templ_cpu);
                
                // Convert template to weights (simplified approach)
                if (templ_cpu.channels() == 1) {
                    Mat templ_f;
                    templ_cpu.convertTo(templ_f, CV_32F);
                    
                    // Flip template for convolution (correlation = convolution with flipped kernel)
                    Mat templ_flipped;
                    flip(templ_f, templ_flipped, -1);
                    
                    // Create new convolution with template weights
                    const float* weights = templ_flipped.ptr<float>();
                    MPSImageConvolution *templateConv = [[MPSImageConvolution alloc]
                        initWithDevice:MetalContext::getInstance().device
                        kernelWidth:templ.cols()
                        kernelHeight:templ.rows()
                        weights:weights];
                    CV_Assert(templateConv != nil);
                    templateConv.edgeMode = MPSImageEdgeModeZero;
                    
                    [templateConv encodeToCommandBuffer:commandBuffer
                                           sourceTexture:image.texture()
                                      destinationTexture:result.texture()];
                } else {
                    CV_Error(Error::StsBadArg, "Multi-channel template matching not implemented yet");
                }
                break;
            }
            case TM_SQDIFF:
            case TM_SQDIFF_NORMED: {
                // Squared difference methods would require custom Metal kernels
                // For now, fall back to error
                CV_Error(Error::StsBadArg, "TM_SQDIFF methods not implemented yet - use TM_CCORR");
                break;
            }
            case TM_CCOEFF:
            case TM_CCOEFF_NORMED: {
                // Correlation coefficient methods require mean subtraction
                // Would need custom implementation
                CV_Error(Error::StsBadArg, "TM_CCOEFF methods not implemented yet - use TM_CCORR");
                break;
            }
            default:
                CV_Error(Error::StsBadArg, "Unknown template matching method");
        }
    }
}

void matchTemplate(const MetalMat& image, const MetalMat& templ, MetalMat& result, int method, InputArray mask)
{
    Stream stream;
    matchTemplate(image, templ, result, method, mask, stream);
    stream.commitAndWait();
}

}} // cv::metal 