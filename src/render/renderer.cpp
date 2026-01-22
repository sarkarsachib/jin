#include "renderer.h"
#include "../utils/math.h"
#include "../autodiff/tape.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

namespace jin {
namespace render {

Renderer::Renderer() {
    config_ = RenderConfig{};
}

Renderer::Renderer(const RenderConfig& config) : config_(config) {}

Tensor Renderer::render_frame(const Tensor& frame, const RenderConfig& config) {
    RenderConfig original_config = config_;
    config_ = config;
    
    Tensor rendered = frame;
    
    // Apply colorspace conversion
    if (config.output_format == RenderConfig::Format::YUV420 || 
        config.output_format == RenderConfig::Format::YUV444) {
        rendered = convert_colorspace(rendered, RenderConfig::ColorSpace::SRGB, 
                                   RenderConfig::ColorSpace::LINEAR);
    }
    
    // Apply tone mapping if needed
    if (config.output_format == RenderConfig::Format::RGB ||
        config.output_format == RenderConfig::Format::RGBA) {
        rendered = apply_tonemapping(rendered, "reinhard");
    }
    
    // Convert to target format
    rendered = convert_format(rendered, config.output_format);
    
    // Apply bit depth conversion
    if (config.bit_depth == RenderConfig::BitDepth::BIT_8) {
        rendered = apply_dithering(rendered, 8);
    } else if (config.bit_depth == RenderConfig::BitDepth::BIT_10) {
        rendered = apply_dithering(rendered, 10);
    }
    
    // Restore original config
    config_ = original_config;
    
    return rendered;
}

std::vector<Tensor> Renderer::render_sequence(const std::vector<Tensor>& frames, const RenderConfig& config) {
    std::vector<Tensor> rendered_frames;
    rendered_frames.reserve(frames.size());
    
    for (const auto& frame : frames) {
        rendered_frames.push_back(render_frame(frame, config));
    }
    
    return rendered_frames;
}

Tensor Renderer::convert_colorspace(const Tensor& frame, 
                                  RenderConfig::ColorSpace from_space,
                                  RenderConfig::ColorSpace to_space) {
    if (from_space == to_space) {
        return frame;
    }
    
    Tensor converted = frame;
    
    // RGB ↔ YUV conversion
    if ((from_space == RenderConfig::ColorSpace::SRGB && to_space == RenderConfig::ColorSpace::LINEAR) ||
        (from_space == RenderConfig::ColorSpace::LINEAR && to_space == RenderConfig::ColorSpace::SRGB)) {
        
        if (from_space == RenderConfig::ColorSpace::SRGB) {
            converted = srgb_to_linear(frame);
        } else {
            converted = linear_to_srgb(frame);
        }
    }
    
    return converted;
}

Tensor Renderer::apply_tonemapping(const Tensor& frame, const std::string& method) {
    if (method == "reinhard") {
        // Reinhard tone mapping: L' = L / (1 + L)
        Tensor normalized = frame / (frame + 1.0f);
        return normalized;
    } else if (method == "aces") {
        // ACES approximate tone mapping
        Tensor a = frame * (2.51f * frame + 0.03f) / (frame * (2.43f * frame + 0.59f) + 0.14f);
        return a.clamp(0.0f, 1.0f);
    } else if (method == "linear") {
        // Linear tone mapping (no change)
        return frame;
    }
    
    return frame;
}

QualityMetrics Renderer::compute_render_quality(const Tensor& original, const Tensor& rendered) {
    QualityMetrics metrics;
    
    metrics.psnr = compute_psnr(original, rendered);
    metrics.ssim = compute_ssim(original, rendered);
    
    // For single frame, avg and min are the same
    metrics.psnr_avg = metrics.psnr;
    metrics.psnr_min = metrics.psnr;
    metrics.ssim_avg = metrics.ssim;
    metrics.ssim_min = metrics.ssim;
    
    return metrics;
}

float Renderer::compute_psnr(const Tensor& original, const Tensor& rendered) {
    // Compute MSE
    Tensor diff = original - rendered;
    Tensor squared_diff = diff * diff;
    float mse = squared_diff.mean().item<float>();
    
    if (mse < 1e-10f) {
        return std::numeric_limits<float>::infinity();
    }
    
    // PSNR = 10 * log10(255^2 / MSE)
    float psnr = 10.0f * std::log10((255.0f * 255.0f) / mse);
    return psnr;
}

float Renderer::compute_ssim(const Tensor& frame_a, const Tensor& frame_b) {
    // Compute SSIM components
    Tensor luminance = compute_luminance(frame_a);
    Tensor luminance_b = compute_luminance(frame_b);
    
    Tensor contrast = compute_contrast(frame_a);
    Tensor contrast_b = compute_contrast(frame_b);
    
    Tensor structure = compute_structure(frame_a, frame_b);
    
    // SSIM formula
    Tensor numerator = (2 * luminance * luminance_b + 0.01f) * 
                      (2 * structure + 0.03f);
    Tensor denominator = (luminance * luminance + luminance_b * luminance_b + 0.01f) * 
                        (contrast + contrast_b + 0.03f);
    
    Tensor ssim_map = numerator / denominator;
    return ssim_map.mean().item<float>();
}

void Renderer::set_resolution(int width, int height) {
    config_.resolution = {width, height};
}

Tensor Renderer::resize_frame(const Tensor& frame, int target_width, int target_height) {
    int current_width = frame.shape()[frame.shape().size() - 1];
    int current_height = frame.shape()[frame.shape().size() - 2];
    
    if (current_width == target_width && current_height == target_height) {
        return frame;
    }
    
    if (current_width < target_width || current_height < target_height) {
        return upsample_frame(frame, target_width, target_height);
    } else {
        return downsample_frame(frame, target_width, target_height);
    }
}

Tensor Renderer::convert_format(const Tensor& frame, RenderConfig::Format target_format) {
    // This is a simplified implementation
    // In a full implementation, this would handle proper channel reordering and format conversion
    
    if (target_format == RenderConfig::Format::RGB) {
        // Convert to 3-channel RGB
        if (frame.shape().size() == 4) { // Assume NCHW or NHWC
            return frame.get_slice(-1, 3); // Take first 3 channels
        }
    } else if (target_format == RenderConfig::Format::RGBA) {
        // Convert to 4-channel RGBA with alpha = 1
        if (frame.shape().size() == 4) {
            Tensor rgb = frame.get_slice(-1, 3);
            Tensor alpha = Tensor::ones_like(rgb.get_slice(0, 1));
            return concat({rgb, alpha}, -1);
        }
    } else if (target_format == RenderConfig::Format::YUV420 || 
               target_format == RenderConfig::Format::YUV444) {
        // RGB to YUV conversion
        Tensor yuv_matrix = get_rgb_to_yuv_matrix();
        return apply_color_matrix(frame, yuv_matrix);
    }
    
    return frame;
}

Tensor Renderer::get_rgb_to_yuv_matrix() {
    // Standard RGB to YUV conversion matrix
    Tensor matrix = Tensor::from_vector({
        0.299f, -0.168736f, 0.5f,
        0.587f, -0.331264f, -0.418688f,
        0.114f, 0.5f, -0.081312f
    }, {3, 3});
    
    return matrix;
}

Tensor Renderer::get_yuv_to_rgb_matrix() {
    // Standard YUV to RGB conversion matrix
    Tensor matrix = Tensor::from_vector({
        1.0f, 1.0f, 1.0f,
        0.0f, -0.344136f, 1.772f,
        1.402f, -0.714136f, 0.0f
    }, {3, 3});
    
    return matrix;
}

Tensor Renderer::apply_color_matrix(const Tensor& frame, const Tensor& matrix) {
    // Apply color transformation matrix to frame
    // This is a simplified version - full implementation would handle proper matrix multiplication
    return frame;
}

Tensor Renderer::apply_dithering(const Tensor& frame, int target_bit_depth) {
    int max_levels = (1 << target_bit_depth) - 1;
    
    // Simple ordered dithering
    Tensor dithered = frame * max_levels;
    dithered = dithered.floor();
    dithered = dithered / max_levels;
    
    return dithered;
}

Tensor Renderer::upsample_frame(const Tensor& frame, int target_width, int target_height) {
    int current_width = frame.shape()[frame.shape().size() - 1];
    int current_height = frame.shape()[frame.shape().size() - 2];
    
    if (config_.upsampling_filter == RenderConfig::UpsamplingFilter::NEAREST) {
        // Nearest neighbor upsampling
        Tensor upsampled = Tensor::zeros({frame.shape()[0], frame.shape()[1], target_height, target_width});
        
        for (int y = 0; y < target_height; ++y) {
            int src_y = y * current_height / target_height;
            for (int x = 0; x < target_width; ++x) {
                int src_x = x * current_width / target_width;
                upsampled.get_slice({0, 0, y, x}, {1, frame.shape()[1], y + 1, x + 1}) = 
                    frame.get_slice({0, 0, src_y, src_x}, {1, frame.shape()[1], src_y + 1, src_x + 1});
            }
        }
        
        return upsampled;
    } else if (config_.upsampling_filter == RenderConfig::UpsamplingFilter::BICUBIC) {
        // Bicubic upsampling (simplified)
        // For simplicity, using bilinear interpolation
        return upsample_frame(frame, target_width, target_height);
    }
    
    // Default to nearest neighbor
    return upsample_frame(frame, target_width, target_height);
}

Tensor Renderer::downsample_frame(const Tensor& frame, int target_width, int target_height) {
    int current_width = frame.shape()[frame.shape().size() - 1];
    int current_height = frame.shape()[frame.shape().size() - 2];
    
    // Area averaging for downsampling
    Tensor downsampled = Tensor::zeros({frame.shape()[0], frame.shape()[1], target_height, target_width});
    
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            int src_y_start = y * current_height / target_height;
            int src_y_end = (y + 1) * current_height / target_height;
            int src_x_start = x * current_width / target_width;
            int src_x_end = (x + 1) * current_width / target_width;
            
            Tensor region = frame.get_slice(
                {0, 0, src_y_start, src_x_start}, 
                {1, frame.shape()[1], src_y_end, src_x_end}
            );
            
            float avg = region.mean().item<float>();
            downsampled.set_item(avg, {0, 0, y, x});
        }
    }
    
    return downsampled;
}

Tensor Renderer::compute_luminance(const Tensor& frame) {
    // Compute luminance using standard coefficients
    Tensor luminance = frame.get_slice(-1, 1) * LUMINANCE_COEFF_R +
                     frame.get_slice(-1, 2) * LUMINANCE_COEFF_G +
                     frame.get_slice(-1, 3) * LUMINANCE_COEFF_B;
    return luminance;
}

Tensor Renderer::compute_contrast(const Tensor& frame) {
    // Compute local contrast using standard deviation
    Tensor mean = frame.mean();
    Tensor variance = ((frame - mean) * (frame - mean)).mean();
    Tensor contrast = variance.sqrt();
    return contrast;
}

Tensor Renderer::compute_structure(const Tensor& frame_a, const Tensor& frame_b) {
    // Compute structure component
    Tensor mean_a = frame_a.mean();
    Tensor mean_b = frame_b.mean();
    
    Tensor numerator = (frame_a - mean_a) * (frame_b - mean_b);
    Tensor denominator_a = ((frame_a - mean_a) * (frame_a - mean_a)).mean();
    Tensor denominator_b = ((frame_b - mean_b) * (frame_b - mean_b)).mean();
    
    Tensor structure = numerator / (denominator_a * denominator_b).sqrt();
    return structure;
}

Tensor Renderer::srgb_to_linear(const Tensor& frame) {
    // Convert sRGB to linear color space
    Tensor linear = frame.clone();
    
    // Apply inverse gamma correction
    linear = (linear <= 0.04045f) ? (linear / 12.92f) : 
             pow((linear + 0.055f) / 1.055f, 2.4f);
    
    return linear;
}

Tensor Renderer::linear_to_srgb(const Tensor& frame) {
    // Convert linear to sRGB color space
    Tensor srgb = frame.clone();
    
    // Apply gamma correction
    srgb = (srgb <= 0.0031308f) ? (srgb * 12.92f) : 
           (1.055f * pow(srgb, 1.0f / 2.4f) - 0.055f);
    
    return srgb;
}

Tensor Renderer::apply_gamma_correction(const Tensor& frame, float gamma) {
    return pow(frame, 1.0f / gamma);
}

Tensor Renderer::compute_gaussian_window(int width, int height, float sigma) {
    // Create 2D Gaussian window for SSIM computation
    Tensor window = Tensor::zeros({height, width});
    
    float sigma_squared = sigma * sigma;
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float distance_squared = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y);
            window.set_item(std::exp(-distance_squared / (2 * sigma_squared)), {y, x});
        }
    }
    
    return window;
}

float Renderer::clamp_to_range(float value, float min_val, float max_val) {
    return std::max(min_val, std::min(value, max_val));
}

std::vector<uint8_t> Renderer::encode_jpeg(const Tensor& frame, int quality) {
    // Simplified JPEG encoding - in a full implementation this would use libjpeg
    // For now, return empty vector as placeholder
    return std::vector<uint8_t>();
}

std::vector<uint8_t> Renderer::encode_png(const Tensor& frame) {
    // Simplified PNG encoding - in a full implementation this would use libpng
    // For now, return empty vector as placeholder
    return std::vector<uint8_t>();
}

Tensor Renderer::decode_jpeg(const std::vector<uint8_t>& data) {
    // Simplified JPEG decoding - placeholder
    return Tensor::zeros({1, 3, 256, 256});
}

Tensor Renderer::decode_png(const std::vector<uint8_t>& data) {
    // Simplified PNG decoding - placeholder
    return Tensor::zeros({1, 3, 256, 256});
}

} // namespace render
} // namespace jin