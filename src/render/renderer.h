#pragma once

#include "../core/tensor.h"
#include <string>
#include <vector>
#include <map>

namespace jin {
namespace render {

// Render configuration
struct RenderConfig {
    std::pair<int, int> resolution;          // (width, height) - 256x256 to 4K
    enum class Format {
        RGB, RGBA, YUV420, YUV444
    } output_format;
    
    enum class BitDepth {
        BIT_8, BIT_10, BIT_16
    } bit_depth;
    
    enum class Compression {
        NONE, JPEG_Q50, JPEG_Q75, JPEG_Q95, PNG, WEBP
    } compression;
    
    int quality;                             // 0-100 for lossy formats
    
    enum class ColorSpace {
        SRGB, LINEAR, DISPLAY_P3
    } color_space;
    
    enum class UpsamplingFilter {
        NEAREST, LINEAR, BICUBIC, LANCZOS
    } upsampling_filter;
    
    RenderConfig()
        : resolution({1920, 1080}), output_format(Format::RGB),
          bit_depth(BitDepth::BIT_8), compression(Compression::NONE),
          quality(95), color_space(ColorSpace::SRGB),
          upsampling_filter(UpsamplingFilter::BICUBIC) {}
};

// Quality metrics result
struct QualityMetrics {
    float psnr;      // Peak Signal-to-Noise Ratio (dB)
    float ssim;      // Structural Similarity Index [0, 1]
    float psnr_avg;  // Average PSNR across frames
    float psnr_min;  // Minimum PSNR across frames
    float ssim_avg;  // Average SSIM across frames
    float ssim_min;  // Minimum SSIM across frames
};

// Renderer class for frame processing and format conversion
class Renderer {
public:
    Renderer();
    explicit Renderer(const RenderConfig& config);
    
    // Frame rendering
    Tensor render_frame(const Tensor& frame, const RenderConfig& config);
    std::vector<Tensor> render_sequence(const std::vector<Tensor>& frames, const RenderConfig& config);
    
    // Colorspace conversion
    Tensor convert_colorspace(const Tensor& frame, 
                           RenderConfig::ColorSpace from_space,
                           RenderConfig::ColorSpace to_space);
    
    // Tone mapping
    Tensor apply_tonemapping(const Tensor& frame, const std::string& method);
    
    // Quality assessment
    QualityMetrics compute_render_quality(const Tensor& original, const Tensor& rendered);
    float compute_psnr(const Tensor& original, const Tensor& rendered);
    float compute_ssim(const Tensor& frame_a, const Tensor& frame_b);
    
    // Resolution handling
    void set_resolution(int width, int height);
    Tensor resize_frame(const Tensor& frame, int target_width, int target_height);
    
    // Format conversion
    Tensor convert_format(const Tensor& frame, RenderConfig::Format target_format);
    
    // Utility methods
    void set_config(const RenderConfig& config) { config_ = config; }
    const RenderConfig& get_config() const { return config_; }
    std::pair<int, int> get_resolution() const { return config_.resolution; }
    
    // Color space matrices and conversion
    Tensor get_rgb_to_yuv_matrix();
    Tensor get_yuv_to_rgb_matrix();
    Tensor apply_color_matrix(const Tensor& frame, const Tensor& matrix);
    
    // Dithering for bit depth reduction
    Tensor apply_dithering(const Tensor& frame, int target_bit_depth);
    
    // Compression helpers
    std::vector<uint8_t> encode_jpeg(const Tensor& frame, int quality);
    std::vector<uint8_t> encode_png(const Tensor& frame);
    Tensor decode_jpeg(const std::vector<uint8_t>& data);
    Tensor decode_png(const std::vector<uint8_t>& data);

private:
    RenderConfig config_;
    
    // Helper methods
    Tensor upsample_frame(const Tensor& frame, int target_width, int target_height);
    Tensor downsample_frame(const Tensor& frame, int target_width, int target_height);
    
    // SSIM computation helpers
    Tensor compute_luminance(const Tensor& frame);
    Tensor compute_contrast(const Tensor& frame);
    Tensor compute_structure(const Tensor& frame_a, const Tensor& frame_b);
    
    // Color space conversion helpers
    Tensor srgb_to_linear(const Tensor& frame);
    Tensor linear_to_srgb(const Tensor& frame);
    Tensor apply_gamma_correction(const Tensor& frame, float gamma);
    
    // Quality metric helpers
    Tensor compute_gaussian_window(int width, int height, float sigma);
    float clamp_to_range(float value, float min_val, float max_val);
    
    // Constants
    static constexpr float LUMINANCE_COEFF_R = 0.299f;
    static constexpr float LUMINANCE_COEFF_G = 0.587f;
    static constexpr float LUMINANCE_COEFF_B = 0.114f;
};

} // namespace render
} // namespace jin