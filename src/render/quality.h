#pragma once

#include "../core/tensor.h"
#include <string>
#include <vector>
#include <map>

namespace jin {
namespace render {

// Quality assessment configuration
struct QualityConfig {
    int gaussian_window_size;           // Size of Gaussian window for SSIM (11, 21, etc.)
    float ssim_sigma;                 // Standard deviation for Gaussian
    int motion_analysis_window;       // Window for motion analysis
    float flicker_threshold;          // Threshold for flicker detection
    
    QualityConfig()
        : gaussian_window_size(11), ssim_sigma(1.5f),
          motion_analysis_window(5), flicker_threshold(0.1f) {}
};

// Comprehensive quality report
struct QualityReport {
    float psnr_avg;                   // Average PSNR across frames (dB)
    float psnr_min;                   // Minimum PSNR across frames (dB)
    float psnr_max;                   // Maximum PSNR across frames (dB)
    float ssim_avg;                   // Average SSIM across frames [0, 1]
    float ssim_min;                   // Minimum SSIM across frames [0, 1]
    float ssim_max;                   // Maximum SSIM across frames [0, 1]
    float motion_magnitude;            // Average optical flow magnitude
    float flicker_score;              // Maximum flicker detected
    float temporal_stability;          // Overall temporal smoothness [0, 1]
    float perceptual_quality;          // Combined quality score [0, 10]
    std::string overall_grade;         // Letter grade (A/B/C/D/F)
    std::map<std::string, float> detailed_metrics; // Additional metrics
};

// Motion analysis result
struct MotionAnalysis {
    std::vector<float> motion_magnitude_per_frame; // Per-frame motion magnitude
    std::vector<float> motion_vectors_x;          // X components of motion vectors
    std::vector<float> motion_vectors_y;           // Y components of motion vectors
    float average_motion;                          // Average motion magnitude
    float motion_consistency;                     // Consistency of motion vectors [0, 1]
};

// Flicker detection result
struct FlickerAnalysis {
    std::vector<float> flicker_per_frame;         // Per-frame flicker intensity
    float max_flicker;                            // Maximum flicker detected
    int flicker_frames;                          // Number of frames with flicker
    float flicker_frequency;                     // Dominant flicker frequency
};

// Temporal coherence analysis
struct TemporalAnalysis {
    float coherence_score;                       // Overall temporal coherence [0, 1]
    std::vector<float> coherence_per_frame;      // Per-frame coherence scores
    float smoothness_score;                     // Motion smoothness score [0, 1]
    float consistency_score;                     // Temporal consistency [0, 1]
};

// Quality metrics computation class
class QualityMetrics {
public:
    QualityMetrics(const QualityConfig& config = QualityConfig{});
    
    // Perceptual quality metrics
    float compute_psnr(const Tensor& original, const Tensor& rendered);
    float compute_ssim(const Tensor& frame_a, const Tensor& frame_b);
    
    // Motion analysis
    std::vector<float> compute_motion_magnitude(const std::vector<Tensor>& frames);
    MotionAnalysis analyze_motion(const std::vector<Tensor>& frames);
    
    // Flicker detection
    std::vector<float> detect_flicker(const std::vector<Tensor>& frames);
    FlickerAnalysis analyze_flicker(const std::vector<Tensor>& frames);
    
    // Temporal stability
    float compute_temporal_stability(const std::vector<Tensor>& frames);
    TemporalAnalysis analyze_temporal_coherence(const std::vector<Tensor>& frames);
    
    // Combined quality assessment
    float compute_perceptual_quality(const std::vector<Tensor>& frames);
    QualityReport compute_comprehensive_quality(
        const std::vector<Tensor>& original_frames,
        const std::vector<Tensor>& rendered_frames
    );
    QualityReport compute_metrics_batch(const std::vector<Tensor>& frames);
    
    // Configuration management
    void set_config(const QualityConfig& config) { config_ = config; }
    const QualityConfig& get_config() const { return config_; }
    
    // Utility methods
    std::map<std::string, float> compute_all_metrics(const std::vector<Tensor>& frames);
    
private:
    QualityConfig config_;
    
    // Helper methods for PSNR/SSIM
    Tensor compute_luminance(const Tensor& frame);
    Tensor compute_contrast(const Tensor& frame, int window_size);
    Tensor compute_structure(const Tensor& frame_a, const Tensor& frame_b, int window_size);
    Tensor create_gaussian_window(int size, float sigma);
    
    // Motion analysis helpers
    std::vector<float> compute_frame_differences(const std::vector<Tensor>& frames);
    std::vector<float> compute_optical_flow_approximation(const std::vector<Tensor>& frames);
    
    // Flicker detection helpers
    std::vector<float> compute_brightness_variance(const std::vector<Tensor>& frames);
    std::vector<float> detect_sudden_changes(const std::vector<float>& signal, float threshold);
    
    // Temporal analysis helpers
    std::vector<float> compute_temporal_gradients(const std::vector<Tensor>& frames);
    std::vector<float> smooth_signal(const std::vector<float>& signal, int window_size);
    
    // Quality grading
    std::string grade_quality(float perceptual_score) const;
    
    // Statistical helpers
    float compute_statistics(const std::vector<float>& values, const std::string& stat_type);
    
    // Constants
    static constexpr float PSNR_EXCELLENT_THRESHOLD = 40.0f;
    static constexpr float PSNR_GOOD_THRESHOLD = 30.0f;
    static constexpr float SSIM_EXCELLENT_THRESHOLD = 0.95f;
    static constexpr float SSIM_GOOD_THRESHOLD = 0.85f;
};

} // namespace render
} // namespace jin