#include "quality.h"
#include "../utils/math.h"
#include "../autodiff/tape.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace jin {
namespace render {

QualityMetrics::QualityMetrics(const QualityConfig& config) : config_(config) {}

float QualityMetrics::compute_psnr(const Tensor& original, const Tensor& rendered) {
    if (original.shape() != rendered.shape()) {
        std::cerr << "Warning: PSNR computation with different tensor shapes" << std::endl;
    }
    
    // Compute MSE
    Tensor diff = original - rendered;
    Tensor squared_diff = diff * diff;
    float mse = squared_diff.mean().item<float>();
    
    if (mse < 1e-10f) {
        return std::numeric_limits<float>::infinity();
    }
    
    // PSNR = 20 * log10(255 / sqrt(MSE)) for 8-bit images
    float psnr = 20.0f * std::log10(255.0f / std::sqrt(mse));
    return psnr;
}

float QualityMetrics::compute_ssim(const Tensor& frame_a, const Tensor& frame_b) {
    if (frame_a.shape() != frame_b.shape()) {
        std::cerr << "Warning: SSIM computation with different tensor shapes" << std::endl;
        return 0.0f;
    }
    
    // SSIM constants
    float C1 = (0.01f * 255.0f) * (0.01f * 255.0f);
    float C2 = (0.03f * 255.0f) * (0.03f * 255.0f);
    
    // Compute luminance
    Tensor luminance_a = compute_luminance(frame_a);
    Tensor luminance_b = compute_luminance(frame_b);
    
    // Compute contrast
    Tensor contrast_a = compute_contrast(frame_a, config_.gaussian_window_size);
    Tensor contrast_b = compute_contrast(frame_b, config_.gaussian_window_size);
    
    // Compute structure
    Tensor structure = compute_structure(frame_a, frame_b, config_.gaussian_window_size);
    
    // Compute means
    Tensor mean_a = frame_a.mean();
    Tensor mean_b = frame_b.mean();
    
    // SSIM formula
    Tensor numerator = (2 * mean_a * mean_b + C1) * (2 * structure + C2);
    Tensor denominator = (mean_a * mean_a + mean_b * mean_b + C1) * 
                       (contrast_a + contrast_b + C2);
    
    Tensor ssim_map = numerator / denominator;
    
    // Return mean SSIM
    return ssim_map.mean().item<float>();
}

std::vector<float> QualityMetrics::compute_motion_magnitude(const std::vector<Tensor>& frames) {
    if (frames.size() < 2) {
        return std::vector<float>(frames.size(), 0.0f);
    }
    
    std::vector<float> motion_magnitudes;
    motion_magnitudes.reserve(frames.size());
    
    for (size_t i = 1; i < frames.size(); ++i) {
        // Compute frame difference
        Tensor diff = frames[i] - frames[i-1];
        
        // Compute L2 norm of difference
        float magnitude = diff.abs().sum().item<float>();
        motion_magnitudes.push_back(magnitude);
    }
    
    // Add zero for first frame
    motion_magnitudes.insert(motion_magnitudes.begin(), 0.0f);
    
    return motion_magnitudes;
}

MotionAnalysis QualityMetrics::analyze_motion(const std::vector<Tensor>& frames) {
    MotionAnalysis analysis;
    
    analysis.motion_magnitude_per_frame = compute_motion_magnitude(frames);
    analysis.average_motion = std::accumulate(
        analysis.motion_magnitude_per_frame.begin(),
        analysis.motion_magnitude_per_frame.end(), 0.0f) / analysis.motion_magnitude_per_frame.size();
    
    // Simplified motion vector computation (using frame differences)
    analysis.motion_vectors_x = std::vector<float>(frames.size(), 0.0f);
    analysis.motion_vectors_y = std::vector<float>(frames.size(), 0.0f);
    
    // Motion consistency (simplified)
    if (analysis.motion_magnitude_per_frame.size() > 1) {
        float motion_variance = 0.0f;
        float mean_motion = analysis.average_motion;
        
        for (float motion : analysis.motion_magnitude_per_frame) {
            float diff = motion - mean_motion;
            motion_variance += diff * diff;
        }
        motion_variance /= analysis.motion_magnitude_per_frame.size();
        
        analysis.motion_consistency = 1.0f / (1.0f + motion_variance);
    } else {
        analysis.motion_consistency = 1.0f;
    }
    
    return analysis;
}

std::vector<float> QualityMetrics::detect_flicker(const std::vector<Tensor>& frames) {
    std::vector<float> flicker_scores;
    
    if (frames.size() < 2) {
        return std::vector<float>(frames.size(), 0.0f);
    }
    
    // Compute brightness variance for flicker detection
    std::vector<float> brightness_values = compute_brightness_variance(frames);
    
    // Detect sudden changes in brightness
    flicker_scores = detect_sudden_changes(brightness_values, config_.flicker_threshold);
    
    return flicker_scores;
}

FlickerAnalysis QualityMetrics::analyze_flicker(const std::vector<Tensor>& frames) {
    FlickerAnalysis analysis;
    
    analysis.flicker_per_frame = detect_flicker(frames);
    
    // Compute statistics
    if (!analysis.flicker_per_frame.empty()) {
        analysis.max_flicker = *std::max_element(analysis.flicker_per_frame.begin(), analysis.flicker_per_frame.end());
        
        analysis.flicker_frames = std::count_if(
            analysis.flicker_per_frame.begin(),
            analysis.flicker_per_frame.end(),
            [](float flicker) { return flicker > config_.flicker_threshold; }
        );
        
        // Simplified flicker frequency computation
        analysis.flicker_frequency = analysis.flicker_frames / static_cast<float>(frames.size());
    }
    
    return analysis;
}

float QualityMetrics::compute_temporal_stability(const std::vector<Tensor>& frames) {
    if (frames.size() < 2) {
        return 1.0f;
    }
    
    // Compute temporal gradients
    std::vector<float> temporal_gradients = compute_temporal_gradients(frames);
    
    // Temporal stability is inverse of average gradient magnitude
    float avg_gradient = std::accumulate(temporal_gradients.begin(), temporal_gradients.end(), 0.0f) / temporal_gradients.size();
    
    float stability = 1.0f / (1.0f + avg_gradient);
    return std::clamp(stability, 0.0f, 1.0f);
}

TemporalAnalysis QualityMetrics::analyze_temporal_coherence(const std::vector<Tensor>& frames) {
    TemporalAnalysis analysis;
    
    if (frames.size() < 2) {
        analysis.coherence_score = 1.0f;
        analysis.smoothness_score = 1.0f;
        analysis.consistency_score = 1.0f;
        analysis.coherence_per_frame = std::vector<float>(frames.size(), 1.0f);
        return analysis;
    }
    
    // Compute per-frame coherence
    analysis.coherence_per_frame = std::vector<float>(frames.size(), 1.0f);
    
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        float coherence = 1.0f / (1.0f + diff.abs().sum().item<float>());
        analysis.coherence_per_frame[i] = coherence;
    }
    
    // Overall coherence score
    analysis.coherence_score = std::accumulate(
        analysis.coherence_per_frame.begin(),
        analysis.coherence_per_frame.end(), 0.0f) / analysis.coherence_per_frame.size();
    
    // Motion smoothness
    std::vector<float> motion_magnitudes = compute_motion_magnitude(frames);
    analysis.smoothness_score = 1.0f / (1.0f + 
        std::accumulate(motion_magnitudes.begin(), motion_magnitudes.end(), 0.0f) / motion_magnitudes.size());
    
    // Temporal consistency
    analysis.consistency_score = compute_temporal_stability(frames);
    
    return analysis;
}

float QualityMetrics::compute_perceptual_quality(const std::vector<Tensor>& frames) {
    if (frames.empty()) {
        return 0.0f;
    }
    
    // Compute individual components
    float temporal_stability = compute_temporal_stability(frames);
    
    // Motion analysis
    MotionAnalysis motion = analyze_motion(frames);
    float motion_smoothness = 1.0f / (1.0f + motion.average_motion / 1000.0f);
    
    // Flicker analysis
    FlickerAnalysis flicker = analyze_flicker(frames);
    float flicker_quality = 1.0f - (flicker.max_flicker / 2.0f); // Normalize flicker
    
    // Weighted combination (total weight = 1.0)
    float perceptual_quality = 0.4f * temporal_stability + 
                             0.3f * motion_smoothness + 
                             0.3f * flicker_quality;
    
    return std::clamp(perceptual_quality, 0.0f, 1.0f) * 10.0f; // Scale to [0, 10]
}

QualityReport QualityMetrics::compute_comprehensive_quality(
    const std::vector<Tensor>& original_frames,
    const std::vector<Tensor>& rendered_frames
) {
    QualityReport report;
    
    if (original_frames.empty() || rendered_frames.empty()) {
        return report;
    }
    
    // Compute frame-wise metrics
    std::vector<float> psnr_values;
    std::vector<float> ssim_values;
    
    size_t min_frames = std::min(original_frames.size(), rendered_frames.size());
    
    for (size_t i = 0; i < min_frames; ++i) {
        psnr_values.push_back(compute_psnr(original_frames[i], rendered_frames[i]));
        ssim_values.push_back(compute_ssim(original_frames[i], rendered_frames[i]));
    }
    
    // Aggregate PSNR statistics
    report.psnr_avg = std::accumulate(psnr_values.begin(), psnr_values.end(), 0.0f) / psnr_values.size();
    report.psnr_min = *std::min_element(psnr_values.begin(), psnr_values.end());
    report.psnr_max = *std::max_element(psnr_values.begin(), psnr_values.end());
    
    // Aggregate SSIM statistics
    report.ssim_avg = std::accumulate(ssim_values.begin(), ssim_values.end(), 0.0f) / ssim_values.size();
    report.ssim_min = *std::min_element(ssim_values.begin(), ssim_values.end());
    report.ssim_max = *std::max_element(ssim_values.begin(), ssim_values.end());
    
    // Motion analysis
    MotionAnalysis motion = analyze_motion(rendered_frames);
    report.motion_magnitude = motion.average_motion;
    
    // Flicker analysis
    FlickerAnalysis flicker = analyze_flicker(rendered_frames);
    report.flicker_score = flicker.max_flicker;
    
    // Temporal stability
    report.temporal_stability = compute_temporal_stability(rendered_frames);
    
    // Perceptual quality
    report.perceptual_quality = compute_perceptual_quality(rendered_frames);
    
    // Overall grade
    report.overall_grade = grade_quality(report.perceptual_quality);
    
    return report;
}

QualityReport QualityMetrics::compute_metrics_batch(const std::vector<Tensor>& frames) {
    QualityReport report;
    
    if (frames.empty()) {
        return report;
    }
    
    // For single set of frames, we can't compute PSNR/SSIM directly
    // So we compute internal quality metrics
    
    // Motion analysis
    MotionAnalysis motion = analyze_motion(frames);
    report.motion_magnitude = motion.average_motion;
    
    // Flicker analysis
    FlickerAnalysis flicker = analyze_flicker(frames);
    report.flicker_score = flicker.max_flicker;
    
    // Temporal stability
    report.temporal_stability = compute_temporal_stability(frames);
    
    // Perceptual quality
    report.perceptual_quality = compute_perceptual_quality(frames);
    
    // Overall grade
    report.overall_grade = grade_quality(report.perceptual_quality);
    
    // Set PSNR/SSIM to invalid values for batch processing
    report.psnr_avg = -1.0f;
    report.psnr_min = -1.0f;
    report.psnr_max = -1.0f;
    report.ssim_avg = -1.0f;
    report.ssim_min = -1.0f;
    report.ssim_max = -1.0f;
    
    return report;
}

std::map<std::string, float> QualityMetrics::compute_all_metrics(const std::vector<Tensor>& frames) {
    std::map<std::string, float> all_metrics;
    
    if (frames.empty()) {
        return all_metrics;
    }
    
    // Motion metrics
    MotionAnalysis motion = analyze_motion(frames);
    all_metrics["motion_average"] = motion.average_motion;
    all_metrics["motion_consistency"] = motion.motion_consistency;
    
    // Flicker metrics
    FlickerAnalysis flicker = analyze_flicker(frames);
    all_metrics["flicker_max"] = flicker.max_flicker;
    all_metrics["flicker_frequency"] = flicker.flicker_frequency;
    all_metrics["flicker_frames"] = static_cast<float>(flicker.flicker_frames);
    
    // Temporal metrics
    TemporalAnalysis temporal = analyze_temporal_coherence(frames);
    all_metrics["temporal_coherence"] = temporal.coherence_score;
    all_metrics["temporal_smoothness"] = temporal.smoothness_score;
    all_metrics["temporal_consistency"] = temporal.consistency_score;
    
    // Overall quality
    all_metrics["perceptual_quality"] = compute_perceptual_quality(frames);
    all_metrics["temporal_stability"] = compute_temporal_stability(frames);
    
    return all_metrics;
}

Tensor QualityMetrics::compute_luminance(const Tensor& frame) {
    // Extract RGB channels and compute luminance
    Tensor r = frame.get_slice(-1, 1);
    Tensor g = frame.get_slice(-1, 2);
    Tensor b = frame.get_slice(-1, 3);
    
    Tensor luminance = 0.299f * r + 0.587f * g + 0.114f * b;
    return luminance;
}

Tensor QualityMetrics::compute_contrast(const Tensor& frame, int window_size) {
    // Compute local contrast using standard deviation
    Tensor mean = frame.mean();
    Tensor variance = ((frame - mean) * (frame - mean)).mean();
    Tensor contrast = variance.sqrt();
    return contrast;
}

Tensor QualityMetrics::compute_structure(const Tensor& frame_a, const Tensor& frame_b, int window_size) {
    // Compute structure component
    Tensor mean_a = frame_a.mean();
    Tensor mean_b = frame_b.mean();
    
    Tensor numerator = (frame_a - mean_a) * (frame_b - mean_b);
    Tensor denominator_a = ((frame_a - mean_a) * (frame_a - mean_a)).mean();
    Tensor denominator_b = ((frame_b - mean_b) * (frame_b - mean_b)).mean();
    
    Tensor structure = numerator / (denominator_a * denominator_b).sqrt();
    return structure;
}

Tensor QualityMetrics::create_gaussian_window(int size, float sigma) {
    Tensor window = Tensor::zeros({size});
    
    float center = (size - 1) / 2.0f;
    float sigma_squared = sigma * sigma;
    
    for (int i = 0; i < size; ++i) {
        float x = i - center;
        window.set_item(std::exp(-x * x / (2 * sigma_squared)), {i});
    }
    
    // Normalize
    float sum = window.sum().item<float>();
    if (sum > 0) {
        window = window / sum;
    }
    
    return window;
}

std::vector<float> QualityMetrics::compute_frame_differences(const std::vector<Tensor>& frames) {
    std::vector<float> differences;
    
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        float magnitude = diff.abs().sum().item<float>();
        differences.push_back(magnitude);
    }
    
    return differences;
}

std::vector<float> QualityMetrics::compute_optical_flow_approximation(const std::vector<Tensor>& frames) {
    // Simplified optical flow using frame differences
    return compute_frame_differences(frames);
}

std::vector<float> QualityMetrics::compute_brightness_variance(const std::vector<Tensor>& frames) {
    std::vector<float> brightness_values;
    
    for (const auto& frame : frames) {
        Tensor luminance = compute_luminance(frame);
        float brightness = luminance.mean().item<float>();
        brightness_values.push_back(brightness);
    }
    
    return brightness_values;
}

std::vector<float> QualityMetrics::detect_sudden_changes(const std::vector<float>& signal, float threshold) {
    std::vector<float> changes;
    
    if (signal.size() < 2) {
        return std::vector<float>(signal.size(), 0.0f);
    }
    
    for (size_t i = 1; i < signal.size(); ++i) {
        float change = std::abs(signal[i] - signal[i-1]);
        changes.push_back(change / threshold);
    }
    
    // Add zero for first frame
    changes.insert(changes.begin(), 0.0f);
    
    return changes;
}

std::vector<float> QualityMetrics::compute_temporal_gradients(const std::vector<Tensor>& frames) {
    std::vector<float> gradients;
    
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        float gradient_magnitude = diff.abs().sum().item<float>();
        gradients.push_back(gradient_magnitude);
    }
    
    return gradients;
}

std::vector<float> QualityMetrics::smooth_signal(const std::vector<float>& signal, int window_size) {
    if (signal.size() < 3) {
        return signal;
    }
    
    std::vector<float> smoothed(signal.size(), 0.0f);
    int half_window = window_size / 2;
    
    for (size_t i = 0; i < signal.size(); ++i) {
        float sum = 0.0f;
        int count = 0;
        
        for (int j = -half_window; j <= half_window; ++j) {
            int idx = i + j;
            if (idx >= 0 && idx < static_cast<int>(signal.size())) {
                sum += signal[idx];
                count++;
            }
        }
        
        smoothed[i] = sum / count;
    }
    
    return smoothed;
}

std::string QualityMetrics::grade_quality(float perceptual_score) const {
    if (perceptual_score >= 8.5f) {
        return "A";
    } else if (perceptual_score >= 7.0f) {
        return "B";
    } else if (perceptual_score >= 5.5f) {
        return "C";
    } else if (perceptual_score >= 4.0f) {
        return "D";
    } else {
        return "F";
    }
}

float QualityMetrics::compute_statistics(const std::vector<float>& values, const std::string& stat_type) {
    if (values.empty()) {
        return 0.0f;
    }
    
    if (stat_type == "mean") {
        return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    } else if (stat_type == "min") {
        return *std::min_element(values.begin(), values.end());
    } else if (stat_type == "max") {
        return *std::max_element(values.begin(), values.end());
    } else if (stat_type == "std") {
        float mean = compute_statistics(values, "mean");
        float variance = 0.0f;
        for (float val : values) {
            float diff = val - mean;
            variance += diff * diff;
        }
        variance /= values.size();
        return std::sqrt(variance);
    }
    
    return 0.0f;
}

} // namespace render
} // namespace jin