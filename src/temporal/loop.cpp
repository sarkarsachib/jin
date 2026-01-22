#include "loop.h"

namespace jin {
namespace temporal {

LoopGenerator::LoopGenerator() : fade_width_default_(10) {}

std::vector<Tensor> LoopGenerator::generate_loop(
    const diffusion::DDPMSampler& diffusion_sampler,
    int num_frames,
    float duration_seconds,
    float fps,
    float guidance_scale
) {
    if (num_frames < 2) {
        throw std::invalid_argument("Need at least 2 frames for a loop");
    }
    
    std::vector<Tensor> frames;
    frames.reserve(num_frames);
    
    // For testing purposes, create dummy frames instead of using actual diffusion
    for (int i = 0; i < num_frames; ++i) {
        float progress = static_cast<float>(i) / num_frames;
        Tensor frame = ones({3, 64, 64}) * progress;
        frames.push_back(frame);
    }
    
    // Enforce cyclic consistency
    enforce_cyclic_consistency(frames, 5); // 5 iterations
    
    return frames;
}

void LoopGenerator::enforce_cyclic_consistency(std::vector<Tensor>& frames, int num_iterations) {
    if (frames.size() < 2) return;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Blend boundary frames toward center
        Tensor& first_frame = frames.front();
        Tensor& last_frame = frames.back();
        
        // Compute blending factor (decreases with iterations)
        float alpha = 0.5f * std::exp(-static_cast<float>(iter) * 0.5f);
        
        // Symmetric blending: first_frame <- mix(first_frame, last_frame)
        // last_frame <- mix(last_frame, first_frame)
        Tensor blended_first = first_frame * (1.0f - alpha) + last_frame * alpha;
        Tensor blended_last = last_frame * (1.0f - alpha) + first_frame * alpha;
        
        first_frame = blended_first;
        last_frame = blended_last;
        
        // Optionally smooth intermediate frames slightly
        if (frames.size() > 2) {
            smooth_boundary_transitions(frames);
        }
    }
}

std::vector<Tensor> LoopGenerator::cross_fade_boundary(const std::vector<Tensor>& frames, int fade_width_frames) {
    if (frames.size() < 2) return frames;
    
    int num_frames = frames.size();
    fade_width_frames = std::min(fade_width_frames, num_frames / 2);
    
    std::vector<Tensor> result = frames;
    
    // Apply alpha ramp at beginning
    for (int i = 0; i < fade_width_frames; ++i) {
        float alpha = static_cast<float>(i) / fade_width_frames;
        result[i] = frames[i] * alpha;
    }
    
    // Apply alpha ramp at end (in reverse)
    for (int i = 0; i < fade_width_frames; ++i) {
        int src_idx = num_frames - fade_width_frames + i;
        int dst_idx = num_frames - fade_width_frames + i;
        float alpha = 1.0f - static_cast<float>(i) / fade_width_frames;
        result[dst_idx] = frames[src_idx] * alpha;
    }
    
    return result;
}

float LoopGenerator::compute_loop_quality(const std::vector<Tensor>& frames) {
    if (frames.empty()) return 0.0f;
    
    // Temporal frequency analysis
    std::vector<float> temporal_signal;
    
    // Extract temporal signal by computing frame differences
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        Tensor l2_norm = jin::ops::sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)));
        temporal_signal.push_back(l2_norm.item());
    }
    
    if (temporal_signal.size() < 2) return 1.0f;
    
    // Compute power spectral density (simplified)
    float low_freq_power = 0.0f;
    float total_power = 0.0f;
    int signal_length = temporal_signal.size();
    
    // Simple DFT approximation
    for (int k = 0; k < signal_length / 2; ++k) {
        float real = 0.0f, imag = 0.0f;
        
        for (int n = 0; n < signal_length; ++n) {
            float angle = -2.0f * M_PI * k * n / signal_length;
            real += temporal_signal[n] * std::cos(angle);
            imag += temporal_signal[n] * std::sin(angle);
        }
        
        float power = (real * real + imag * imag) / signal_length;
        total_power += power;
        
        // Low frequencies are first half of spectrum
        if (k < signal_length / 4) {
            low_freq_power += power;
        }
    }
    
    if (total_power < 1e-6f) return 1.0f;
    
    // Quality = fraction of power in low frequencies
    // Good loops should have smooth temporal variations (low frequencies)
    float quality = low_freq_power / total_power;
    
    // Penalize high frequency content (flicker)
    return quality;
}

float LoopGenerator::compute_boundary_error(const std::vector<Tensor>& frames) {
    if (frames.size() < 2) return 0.0f;
    
    Tensor diff = frames.front() - frames.back();
    Tensor mse = jin::ops::mean(jin::ops::pow(diff, 2.0f));
    
    return mse.item();
}

Tensor LoopGenerator::create_time_conditioning(float timestamp, float duration) {
    // Simple sinusoidal encoding for time
    float normalized_time = timestamp / duration;
    
    // Create a simple conditioning tensor with time encoding
    // For now, use a simple 4D tensor that could represent time features
    Tensor conditioning = zeros({4});
    
    // Sinusoidal position encoding (simplified)
    conditioning.data_ptr<float>()[0] = std::sin(2 * M_PI * normalized_time);
    conditioning.data_ptr<float>()[1] = std::cos(2 * M_PI * normalized_time);
    conditioning.data_ptr<float>()[2] = normalized_time;
    conditioning.data_ptr<float>()[3] = 1.0f - normalized_time;
    
    return conditioning;
}

void LoopGenerator::smooth_boundary_transitions(std::vector<Tensor>& frames) {
    if (frames.size() < 3) return;
    
    int num_frames = frames.size();
    
    // Smooth first few frames
    for (int i = 1; i < std::min(3, num_frames - 1); ++i) {
        Tensor smoothed = frames[i-1] * 0.25f + frames[i] * 0.5f + frames[i+1] * 0.25f;
        frames[i] = smoothed;
    }
    
    // Smooth last few frames
    for (int i = std::max(1, num_frames - 3); i < num_frames - 1; ++i) {
        Tensor smoothed = frames[i-1] * 0.25f + frames[i] * 0.5f + frames[i+1] * 0.25f;
        frames[i] = smoothed;
    }
}

} // namespace temporal
} // namespace jin