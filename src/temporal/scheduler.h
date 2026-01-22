#pragma once

#include "../core/tensor.h"
#include "../diffusion/sampler.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace jin {
namespace temporal {

class FrameScheduler {
public:
    // Ease type enumeration
    enum class EaseType {
        LINEAR,
        EASE_IN,
        EASE_OUT,
        EASE_IN_OUT,
        CUBIC,
        BOUNCE
    };
    
    FrameScheduler();
    
    // Schedule frames for animation timing
    std::vector<float> schedule_frames(
        float duration_seconds,
        float fps,
        EaseType ease_type = EaseType::LINEAR
    );
    
    // Get time value [0, 1] for frame index
    float get_time_at_frame(int frame_idx, float duration_seconds, float fps, EaseType ease_type = EaseType::LINEAR);
    
    // Apply easing function to normalized time [0, 1]
    float apply_easing(float t, EaseType ease_type);
    
    // Generate frame sequence with easing using diffusion sampler
    std::vector<Tensor> batch_generate(
        const diffusion::DDPMSampler& diffusion_sampler,
        int num_frames,
        EaseType ease_type,
        float guidance_scale = 7.5f
    );
    
    // Compute frame spacing based on ease curve
    std::vector<float> compute_frame_spacing(EaseType ease_type);

private:
    // Linear easing: f(t) = t
    float linear_ease(float t);
    
    // Ease in quadratic: f(t) = t^2
    float ease_in_quad(float t);
    
    // Ease out quadratic: f(t) = 1 - (1-t)^2
    float ease_out_quad(float t);
    
    // Ease in-out quadratic: combination of ease-in and ease-out
    float ease_in_out_quad(float t);
    
    // Ease in cubic: f(t) = t^3
    float ease_in_cubic(float t);
    
    // Ease out bounce: f(t) = 1 - (1-t)^3 with bounce at end
    float ease_out_bounce(float t);
    
    // Create conditioning tensor based on progress
    Tensor create_progress_conditioning(float progress);
};

} // namespace temporal
} // namespace jin