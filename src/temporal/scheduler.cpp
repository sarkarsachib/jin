#include "scheduler.h"

namespace jin {
namespace temporal {

FrameScheduler::FrameScheduler() {}

std::vector<float> FrameScheduler::schedule_frames(
    float duration_seconds,
    float fps,
    EaseType ease_type
) {
    if (duration_seconds <= 0 || fps <= 0) {
        return {};
    }
    
    int num_frames = static_cast<int>(duration_seconds * fps);
    std::vector<float> timestamps;
    timestamps.reserve(num_frames);
    
    for (int i = 0; i < num_frames; ++i) {
        float normalized_time = static_cast<float>(i) / (num_frames - 1);
        float eased_time = apply_easing(normalized_time, ease_type);
        timestamps.push_back(eased_time);
    }
    
    return timestamps;
}

float FrameScheduler::get_time_at_frame(int frame_idx, float duration_seconds, float fps, EaseType ease_type) {
    if (duration_seconds <= 0 || fps <= 0) return 0.0f;
    
    int total_frames = static_cast<int>(duration_seconds * fps);
    if (total_frames <= 1) return 0.0f;
    
    float normalized_time = static_cast<float>(frame_idx) / (total_frames - 1);
    return apply_easing(normalized_time, ease_type);
}

float FrameScheduler::apply_easing(float t, EaseType ease_type) {
    t = std::clamp(t, 0.0f, 1.0f);
    
    switch (ease_type) {
        case EaseType::LINEAR:
            return linear_ease(t);
        case EaseType::EASE_IN:
            return ease_in_quad(t);
        case EaseType::EASE_OUT:
            return ease_out_quad(t);
        case EaseType::EASE_IN_OUT:
            return ease_in_out_quad(t);
        case EaseType::CUBIC:
            return ease_in_cubic(t);
        case EaseType::BOUNCE:
            return ease_out_bounce(t);
        default:
            return linear_ease(t);
    }
}

std::vector<Tensor> FrameScheduler::batch_generate(
    const diffusion::DDPMSampler& diffusion_sampler,
    int num_frames,
    EaseType ease_type,
    float guidance_scale
) {
    std::vector<Tensor> frames;
    frames.reserve(num_frames);
    
    for (int i = 0; i < num_frames; ++i) {
        // Apply easing to control timing
        float eased_progress = apply_easing(static_cast<float>(i) / (num_frames - 1), ease_type);
        
        // Create conditioning based on eased progress
        Tensor conditioning = create_progress_conditioning(eased_progress);
        
        // Generate frame - for now, create dummy frame
        Tensor frame = ones({3, 64, 64}) * eased_progress;
        frames.push_back(frame);
    }
    
    return frames;
}

std::vector<float> FrameScheduler::compute_frame_spacing(EaseType ease_type) {
    // Generate a standard sequence and compute spacings
    std::vector<float> timestamps = schedule_frames(1.0f, 30.0f, ease_type); // 30fps standard
    std::vector<float> spacings;
    
    for (size_t i = 1; i < timestamps.size(); ++i) {
        spacings.push_back(timestamps[i] - timestamps[i-1]);
    }
    
    return spacings;
}

float FrameScheduler::linear_ease(float t) {
    return t;
}

float FrameScheduler::ease_in_quad(float t) {
    return t * t;
}

float FrameScheduler::ease_out_quad(float t) {
    return 1.0f - (1.0f - t) * (1.0f - t);
}

float FrameScheduler::ease_in_out_quad(float t) {
    if (t < 0.5f) {
        return 2.0f * t * t;
    } else {
        return 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;
    }
}

float FrameScheduler::ease_in_cubic(float t) {
    return t * t * t;
}

float FrameScheduler::ease_out_bounce(float t) {
    const float n1 = 7.5625f;
    const float d1 = 2.75f;
    
    if (t < 1.0f / d1) {
        return n1 * t * t;
    } else if (t < 2.0f / d1) {
        t -= 1.5f / d1;
        return n1 * t * t + 0.75f;
    } else if (t < 2.5f / d1) {
        t -= 2.25f / d1;
        return n1 * t * t + 0.9375f;
    } else {
        t -= 2.625f / d1;
        return n1 * t * t + 0.984375f;
    }
}

Tensor FrameScheduler::create_progress_conditioning(float progress) {
    Tensor conditioning = zeros({8});
    
    // Encode progress in multiple ways
    conditioning.data_ptr<float>()[0] = progress;
    conditioning.data_ptr<float>()[1] = 1.0f - progress;
    conditioning.data_ptr<float>()[2] = std::sin(2 * M_PI * progress);
    conditioning.data_ptr<float>()[3] = std::cos(2 * M_PI * progress);
    conditioning.data_ptr<float>()[4] = progress * progress;
    conditioning.data_ptr<float>()[5] = (1.0f - progress) * (1.0f - progress);
    conditioning.data_ptr<float>()[6] = std::sqrt(progress);
    conditioning.data_ptr<float>()[7] = std::sqrt(1.0f - progress);
    
    return conditioning;
}

} // namespace temporal
} // namespace jin