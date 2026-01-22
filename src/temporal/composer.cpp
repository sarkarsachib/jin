#include "composer.h"

namespace jin {
namespace temporal {

AnimationComposer::AnimationComposer() {}

std::vector<Tensor> AnimationComposer::compose_sequence(const std::vector<Tensor>& frames, const std::vector<Effect>& effects) {
    if (frames.empty()) return {};
    
    std::vector<Tensor> result = frames;
    
    // Apply each effect
    for (const auto& effect : effects) {
        apply_effect(result, effect);
    }
    
    return result;
}

std::vector<Tensor> AnimationComposer::add_transition(
    const std::vector<Tensor>& frames_a,
    const std::vector<Tensor>& frames_b,
    TransitionType transition_type,
    int duration_frames
) {
    if (frames_a.empty() || frames_b.empty() || duration_frames <= 0) {
        return frames_a;
    }
    
    std::vector<Tensor> result;
    result.reserve(frames_a.size() + frames_b.size());
    
    // Add all frames from sequence A
    result.insert(result.end(), frames_a.begin(), frames_a.end());
    
    // Add transition frames
    for (int i = 0; i < duration_frames; ++i) {
        float alpha = static_cast<float>(i + 1) / (duration_frames + 1);
        Tensor transition_frame = blend_frames(frames_a.back(), frames_b.front(), alpha, transition_type);
        result.push_back(transition_frame);
    }
    
    // Add all frames from sequence B
    result.insert(result.end(), frames_b.begin(), frames_b.end());
    
    return result;
}

void AnimationComposer::render_batch(
    const std::vector<Tensor>& frames,
    const std::string& output_format,
    const std::string& output_dir
) {
    if (frames.empty()) return;
    
    // Create output directory if it doesn't exist
    // (In a real implementation, would use filesystem operations)
    
    // For now, just log the rendering operation
    std::cout << "Rendering " << frames.size() << " frames to " << output_dir 
              << " in " << output_format << " format" << std::endl;
    
    // Note: Actual file I/O would require image encoding/decoding libraries
    // This is a placeholder implementation
}

std::map<std::string, float> AnimationComposer::compute_quality_metrics(const std::vector<Tensor>& frames) {
    std::map<std::string, float> metrics;
    
    if (frames.empty()) {
        metrics["temporal_smoothness"] = 0.0f;
        metrics["flicker_detection"] = 0.0f;
        metrics["motion_magnitude"] = 0.0f;
        return metrics;
    }
    
    // Temporal smoothness: average temporal gradient
    float total_temporal_gradient = 0.0f;
    int num_gradients = 0;
    
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        Tensor grad_mag = jin::ops::sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)));
        total_temporal_gradient += grad_mag.item();
        num_gradients++;
    }
    
    metrics["temporal_smoothness"] = (num_gradients > 0) ? 
        total_temporal_gradient / num_gradients : 0.0f;
    
    // Flicker detection: variance of frame differences
    float total_variance = 0.0f;
    std::vector<float> frame_diffs;
    
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        Tensor l2_norm = jin::ops::sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)));
        float diff_val = l2_norm.item();
        frame_diffs.push_back(diff_val);
        total_variance += diff_val;
    }
    
    if (!frame_diffs.empty()) {
        float mean_diff = total_variance / frame_diffs.size();
        float variance = 0.0f;
        for (float diff : frame_diffs) {
            variance += (diff - mean_diff) * (diff - mean_diff);
        }
        variance /= frame_diffs.size();
        metrics["flicker_detection"] = variance;
    } else {
        metrics["flicker_detection"] = 0.0f;
    }
    
    // Motion magnitude: average frame-to-frame displacement
    float total_motion = 0.0f;
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        Tensor motion = jin::ops::sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)));
        total_motion += motion.item();
    }
    
    metrics["motion_magnitude"] = (frames.size() > 1) ? 
        total_motion / (frames.size() - 1) : 0.0f;
    
    return metrics;
}

Tensor AnimationComposer::blend_layers(
    const Tensor& layer1,
    const Tensor& layer2,
    const std::string& blend_mode,
    float alpha
) {
    if (layer1.shape() != layer2.shape()) {
        throw std::invalid_argument("Layers must have the same shape");
    }
    
    // Clamp alpha to [0, 1]
    alpha = std::max(0.0f, std::min(alpha, 1.0f));
    
    if (blend_mode == "normal") {
        return layer1 * (1.0f - alpha) + layer2 * alpha;
    } else if (blend_mode == "multiply") {
        return layer1 * (layer2 * alpha + (1.0f - alpha));
    } else if (blend_mode == "screen") {
        // Screen blend: 1 - (1-a) * (1-b * alpha)
        Tensor ones_tensor = ones(layer1.shape());
        Tensor one_minus_layer1 = ones_tensor - layer1;
        Tensor one_minus_layer2 = ones_tensor - (layer2 * alpha);
        return ones_tensor - (one_minus_layer1 * one_minus_layer2);
    } else if (blend_mode == "overlay") {
        // Overlay blend mode
        Tensor result = zeros(layer1.shape());
        auto shape = layer1.shape();
        int size = 1;
        for (int dim : shape) size *= dim;
        
        for (int i = 0; i < size; ++i) {
            float l1 = layer1.data_ptr<float>()[i];
            float l2 = layer2.data_ptr<float>()[i];
            float blended;
            
            if (l1 < 0.5f) {
                blended = 2.0f * l1 * l2 * alpha + l1 * (1.0f - alpha);
            } else {
                blended = 1.0f - 2.0f * (1.0f - l1) * (1.0f - l2) * alpha;
            }
            
            result.data_ptr<float>()[i] = blended;
        }
        return result;
    } else {
        // Default to normal blend
        return layer1 * (1.0f - alpha) + layer2 * alpha;
    }
}

void AnimationComposer::apply_effect(std::vector<Tensor>& frames, const Effect& effect) {
    if (frames.empty() || effect.start_frame >= effect.end_frame || 
        effect.start_frame >= static_cast<int>(frames.size())) {
        return;
    }
    
    int actual_end = std::min(effect.end_frame, static_cast<int>(frames.size()));
    
    // Apply transition based on type
    switch (effect.type) {
        case TransitionType::CROSS_FADE:
            apply_cross_fade(frames, effect.start_frame, actual_end);
            break;
        case TransitionType::DISSOLVE:
            apply_dissolve(frames, effect.start_frame, actual_end);
            break;
        case TransitionType::WIPE:
            apply_wipe(frames, effect.start_frame, actual_end);
            break;
        case TransitionType::FADE_IN:
            apply_fade_in(frames, effect.start_frame, actual_end);
            break;
        case TransitionType::FADE_OUT:
            apply_fade_out(frames, effect.start_frame, actual_end);
            break;
    }
}

void AnimationComposer::apply_cross_fade(std::vector<Tensor>& frames, int start, int end) {
    int duration = end - start;
    if (duration <= 0) return;
    
    for (int i = 0; i < duration; ++i) {
        float alpha = static_cast<float>(i + 1) / (duration + 1);
        frames[start + i] = frames[start] * (1.0f - alpha);
    }
}

void AnimationComposer::apply_dissolve(std::vector<Tensor>& frames, int start, int end) {
    int duration = end - start;
    if (duration <= 0) return;
    
    for (int i = 0; i < duration; ++i) {
        float alpha = static_cast<float>(i + 1) / (duration + 1);
        // Add some randomness to dissolve pattern
        float random_offset = static_cast<float>(rand()) / RAND_MAX * 0.1f;
        alpha = std::max(0.0f, std::min(alpha + random_offset, 1.0f));
        frames[start + i] = frames[start] * (1.0f - alpha);
    }
}

void AnimationComposer::apply_wipe(std::vector<Tensor>& frames, int start, int end) {
    int duration = end - start;
    if (duration <= 0) return;
    
    for (int i = 0; i < duration; ++i) {
        float progress = static_cast<float>(i + 1) / (duration + 1);
        // Simplified wipe: fade based on progress
        frames[start + i] = frames[start] * (1.0f - progress);
    }
}

void AnimationComposer::apply_fade_in(std::vector<Tensor>& frames, int start, int end) {
    int duration = end - start;
    if (duration <= 0) return;
    
    for (int i = 0; i < duration; ++i) {
        float alpha = static_cast<float>(i + 1) / (duration + 1);
        frames[start + i] = frames[start] * alpha;
    }
}

void AnimationComposer::apply_fade_out(std::vector<Tensor>& frames, int start, int end) {
    int duration = end - start;
    if (duration <= 0) return;
    
    for (int i = 0; i < duration; ++i) {
        float alpha = 1.0f - static_cast<float>(i + 1) / (duration + 1);
        frames[start + i] = frames[start] * alpha;
    }
}

Tensor AnimationComposer::blend_frames(const Tensor& frame_a, const Tensor& frame_b, float alpha, TransitionType type) {
    switch (type) {
        case TransitionType::CROSS_FADE:
        case TransitionType::DISSOLVE:
            return frame_a * (1.0f - alpha) + frame_b * alpha;
        case TransitionType::WIPE:
            return alpha < 0.5f ? frame_a : frame_b;
        case TransitionType::FADE_IN:
            return frame_a * alpha;
        case TransitionType::FADE_OUT:
            return frame_a * (1.0f - alpha);
        default:
            return frame_a * (1.0f - alpha) + frame_b * alpha;
    }
}

} // namespace temporal
} // namespace jin