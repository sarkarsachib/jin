#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include "frame_state.h"
#include <vector>

namespace jin {
namespace temporal {

// Loss 1: Frame-to-frame smoothness
Tensor smoothness_loss(const std::vector<Tensor>& frame_features, float weight = 0.1f) {
    if (frame_features.size() < 2) {
        return zeros({1});  // No smoothness loss if less than 2 frames
    }
    
    Tensor total_smoothness = zeros({1});
    
    for (size_t i = 1; i < frame_features.size(); ++i) {
        Tensor diff = frame_features[i] - frame_features[i-1];
        Tensor squared_diff = jin::ops::pow(diff, 2.0f);
        total_smoothness = total_smoothness + jin::ops::sum(squared_diff);
    }
    
    // Normalize by number of frame pairs
    total_smoothness = total_smoothness / static_cast<float>(frame_features.size() - 1);
    
    // Apply weight
    return total_smoothness * weight;
}

// Loss 2: Drift penalty
Tensor drift_loss(const std::vector<Tensor>& frame_latents, const Tensor& reference_latent, float weight = 0.05f) {
    if (frame_latents.empty() || reference_latent.size() == 0) {
        return zeros({1});
    }
    
    Tensor total_drift = zeros({1});
    
    for (const auto& frame_latent : frame_latents) {
        if (frame_latent.size() == reference_latent.size()) {
            Tensor diff = frame_latent - reference_latent;
            Tensor squared_diff = jin::ops::pow(diff, 2.0f);
            total_drift = total_drift + jin::ops::sum(squared_diff);
        }
    }
    
    // Normalize by number of frames
    total_drift = total_drift / static_cast<float>(frame_latents.size());
    
    // Apply weight
    return total_drift * weight;
}

// Loss 3: Motion continuity
Tensor motion_continuity_loss(const std::vector<Tensor>& motion_codes, float weight = 0.1f) {
    if (motion_codes.size() < 2) {
        return zeros({1});  // No continuity loss if less than 2 frames
    }
    
    Tensor total_continuity = zeros({1});
    
    for (size_t i = 1; i < motion_codes.size(); ++i) {
        if (motion_codes[i].size() == motion_codes[i-1].size()) {
            Tensor diff = motion_codes[i] - motion_codes[i-1];
            Tensor squared_diff = jin::ops::pow(diff, 2.0f);
            total_continuity = total_continuity + jin::ops::sum(squared_diff);
        }
    }
    
    // Normalize by number of frame pairs
    total_continuity = total_continuity / static_cast<float>(motion_codes.size() - 1);
    
    // Apply weight
    return total_continuity * weight;
}

// Loss 4: Scene stability
Tensor scene_loss(const std::vector<Tensor>& scene_states, float weight = 0.05f) {
    if (scene_states.size() < 2) {
        return zeros({1});  // No scene loss if less than 2 frames
    }
    
    Tensor total_scene_stability = zeros({1});
    
    for (size_t i = 1; i < scene_states.size(); ++i) {
        if (scene_states[i].size() == scene_states[i-1].size()) {
            Tensor diff = scene_states[i] - scene_states[i-1];
            Tensor squared_diff = jin::ops::pow(diff, 2.0f);
            total_scene_stability = total_scene_stability + jin::ops::sum(squared_diff);
        }
    }
    
    // Normalize by number of frame pairs
    total_scene_stability = total_scene_stability / static_cast<float>(scene_states.size() - 1);
    
    // Apply weight
    return total_scene_stability * weight;
}

// Loss 5: Character identity consistency
Tensor identity_consistency_loss(const std::vector<FrameState>& frame_states, const Tensor& reference_character_latent, float weight = 0.15f) {
    if (frame_states.empty() || reference_character_latent.size() == 0) {
        return zeros({1});
    }
    
    Tensor total_identity_loss = zeros({1});
    
    for (const auto& state : frame_states) {
        if (state.character_latent.size() == reference_character_latent.size()) {
            Tensor diff = state.character_latent - reference_character_latent;
            Tensor squared_diff = jin::ops::pow(diff, 2.0f);
            total_identity_loss = total_identity_loss + jin::ops::sum(squared_diff);
        }
    }
    
    // Normalize by number of frames
    total_identity_loss = total_identity_loss / static_cast<float>(frame_states.size());
    
    // Apply weight
    return total_identity_loss * weight;
}

// Loss 6: Temporal attention consistency (encourages smooth attention patterns)
Tensor attention_consistency_loss(
    const std::vector<std::vector<float>>& attention_weights_per_frame,
    float weight = 0.05f
) {
    if (attention_weights_per_frame.size() < 2) {
        return zeros({1});
    }
    
    Tensor total_attention_consistency = zeros({1});
    
    for (size_t i = 1; i < attention_weights_per_frame.size(); ++i) {
        const auto& prev_weights = attention_weights_per_frame[i-1];
        const auto& curr_weights = attention_weights_per_frame[i];
        
        if (prev_weights.size() == curr_weights.size() && prev_weights.size() > 0) {
            float frame_consistency = 0.0f;
            for (size_t j = 0; j < prev_weights.size(); ++j) {
                float diff = curr_weights[j] - prev_weights[j];
                frame_consistency += diff * diff;
            }
            frame_consistency /= static_cast<float>(prev_weights.size());
            
            Tensor consistency_tensor({1});
            consistency_tensor.fill(frame_consistency);
            total_attention_consistency = total_attention_consistency + consistency_tensor;
        }
    }
    
    // Normalize by number of frame pairs
    total_attention_consistency = total_attention_consistency / static_cast<float>(attention_weights_per_frame.size() - 1);
    
    // Apply weight
    return total_attention_consistency * weight;
}

// Combined loss (used during training)
struct TemporalLosses {
    Tensor smoothness;
    Tensor drift;
    Tensor motion_continuity;
    Tensor scene_stability;
    Tensor identity_consistency;
    Tensor attention_consistency;
    Tensor total;
    
    TemporalLosses() {
        smoothness = zeros({1});
        drift = zeros({1});
        motion_continuity = zeros({1});
        scene_stability = zeros({1});
        identity_consistency = zeros({1});
        attention_consistency = zeros({1});
        total = zeros({1});
    }
};

TemporalLosses compute_temporal_loss(
    const std::vector<FrameState>& frames,
    const Tensor& reference_character_latent,
    const std::vector<std::vector<float>>& attention_weights = {},
    float λ_smooth = 0.1f,
    float λ_drift = 0.05f,
    float λ_motion = 0.1f,
    float λ_scene = 0.05f,
    float λ_identity = 0.15f,
    float λ_attention = 0.05f
) {
    TemporalLosses losses;
    
    // Extract features for loss computation
    std::vector<Tensor> frame_features;
    std::vector<Tensor> character_latents;
    std::vector<Tensor> motion_codes;
    std::vector<Tensor> scene_states;
    
    for (const auto& frame : frames) {
        // Combine all latent information for frame features
        Tensor combined_feature;
        if (frame.character_latent.size() > 0) {
            combined_feature = frame.character_latent.clone();
            if (frame.scene_latent.size() > 0) {
                combined_feature = combined_feature * 0.7f + frame.scene_latent * 0.3f;
            }
            if (frame.motion_code.size() > 0) {
                combined_feature = combined_feature * 0.8f + frame.motion_code * 0.2f;
            }
            frame_features.push_back(combined_feature);
        }
        
        // Character latents
        if (frame.character_latent.size() > 0) {
            character_latents.push_back(frame.character_latent);
        }
        
        // Motion codes
        if (frame.motion_code.size() > 0) {
            motion_codes.push_back(frame.motion_code);
        }
        
        // Scene states
        if (frame.scene_latent.size() > 0) {
            scene_states.push_back(frame.scene_latent);
        }
    }
    
    // Compute individual losses
    losses.smoothness = smoothness_loss(frame_features, λ_smooth);
    losses.drift = drift_loss(character_latents, reference_character_latent, λ_drift);
    losses.motion_continuity = motion_continuity_loss(motion_codes, λ_motion);
    losses.scene_stability = scene_loss(scene_states, λ_scene);
    losses.identity_consistency = identity_consistency_loss(frames, reference_character_latent, λ_identity);
    losses.attention_consistency = attention_consistency_loss(attention_weights, λ_attention);
    
    // Compute total loss
    losses.total = losses.smoothness + losses.drift + losses.motion_continuity + 
                  losses.scene_stability + losses.identity_consistency + losses.attention_consistency;
    
    return losses;
}

// Utility function to check numerical stability
bool has_nan_or_inf(const Tensor& tensor) {
    if (tensor.size() == 0) return false;
    
    const float* data = tensor.data_ptr<float>();
    for (size_t i = 0; i < tensor.size(); ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return true;
        }
    }
    return false;
}

// Check stability for all losses in TemporalLosses
bool has_nan_or_inf(const TemporalLosses& losses) {
    return has_nan_or_inf(losses.smoothness) ||
           has_nan_or_inf(losses.drift) ||
           has_nan_or_inf(losses.motion_continuity) ||
           has_nan_or_inf(losses.scene_stability) ||
           has_nan_or_inf(losses.identity_consistency) ||
           has_nan_or_inf(losses.attention_consistency) ||
           has_nan_or_inf(losses.total);
}

// Normalize losses to prevent numerical issues
TemporalLosses normalize_losses(const TemporalLosses& losses) {
    TemporalLosses normalized = losses;
    
    // Clamp individual losses to reasonable ranges
    auto clamp_loss = [](Tensor loss, float min_val, float max_val) -> Tensor {
        if (loss.size() == 0) return loss;
        
        float loss_value = loss.item();
        if (loss_value < min_val) {
            loss.fill(min_val);
        } else if (loss_value > max_val) {
            loss.fill(max_val);
        }
        return loss;
    };
    
    normalized.smoothness = clamp_loss(normalized.smoothness, 0.0f, 10.0f);
    normalized.drift = clamp_loss(normalized.drift, 0.0f, 10.0f);
    normalized.motion_continuity = clamp_loss(normalized.motion_continuity, 0.0f, 10.0f);
    normalized.scene_stability = clamp_loss(normalized.scene_stability, 0.0f, 10.0f);
    normalized.identity_consistency = clamp_loss(normalized.identity_consistency, 0.0f, 10.0f);
    normalized.attention_consistency = clamp_loss(normalized.attention_consistency, 0.0f, 10.0f);
    
    // Recompute total
    normalized.total = normalized.smoothness + normalized.drift + normalized.motion_continuity + 
                      normalized.scene_stability + normalized.identity_consistency + normalized.attention_consistency;
    
    return normalized;
}

} // namespace temporal
} // namespace jin