#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include <vector>
#include <cmath>
#include <cstring>

namespace jin {
namespace temporal {

class MotionEncoder {
public:
    MotionEncoder(int motion_dim = 64) : motion_dim_(motion_dim) {}
    
    // Encode a sequence of frames into a compact motion descriptor
    Tensor encode(const std::vector<Tensor>& frame_features) {
        if (frame_features.empty()) {
            return zeros({motion_dim_});
        }
        
        if (frame_features.size() == 1) {
            // Single frame - use as-is (padded or truncated)
            return normalize_and_project(frame_features[0]);
        }
        
        // 1. Compute frame-to-frame deltas
        std::vector<Tensor> deltas;
        for (size_t i = 1; i < frame_features.size(); ++i) {
            Tensor delta = frame_features[i] - frame_features[i-1];
            deltas.push_back(delta);
        }
        
        // 2. Pool over sequence
        Tensor motion_descriptor = pool_deltas(deltas);
        
        // 3. Project to desired motion dimension
        Tensor final_descriptor = normalize_and_project(motion_descriptor);
        
        return final_descriptor;
    }
    
    // Encode FrameState sequence
    Tensor encode_frame_states(const std::vector<FrameState>& frame_states) {
        if (frame_states.empty()) {
            return zeros({motion_dim_});
        }
        
        // Extract motion-relevant features from each frame state
        std::vector<Tensor> motion_features;
        
        for (const auto& state : frame_states) {
            // Combine character motion and scene motion
            Tensor combined_motion;
            
            if (state.character_latent.size() > 0) {
                combined_motion = state.character_latent.clone();
                
                if (state.scene_latent.size() > 0) {
                    // Weighted combination of character and scene motion
                    Tensor scene_motion = state.scene_latent.clone();
                    combined_motion = combined_motion * 0.7f + scene_motion * 0.3f;
                }
            } else if (state.scene_latent.size() > 0) {
                combined_motion = state.scene_latent.clone();
            } else {
                continue;  // Skip frames with no motion information
            }
            
            motion_features.push_back(combined_motion);
        }
        
        return encode(motion_features);
    }
    
    // Decode motion code back to feature sequence (for analysis)
    std::vector<Tensor> decode(const Tensor& motion_code, int num_frames) {
        std::vector<Tensor> decoded_features;
        
        // For now, just replicate the motion code
        // In a real implementation, this would be a learned decoder
        for (int i = 0; i < num_frames; ++i) {
            decoded_features.push_back(motion_code.clone());
        }
        
        return decoded_features;
    }
    
    // Extract motion patterns from motion code
    struct MotionPattern {
        float walk_cycle_score;      // How much walking motion
        float action_intensity;      // Overall motion intensity
        float direction_change;      // How much direction changes
        float smoothness;           // Motion smoothness
        float transition_probability; // Likelihood of action boundary
    };
    
    MotionPattern analyze_motion_pattern(const Tensor& motion_code) {
        MotionPattern pattern{};
        
        if (motion_code.size() == 0) return pattern;
        
        // Simple heuristic analysis
        // In practice, this would be learned patterns
        
        float magnitude = std::sqrt(jin::ops::sum(jin::ops::pow(motion_code, 2.0f)).item());
        float normalized_intensity = static_cast<float>(magnitude / std::sqrt(static_cast<double>(motion_dim_)));
        pattern.action_intensity = std::min(normalized_intensity, 1.0f);
        
        // Motion smoothness (inverse of high-frequency components)
        float smoothness = 1.0f;
        if (motion_code.size() >= 4) {
            float variance = compute_variance(motion_code);
            smoothness = 1.0f / (1.0f + variance);
        }
        pattern.smoothness = smoothness;
        
        // Walk cycle detection (simplified - looks for periodic patterns)
        pattern.walk_cycle_score = detect_walk_cycle(motion_code);
        
        // Direction changes (high variance indicates changes)
        pattern.direction_change = compute_variance(motion_code);
        
        // Transition probability (based on action intensity)
        pattern.transition_probability = pattern.action_intensity * 0.5f;
        
        return pattern;
    }
    
    // Check if two motion codes are similar
    float motion_similarity(const Tensor& code1, const Tensor& code2) {
        if (code1.size() == 0 || code2.size() == 0 || code1.size() != code2.size()) {
            return 0.0f;
        }
        
        // Cosine similarity
        Tensor dot_product = jin::ops::sum(code1 * code2);
        float norm1 = std::sqrt(jin::ops::sum(jin::ops::pow(code1, 2.0f)).item());
        float norm2 = std::sqrt(jin::ops::sum(jin::ops::pow(code2, 2.0f)).item());
        
        if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
        
        return dot_product.item() / (norm1 * norm2);
    }
    
    // Interpolate between two motion codes
    Tensor interpolate_motion(const Tensor& code1, const Tensor& code2, float t) {
        if (code1.size() == 0 || code2.size() == 0 || code1.size() != code2.size()) {
            return Tensor();
        }
        
        t = std::clamp(t, 0.0f, 1.0f);
        return code1 * (1.0f - t) + code2 * t;
    }

private:
    int motion_dim_;
    
    // Pool deltas over sequence (mean pooling)
    Tensor pool_deltas(const std::vector<Tensor>& deltas) {
        if (deltas.empty()) {
            return zeros({static_cast<size_t>(motion_dim_)});
        }
        
        // Ensure all deltas have the same size
        if (deltas[0].size() == 0) {
            return zeros({static_cast<size_t>(motion_dim_)});
        }
        
        Tensor pooled = zeros(deltas[0].shape());
        
        for (const auto& delta : deltas) {
            pooled = pooled + delta;
        }
        
        return pooled / static_cast<float>(deltas.size());
    }
    
    // Normalize and project to target dimension
    Tensor normalize_and_project(const Tensor& features) {
        if (features.size() == 0) {
            return zeros({static_cast<size_t>(motion_dim_)});
        }
        
        // L2 normalize
        float norm = std::sqrt(jin::ops::sum(jin::ops::pow(features, 2.0f)).item());
        Tensor normalized = norm > 0 ? features / norm : features;
        
        // Project to target dimension (simple resize/truncate)
        if (static_cast<int>(normalized.size()) == motion_dim_) {
            return normalized;
        } else if (normalized.size() > static_cast<size_t>(motion_dim_)) {
            // Truncate
            return normalized.view({static_cast<size_t>(motion_dim_)});
        } else {
            // Pad with zeros
            Tensor projected = zeros({static_cast<size_t>(motion_dim_)});
            std::memcpy(
                projected.data(),
                normalized.data(),
                normalized.size() * sizeof(float)
            );
            return projected;
        }
    }
    
    // Compute variance of tensor elements
    float compute_variance(const Tensor& tensor) {
        if (tensor.size() == 0) return 0.0f;
        
        float mean = jin::ops::sum(tensor).item() / tensor.size();
        Tensor centered = tensor - mean;
        return jin::ops::sum(jin::ops::pow(centered, 2.0f)).item() / tensor.size();
    }
    
    // Simple walk cycle detection
    float detect_walk_cycle(const Tensor& motion_code) {
        // Simplified: look for periodic patterns in the motion code
        // In practice, this would use FFT or learned detectors
        
        if (motion_code.size() < 8) return 0.0f;  // Need minimum length
        
        // Count sign changes (indicates periodic behavior)
        int sign_changes = 0;
        const float* data = motion_code.data_ptr<float>();
        
        for (size_t i = 1; i < motion_code.size(); ++i) {
            if ((data[i] >= 0) != (data[i-1] >= 0)) {
                sign_changes++;
            }
        }
        
        float periodicity_score = static_cast<float>(sign_changes) / motion_code.size();
        return std::min(periodicity_score * 2.0f, 1.0f);  // Normalize to [0,1]
    }
};

} // namespace temporal
} // namespace jin