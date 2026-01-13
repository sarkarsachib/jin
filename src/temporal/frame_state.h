#pragma once

#include "../core/tensor.h"
#include <vector>
#include <string>

namespace jin {
namespace temporal {

struct FrameState {
    Tensor character_latent;      // [latent_dim] - compressed character identity
    Tensor scene_latent;           // [latent_dim] - scene/background state
    Tensor motion_code;            // [motion_dim] - motion/action descriptor
    Tensor emotion_embedding;      // [emotion_dim] - emotional state
    int frame_index;               // Frame number in sequence
    float timestamp;               // For timing analysis
    
    // Consistency metrics
    float drift_score;             // 0-1: how far from original character
    float motion_smoothness;       // 0-1: motion continuity with previous frame
    float scene_stability;         // 0-1: background consistency

    FrameState() : frame_index(0), timestamp(0.0f), 
                   drift_score(0.0f), motion_smoothness(1.0f), scene_stability(1.0f) {}

    // Serialization to Tensor for storage
    Tensor to_tensor() const {
        // Concatenate all latent representations into a single tensor
        std::vector<Tensor> components;
        components.push_back(character_latent);
        components.push_back(scene_latent);
        components.push_back(motion_code);
        components.push_back(emotion_embedding);
        
        // Add scalar metrics
        Tensor metrics = zeros({4});
        metrics.data_ptr<float>()[0] = drift_score;
        metrics.data_ptr<float>()[1] = motion_smoothness;
        metrics.data_ptr<float>()[2] = scene_stability;
        metrics.data_ptr<float>()[3] = static_cast<float>(frame_index);
        components.push_back(metrics);
        
        return concatenate(components, 0);
    }

    // Deserialization from Tensor
    static FrameState from_tensor(const Tensor& tensor, int latent_dim, int motion_dim, int emotion_dim) {
        FrameState state;
        size_t offset = 0;
        
        // Extract character latent
        if (latent_dim > 0 && offset + latent_dim <= tensor.size()) {
            state.character_latent = tensor.view({latent_dim}).slice(offset, offset + latent_dim);
            offset += latent_dim;
        }
        
        // Extract scene latent
        if (latent_dim > 0 && offset + latent_dim <= tensor.size()) {
            state.scene_latent = tensor.view({latent_dim}).slice(offset, offset + latent_dim);
            offset += latent_dim;
        }
        
        // Extract motion code
        if (motion_dim > 0 && offset + motion_dim <= tensor.size()) {
            state.motion_code = tensor.view({motion_dim}).slice(offset, offset + motion_dim);
            offset += motion_dim;
        }
        
        // Extract emotion embedding
        if (emotion_dim > 0 && offset + emotion_dim <= tensor.size()) {
            state.emotion_embedding = tensor.view({static_cast<size_t>(emotion_dim)}).slice(offset, offset + emotion_dim);
            offset += emotion_dim;
        }
        
        // Extract metrics
        if (offset + 4 <= tensor.size()) {
            Tensor metrics = tensor.slice(offset, offset + 4);
            const float* metrics_data = metrics.data_ptr<float>();
            state.drift_score = metrics_data[0];
            state.motion_smoothness = metrics_data[1];
            state.scene_stability = metrics_data[2];
            state.frame_index = static_cast<int>(metrics_data[3]);
        }
        
        return state;
    }

    // Hash function for caching
    size_t hash() const {
        // Simple hash based on frame index and metrics
        size_t h = std::hash<int>{}(frame_index);
        h ^= std::hash<float>{}(drift_score) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(motion_smoothness) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<float>{}(scene_stability) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }

    // Comparison operators for drift detection
    bool operator==(const FrameState& other) const {
        return frame_index == other.frame_index &&
               std::abs(drift_score - other.drift_score) < 1e-5f &&
               std::abs(motion_smoothness - other.motion_smoothness) < 1e-5f &&
               std::abs(scene_stability - other.scene_stability) < 1e-5f;
    }

    bool operator!=(const FrameState& other) const {
        return !(*this == other);
    }

    // Calculate L2 distance between two frame states
    float distance_to(const FrameState& other) const {
        float dist = 0.0f;
        
        if (character_latent.size() > 0 && other.character_latent.size() > 0) {
            Tensor diff = character_latent - other.character_latent;
            dist += sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)).item());
        }
        
        if (scene_latent.size() > 0 && other.scene_latent.size() > 0) {
            Tensor diff = scene_latent - other.scene_latent;
            dist += sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)).item());
        }
        
        return dist;
    }
};

} // namespace temporal
} // namespace jin

// Hash specialization for unordered_map (must be in global std namespace)
namespace std {
    template<>
    struct hash<jin::temporal::FrameState> {
        size_t operator()(const jin::temporal::FrameState& state) const {
            return state.hash();
        }
    };
}