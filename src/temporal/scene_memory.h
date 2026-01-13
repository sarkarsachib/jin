#pragma once

#include "../core/tensor.h"
#include "frame_state.h"
#include <vector>
#include <deque>
#include <cstring>

namespace jin {
namespace temporal {

class SceneMemory {
public:
    struct SceneState {
        Tensor background_latent;     // Compressed background
        Tensor lighting_latent;        // Lighting conditions
        Tensor camera_pose;            // Camera position/angle
        Tensor env_features;           // Environmental details
        int frame_index;              // Associated frame
        float timestamp;              // For temporal tracking
        
        SceneState() : frame_index(0), timestamp(0.0f) {}
        
        // Serialization
        Tensor to_tensor() const {
            std::vector<Tensor> components;

            if (background_latent.size() > 0) components.push_back(background_latent);
            if (lighting_latent.size() > 0) components.push_back(lighting_latent);
            if (camera_pose.size() > 0) components.push_back(camera_pose);
            if (env_features.size() > 0) components.push_back(env_features);

            // Add metadata
            Tensor metadata = zeros({2});
            metadata.data_ptr<float>()[0] = static_cast<float>(frame_index);
            metadata.data_ptr<float>()[1] = timestamp;
            components.push_back(metadata);

            // Simple concatenation for now
            if (components.empty()) return Tensor();

            size_t total_size = 0;
            for (const auto& tensor : components) {
                total_size += tensor.size();
            }

            Tensor result({static_cast<size_t>(total_size)});
            size_t offset = 0;

            for (const auto& tensor : components) {
                size_t tensor_size = tensor.size() * sizeof(float);
                std::memcpy(
                    static_cast<char*>(result.data()) + offset,
                    tensor.data(),
                    tensor_size
                );
                offset += tensor.size();
            }

            return result;
        }
        
        // Deserialization
        static SceneState from_tensor(const Tensor& tensor, int bg_dim, int light_dim, int cam_dim, int env_dim) {
            SceneState state;
            size_t offset = 0;
            
            if (bg_dim > 0) {
                state.background_latent = tensor.view({static_cast<size_t>(bg_dim)}).slice(offset, offset + bg_dim);
                offset += bg_dim;
            }
            
            if (light_dim > 0) {
                state.lighting_latent = tensor.view({static_cast<size_t>(light_dim)}).slice(offset, offset + light_dim);
                offset += light_dim;
            }
            
            if (cam_dim > 0) {
                state.camera_pose = tensor.view({static_cast<size_t>(cam_dim)}).slice(offset, offset + cam_dim);
                offset += cam_dim;
            }
            
            if (env_dim > 0) {
                state.env_features = tensor.view({static_cast<size_t>(env_dim)}).slice(offset, offset + env_dim);
                offset += env_dim;
            }
            
            // Extract metadata
            if (offset + 2 <= tensor.size()) {
                Tensor metadata = tensor.slice(offset, offset + 2);
                state.frame_index = static_cast<int>(metadata.data_ptr<float>()[0]);
                state.timestamp = metadata.data_ptr<float>()[1];
            }
            
            return state;
        }
        
        // Compute scene consistency with another scene
        float consistency_with(const SceneState& other) const {
            float consistency = 0.0f;
            int components = 0;
            
            // Background consistency
            if (background_latent.size() > 0 && other.background_latent.size() > 0) {
                Tensor diff = background_latent - other.background_latent;
                float bg_consistency = 1.0f / (1.0f + sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)).item()));
                consistency += bg_consistency;
                components++;
            }
            
            // Lighting consistency
            if (lighting_latent.size() > 0 && other.lighting_latent.size() > 0) {
                Tensor diff = lighting_latent - other.lighting_latent;
                float light_consistency = 1.0f / (1.0f + sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)).item()));
                consistency += light_consistency;
                components++;
            }
            
            // Camera pose consistency (position and orientation)
            if (camera_pose.size() > 0 && other.camera_pose.size() > 0) {
                Tensor diff = camera_pose - other.camera_pose;
                float cam_consistency = 1.0f / (1.0f + sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)).item()));
                consistency += cam_consistency;
                components++;
            }
            
            return components > 0 ? consistency / components : 1.0f;
        }
    };
    
    SceneMemory() : max_history_(10), consistency_threshold_(0.7f) {}
    
    // Get current scene state
    SceneState get_current() {
        if (history_.empty()) {
            return SceneState();
        }
        return history_.back();
    }
    
    // Update scene state
    void update(const Tensor& background, const Tensor& lighting, const Tensor& camera, int frame_idx) {
        SceneState new_state;
        new_state.background_latent = background;
        new_state.lighting_latent = lighting;
        new_state.camera_pose = camera;
        new_state.frame_index = frame_idx;
        new_state.timestamp = frame_idx * 0.033f;  // Assume 30 FPS
        
        // Enforce consistency with previous state if exists
        if (!history_.empty()) {
            enforce_consistency(new_state, history_.back());
        }
        
        history_.push_back(new_state);
        
        // Keep only last max_history_ scenes
        if (history_.size() > static_cast<size_t>(max_history_)) {
            history_.pop_front();
        }
    }
    
    // Update from FrameState
    void update_from_frame_state(const FrameState& frame_state) {
        SceneState scene_state;
        
        // Extract scene information from FrameState
        scene_state.background_latent = frame_state.scene_latent;
        
        // If no specific scene latents, use scene_latent as background
        if (scene_state.background_latent.size() == 0) {
            scene_state.background_latent = frame_state.scene_latent;
        }
        
        scene_state.frame_index = frame_state.frame_index;
        scene_state.timestamp = frame_state.timestamp;
        
        if (!history_.empty()) {
            enforce_consistency(scene_state, history_.back());
        }
        
        history_.push_back(scene_state);
        
        if (history_.size() > static_cast<size_t>(max_history_)) {
            history_.pop_front();
        }
    }
    
    // Compute scene consistency between two frames
    float scene_consistency(int frame_a, int frame_b) {
        const SceneState* state_a = find_scene_by_frame(frame_a);
        const SceneState* state_b = find_scene_by_frame(frame_b);
        
        if (!state_a || !state_b) return 0.0f;
        
        return state_a->consistency_with(*state_b);
    }
    
    // Find scene state by frame index
    const SceneState* find_scene_by_frame(int frame_idx) const {
        for (const auto& state : history_) {
            if (state.frame_index == frame_idx) {
                return &state;
            }
        }
        return nullptr;
    }
    
    // Get recent scenes (last K scenes)
    std::vector<SceneState> get_recent_scenes(int k) const {
        if (history_.empty() || k <= 0) return {};
        
        int start = std::max(0, static_cast<int>(history_.size()) - k);
        return std::vector<SceneState>(history_.begin() + start, history_.end());
    }
    
    // Detect scene changes/transitions
    struct SceneChange {
        int frame_index;
        float change_severity;  // 0-1, how much the scene changed
        SceneState before_state;
        SceneState after_state;
    };
    
    std::vector<SceneChange> detect_scene_changes(float threshold) const {
        std::vector<SceneChange> changes;
        
        if (history_.size() < 2) return changes;
        
        for (size_t i = 1; i < history_.size(); ++i) {
            float consistency = history_[i-1].consistency_with(history_[i]);
            
            if (consistency < threshold) {
                SceneChange change;
                change.frame_index = history_[i].frame_index;
                change.change_severity = 1.0f - consistency;
                change.before_state = history_[i-1];
                change.after_state = history_[i];
                changes.push_back(change);
            }
        }
        
        return changes;
    }
    
    // Reset scene memory (for scene cuts)
    void reset() {
        history_.clear();
    }
    
    // Get scene history
    const std::deque<SceneState>& get_history() const { return history_; }
    
    // Configuration
    void set_max_history(int max_history) {
        max_history_ = std::max(1, max_history);
    }
    
    void set_consistency_threshold(float threshold) {
        consistency_threshold_ = std::clamp(threshold, 0.0f, 1.0f);
    }
    
    size_t size() const { return history_.size(); }
    bool empty() const { return history_.empty(); }

private:
    std::deque<SceneState> history_;     // Keep last N scenes
    int max_history_;                     // Maximum number of scenes to remember
    float consistency_threshold_;          // Threshold for detecting scene changes
    
    // Enforce consistency with previous scene
    void enforce_consistency(SceneState& current, const SceneState& previous) {
        float consistency = previous.consistency_with(current);
        
        if (consistency < consistency_threshold_) {
            // Scene changed significantly - don't enforce consistency
            return;
        }
        
        // Apply slight smoothing to prevent abrupt changes
        float smoothing_factor = 0.1f;
        
        if (current.background_latent.size() > 0 && previous.background_latent.size() > 0) {
            current.background_latent = previous.background_latent * smoothing_factor + 
                                      current.background_latent * (1.0f - smoothing_factor);
        }
        
        if (current.lighting_latent.size() > 0 && previous.lighting_latent.size() > 0) {
            current.lighting_latent = previous.lighting_latent * smoothing_factor + 
                                    current.lighting_latent * (1.0f - smoothing_factor);
        }
        
        if (current.camera_pose.size() > 0 && previous.camera_pose.size() > 0) {
            current.camera_pose = previous.camera_pose * smoothing_factor + 
                                current.camera_pose * (1.0f - smoothing_factor);
        }
    }
    
    // Helper function for tensor concatenation
    Tensor concatenate_tensors(const std::vector<Tensor>& tensors) {
        if (tensors.empty()) return Tensor();
        
        size_t total_size = 0;
        for (const auto& tensor : tensors) {
            total_size += tensor.size();
        }
        
        Tensor result({static_cast<size_t>(total_size)});
        size_t offset = 0;
        
        for (const auto& tensor : tensors) {
            size_t tensor_size = tensor.size() * sizeof(float);
            std::memcpy(
                static_cast<char*>(result.data()) + offset,
                tensor.data(),
                tensor_size
            );
            offset += tensor.size();
        }
        
        return result;
    }
};

} // namespace temporal
} // namespace jin