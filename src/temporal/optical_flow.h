#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include <vector>

namespace jin {
namespace temporal {

class LatentOpticalFlow {
public:
    LatentOpticalFlow() : flow_smoothing_(0.3f), saccade_threshold_(0.5f) {}
    
    // Compute flow from frame_t to frame_t+1
    Tensor estimate_flow(
        const Tensor& latent_t,            // Character latent at frame t
        const Tensor& latent_t1            // Character latent at frame t+1
    ) {
        if (latent_t.size() != latent_t1.size() || latent_t.size() == 0) {
            return Tensor();  // Return empty tensor if dimensions don't match
        }
        
        // Simple approach: (latent_t+1 - latent_t) smoothed
        Tensor raw_flow = latent_t1 - latent_t;
        
        // Apply exponential smoothing to prevent jitter
        if (last_flow_.size() > 0) {
            Tensor smoothed_flow = last_flow_ * flow_smoothing_ + raw_flow * (1.0f - flow_smoothing_);
            last_flow_ = smoothed_flow;
            return smoothed_flow;
        } else {
            last_flow_ = raw_flow;
            return raw_flow;
        }
    }
    
    // Compute motion for frame state
    Tensor estimate_flow_frame_state(
        const FrameState& state_t,
        const FrameState& state_t1
    ) {
        if (state_t.character_latent.size() == 0 || state_t1.character_latent.size() == 0) {
            return Tensor();
        }
        
        return estimate_flow(state_t.character_latent, state_t1.character_latent);
    }
    
    // Compute flow magnitude (L2 norm)
    float flow_magnitude(const Tensor& flow) {
        if (flow.size() == 0) return 0.0f;
        
        Tensor squared_flow = pow(flow, 2.0f);
        return sqrt(sum(squared_flow).item());
    }
    
    // Get average flow magnitude over a sequence
    float average_flow_magnitude(const std::vector<Tensor>& flows) {
        if (flows.empty()) return 0.0f;
        
        float total_magnitude = 0.0f;
        for (const auto& flow : flows) {
            total_magnitude += flow_magnitude(flow);
        }
        
        return total_magnitude / flows.size();
    }
    
    // Detect saccades (sudden jumps)
    bool is_saccade(const Tensor& flow, float threshold) {
        float magnitude = flow_magnitude(flow);
        return magnitude > threshold;
    }
    
    // Detect motion pattern changes
    bool is_motion_change(const Tensor& current_flow, const Tensor& previous_flow) {
        if (current_flow.size() == 0 || previous_flow.size() == 0) return false;
        
        // Compute direction change
        float current_mag = flow_magnitude(current_flow);
        float previous_mag = flow_magnitude(previous_flow);
        
        // Large magnitude change or direction change indicates motion boundary
        float magnitude_ratio = current_mag > 0 ? current_mag / previous_mag : 0.0f;
        
        return magnitude_ratio > 2.0f || magnitude_ratio < 0.5f;
    }
    
    // Classify motion type based on flow characteristics
    enum class MotionType {
        STATIC,      // Minimal motion
        SMOOTH,      // Continuous motion
        SACCADE,     // Sudden jump
        TRANSITION   // Action boundary
    };
    
    MotionType classify_motion(const Tensor& flow) {
        float magnitude = flow_magnitude(flow);
        
        if (magnitude < 0.1f) {
            return MotionType::STATIC;
        } else if (magnitude < 0.5f) {
            return MotionType::SMOOTH;
        } else if (magnitude < 1.0f) {
            return MotionType::SACCADE;
        } else {
            return MotionType::TRANSITION;
        }
    }
    
    // Reset internal state (call when starting new sequence)
    void reset() {
        last_flow_ = Tensor();
    }
    
    // Get last computed flow
    const Tensor& get_last_flow() const { return last_flow_; }
    
    // Configuration
    void set_flow_smoothing(float smoothing) {
        flow_smoothing_ = std::clamp(smoothing, 0.0f, 1.0f);
    }
    
    void set_saccade_threshold(float threshold) {
        saccade_threshold_ = std::max(0.0f, threshold);
    }
    
    float get_flow_smoothing() const { return flow_smoothing_; }
    float get_saccade_threshold() const { return saccade_threshold_; }

private:
    Tensor last_flow_;
    float flow_smoothing_;    // Smoothing factor for flow (0-1)
    float saccade_threshold_; // Threshold for detecting saccades
};

} // namespace temporal
} // namespace jin