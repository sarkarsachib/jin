#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include "frame_state.h"

namespace jin {
namespace temporal {

class LatentLock {
public:
    LatentLock() : lock_strength_(0.7f), is_initialized_(false) {}
    
    // Initialize with reference latent from frame 0
    void initialize(const Tensor& anchor_latent) {
        anchor_latent_ = anchor_latent;
        is_initialized_ = true;
    }
    
    // Apply lock during generation of frame N
    Tensor enforce_lock(const Tensor& noisy_latent) const {
        if (!is_initialized_) {
            return noisy_latent;  // No lock applied if not initialized
        }
        
        // Blend: result = anchor_latent * strength + noisy_latent * (1 - strength)
        Tensor result = anchor_latent_ * lock_strength_ + noisy_latent * (1.0f - lock_strength_);
        return result;
    }
    
    // Compute drift from anchor (L2 distance)
    float compute_drift(const Tensor& current_latent) const {
        if (!is_initialized_) return 0.0f;
        
        Tensor diff = current_latent - anchor_latent_;
        Tensor squared_diff = pow(diff, 2.0f);
        float drift = sqrt(sum(squared_diff).item());
        return drift;
    }
    
    // Compute component-wise drift (character vs scene)
    struct DriftComponents {
        float character_drift;
        float scene_drift;
        float total_drift;
    };
    
    DriftComponents compute_drift_components(const FrameState& current_state) const {
        DriftComponents drift{};
        
        if (!is_initialized_) return drift;
        
        // Character drift
        if (current_state.character_latent.size() > 0) {
            Tensor diff = current_state.character_latent - anchor_latent_;
            drift.character_drift = sqrt(sum(pow(diff, 2.0f)).item());
        }
        
        // Scene drift (if we have scene information)
        if (current_state.scene_latent.size() > 0) {
            Tensor diff = current_state.scene_latent - anchor_latent_;
            drift.scene_drift = sqrt(sum(pow(diff, 2.0f)).item());
        }
        
        drift.total_drift = (drift.character_drift + drift.scene_drift) * 0.5f;
        
        return drift;
    }
    
    // Set lock strength (0-1, where 1 = maximum consistency)
    void set_lock_strength(float strength) {
        lock_strength_ = std::clamp(strength, 0.0f, 1.0f);
    }
    
    float get_lock_strength() const { return lock_strength_; }
    bool is_initialized() const { return is_initialized_; }
    const Tensor& get_anchor_latent() const { return anchor_latent_; }

private:
    Tensor anchor_latent_;      // Character identity from frame 0
    float lock_strength_;        // How hard to enforce identity (0-1)
    bool is_initialized_;       // Whether lock has been initialized
};

} // namespace temporal
} // namespace jin