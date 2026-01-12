#pragma once

#include "../core/tensor.h"
#include "frame_state.h"
#include "latent_lock.h"

namespace jin {
namespace temporal {

class DriftSuppressor {
public:
    DriftSuppressor() : drift_threshold_(0.3f), correction_strength_(0.1f), smoothing_factor_(0.8f) {}
    
    // Initialize with reference latent for drift detection
    void initialize(const Tensor& reference_latent) {
        reference_latent_ = reference_latent;
        is_initialized_ = true;
        last_correction_ = zeros(reference_latent.shape());
    }
    
    struct DriftCorrection {
        bool detected;
        Tensor correction_vector;
        float confidence;
        float drift_score;
        
        DriftCorrection() : detected(false), confidence(0.0f), drift_score(0.0f) {}
    };
    
    // Detect and correct drift between frames
    DriftCorrection detect_and_correct(
        const Tensor& prev_frame_latent,
        const Tensor& curr_frame_latent,
        const Tensor& optical_flow = Tensor()
    ) {
        DriftCorrection correction;
        
        if (!is_initialized_) return correction;
        
        // 1. Compute L2 distance from reference
        float current_drift = compute_l2_distance(curr_frame_latent, reference_latent_);
        correction.drift_score = current_drift;
        
        // 2. Check if drift exceeds threshold
        if (current_drift > drift_threshold_) {
            correction.detected = true;
            
            // 3. Compute correction vector
            Tensor raw_correction = reference_latent_ - curr_frame_latent;
            
            // 4. Apply exponential smoothing to prevent jitter
            if (last_correction_.size() > 0) {
                correction.correction_vector = last_correction_ * smoothing_factor_ + 
                                              raw_correction * (1.0f - smoothing_factor_);
            } else {
                correction.correction_vector = raw_correction;
            }
            
            // 5. Apply correction strength
            correction.correction_vector = correction.correction_vector * correction_strength_;
            
            // 6. Compute confidence based on drift magnitude
            correction.confidence = std::min(current_drift / (drift_threshold_ * 2.0f), 1.0f);
            
            // Store for next iteration
            last_correction_ = correction.correction_vector;
        }
        
        return correction;
    }
    
    // Specialized method for FrameState
    DriftCorrection detect_and_correct_frame_state(
        const FrameState& prev_state,
        const FrameState& curr_state
    ) {
        DriftCorrection correction;
        
        if (!is_initialized_) return correction;
        
        // Focus on character latent for drift detection
        if (curr_state.character_latent.size() > 0 && reference_latent_.size() > 0) {
            float character_drift = compute_l2_distance(curr_state.character_latent, reference_latent_);
            correction.drift_score = character_drift;
            
            if (character_drift > drift_threshold_) {
                correction.detected = true;
                
                // Compute correction for character latent
                Tensor raw_correction = reference_latent_ - curr_state.character_latent;
                
                // Smooth correction
                if (last_correction_.size() > 0) {
                    correction.correction_vector = last_correction_ * smoothing_factor_ + 
                                                  raw_correction * (1.0f - smoothing_factor_);
                } else {
                    correction.correction_vector = raw_correction;
                }
                
                // Apply correction strength
                correction.correction_vector = correction.correction_vector * correction_strength_;
                
                // Confidence based on drift severity
                correction.confidence = std::min(character_drift / (drift_threshold_ * 2.0f), 1.0f);
                
                last_correction_ = correction.correction_vector;
            }
        }
        
        return correction;
    }
    
    // Apply correction to a latent
    Tensor apply_correction(const Tensor& latent, const DriftCorrection& correction) const {
        if (!correction.detected || correction.correction_vector.size() == 0) {
            return latent;
        }
        
        return latent + correction.correction_vector;
    }
    
    // Get current drift score without applying correction
    float get_current_drift(const Tensor& current_latent) const {
        if (!is_initialized_) return 0.0f;
        return compute_l2_distance(current_latent, reference_latent_);
    }
    
    // Configuration
    void set_drift_threshold(float threshold) {
        drift_threshold_ = std::max(0.0f, threshold);
    }
    
    void set_correction_strength(float strength) {
        correction_strength_ = std::clamp(strength, 0.0f, 1.0f);
    }
    
    void set_smoothing_factor(float factor) {
        smoothing_factor_ = std::clamp(factor, 0.0f, 1.0f);
    }
    
    float get_drift_threshold() const { return drift_threshold_; }
    float get_correction_strength() const { return correction_strength_; }
    bool is_initialized() const { return is_initialized_; }

private:
    Tensor reference_latent_;
    Tensor last_correction_;
    float drift_threshold_;
    float correction_strength_;
    float smoothing_factor_;
    bool is_initialized_;
    
    // Helper: compute L2 distance between tensors
    float compute_l2_distance(const Tensor& a, const Tensor& b) const {
        if (a.size() != b.size() || a.size() == 0) return 0.0f;
        
        Tensor diff = a - b;
        Tensor squared_diff = pow(diff, 2.0f);
        return sqrt(sum(squared_diff).item());
    }
};

} // namespace temporal
} // namespace jin