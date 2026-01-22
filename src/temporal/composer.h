#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <stdexcept>

namespace jin {
namespace temporal {

class AnimationComposer {
public:
    // Transition type enumeration
    enum class TransitionType {
        CROSS_FADE,
        DISSOLVE,
        WIPE,
        FADE_IN,
        FADE_OUT
    };
    
    // Effect descriptor
    struct Effect {
        int start_frame;
        int end_frame;
        TransitionType type;
        std::map<std::string, float> parameters;
        
        Effect(int start, int end, TransitionType t) 
            : start_frame(start), end_frame(end), type(t) {}
    };
    
    AnimationComposer();
    
    // Compose sequence with effects
    std::vector<Tensor> compose_sequence(const std::vector<Tensor>& frames, const std::vector<Effect>& effects);
    
    // Add transition between two frame sequences
    std::vector<Tensor> add_transition(
        const std::vector<Tensor>& frames_a,
        const std::vector<Tensor>& frames_b,
        TransitionType transition_type,
        int duration_frames
    );
    
    // Render batch to files
    void render_batch(
        const std::vector<Tensor>& frames,
        const std::string& output_format,
        const std::string& output_dir
    );
    
    // Compute quality metrics for frame sequence
    std::map<std::string, float> compute_quality_metrics(const std::vector<Tensor>& frames);
    
    // Blend two layers
    Tensor blend_layers(
        const Tensor& layer1,
        const Tensor& layer2,
        const std::string& blend_mode,
        float alpha
    );

private:
    // Apply effect to frame sequence
    void apply_effect(std::vector<Tensor>& frames, const Effect& effect);
    
    // Cross-fade effect
    void apply_cross_fade(std::vector<Tensor>& frames, int start, int end);
    
    // Dissolve effect
    void apply_dissolve(std::vector<Tensor>& frames, int start, int end);
    
    // Wipe effect
    void apply_wipe(std::vector<Tensor>& frames, int start, int end);
    
    // Fade in effect
    void apply_fade_in(std::vector<Tensor>& frames, int start, int end);
    
    // Fade out effect
    void apply_fade_out(std::vector<Tensor>& frames, int start, int end);
    
    // Blend frames based on transition type
    Tensor blend_frames(const Tensor& frame_a, const Tensor& frame_b, float alpha, TransitionType type);
};

} // namespace temporal
} // namespace jin