#pragma once

#include "../diffusion/sampler.h"
#include "coherence.h"
#include "interpolation.h"
#include "loop.h"
#include "scheduler.h"
#include "composer.h"
#include "../core/tensor.h"
#include <vector>
#include <map>
#include <string>
#include <functional>

namespace jin {
namespace temporal {

// Forward declarations
namespace diffusion {
    class DDPMSampler;
}

// Configuration structure for animation generation
struct AnimationConfig {
    int num_frames;                    // Total frames to generate
    float fps;                         // Frames per second
    float duration_seconds;            // Animation length
    FrameScheduler::EaseType ease_type; // Frame scheduling ease type
    bool loop_enabled;                // Whether to enforce cyclic consistency
    float guidance_scale;              // Classifier-free guidance strength [1.0, 15.0]
    int num_sampling_steps;            // DDIM steps per frame
    int interpolation_steps;           // Frames to interpolate between keyframes
    
    AnimationConfig() 
        : num_frames(30), fps(30.0f), duration_seconds(1.0f), 
          ease_type(FrameScheduler::EaseType::LINEAR), 
          loop_enabled(true), guidance_scale(7.5f), 
          num_sampling_steps(50), interpolation_steps(4) {}
};

class DiffusionAnimationLoop {
public:
    DiffusionAnimationLoop(
        const diffusion::DDPMSampler& diffusion_sampler,
        const AnimationConfig& config = AnimationConfig{}
    );
    
    // Generate complete animation
    std::vector<Tensor> generate_animation();
    
    // Refine frames using guidance from reference frames
    std::vector<Tensor> refine_with_guidance(
        const std::vector<Tensor>& frames,
        const std::vector<Tensor>& guidance_frames
    );
    
    // Progressive generation with callbacks
    void progressive_generation(
        int steps_per_frame,
        std::function<void(const std::vector<Tensor>&, int)> callback
    );
    
    // Export frames to files
    void export_frames(
        const std::vector<Tensor>& frames,
        const std::string& output_dir,
        const std::string& format = "png"
    );
    
    // Get animation metadata
    std::map<std::string, float> get_animation_metadata() const;
    
    // Update configuration
    void update_config(const AnimationConfig& new_config);
    
    // Get current configuration
    const AnimationConfig& get_config() const;

private:
    const diffusion::DDPMSampler& diffusion_sampler_;
    AnimationConfig config_;
    
    // Component instances
    TemporalCoherence temporal_coherence_;
    MotionInterpolator motion_interpolator_;
    LoopGenerator loop_generator_;
    FrameScheduler frame_scheduler_;
    AnimationComposer animation_composer_;
    
    // Store last generated frames for metadata
    std::vector<Tensor> last_generated_frames_;
    
    // Helper methods
    
    // Add interpolation between keyframes
    std::vector<Tensor> add_interpolation(const std::vector<Tensor>& keyframes);
    
    // Apply temporal coherence smoothing
    std::vector<Tensor> apply_temporal_coherence(const std::vector<Tensor>& frames);
    
    // Generate a single frame with progressive refinement
    Tensor generate_progressive_frame(int frame_idx, int steps);
    
    // Store frames for metadata computation
    void store_frames_for_metadata(const std::vector<Tensor>& frames);
};

} // namespace temporal
} // namespace jin