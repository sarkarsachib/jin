#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include "../diffusion/sampler.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace jin {
namespace temporal {

class LoopGenerator {
public:
    LoopGenerator();
    
    // Generate a cyclic frame sequence using diffusion sampler
    std::vector<Tensor> generate_loop(
        const diffusion::DDPMSampler& diffusion_sampler,
        int num_frames,
        float duration_seconds,
        float fps,
        float guidance_scale = 7.5f
    );
    
    // Enforce cyclic consistency through iterative blending
    void enforce_cyclic_consistency(std::vector<Tensor>& frames, int num_iterations);
    
    // Apply cross-fade to loop boundaries
    std::vector<Tensor> cross_fade_boundary(const std::vector<Tensor>& frames, int fade_width_frames);
    
    // Compute loop quality using spectral smoothness
    float compute_loop_quality(const std::vector<Tensor>& frames);
    
    // Compute boundary error (MSE between first and last frame)
    float compute_boundary_error(const std::vector<Tensor>& frames);

private:
    int fade_width_default_;
    
    // Create time-based conditioning for diffusion
    Tensor create_time_conditioning(float timestamp, float duration);
    
    // Smooth transitions around boundaries
    void smooth_boundary_transitions(std::vector<Tensor>& frames);
};

} // namespace temporal
} // namespace jin