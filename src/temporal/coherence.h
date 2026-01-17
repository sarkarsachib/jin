#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace jin {
namespace temporal {

class TemporalCoherence {
public:
    TemporalCoherence();
    
    // Compute motion field between consecutive frames
    // Returns 2-channel tensor: [height, width, 2] where channel 0 = dx, channel 1 = dy
    Tensor compute_motion_field(const Tensor& frame_t, const Tensor& frame_t1);
    
    // Estimate optical flow using simplified Lucas-Kanade
    Tensor estimate_optical_flow(const Tensor& frame_a, const Tensor& frame_b);
    
    // Apply temporal smoothing via Gaussian filtering
    Tensor smooth_temporal(const std::vector<Tensor>& frames, float sigma);
    
    // Compute consistency score [0, 1] - higher is more consistent
    float compute_consistency_score(const std::vector<Tensor>& frames);

private:
    float sigma_;  // Gaussian smoothing parameter
    
    // Compute gradient in x direction
    Tensor compute_gradient_x(const Tensor& image);
    
    // Compute gradient in y direction
    Tensor compute_gradient_y(const Tensor& image);
    
    // Concatenate frames along time dimension
    Tensor concatenate_temporal(const std::vector<Tensor>& frames);
};

} // namespace temporal
} // namespace jin