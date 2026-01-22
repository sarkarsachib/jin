#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace jin {
namespace temporal {

class MotionInterpolator {
public:
    MotionInterpolator();
    
    // Interpolate frames between frame_a and frame_b
    // Returns num_steps+2 frames total (including start and end frames)
    std::vector<Tensor> interpolate_frames(
        const Tensor& frame_a, 
        const Tensor& frame_b, 
        int num_steps,
        const std::string& method = "linear"
    );
    
    // Blend two frames with parameter t in [0, 1]
    Tensor blend_motion(const Tensor& frame1, const Tensor& frame2, float t);
    
    // Adaptive frame insertion to match target FPS
    std::vector<Tensor> adaptive_insert(
        const std::vector<Tensor>& frames,
        float target_fps,
        float source_fps
    );
    
    // Warp frame using motion field and alpha parameter
    Tensor warp_frame(const Tensor& frame, const Tensor& motion_field, float alpha);
    
    // Compute interpolation quality metric (lower is smoother)
    float compute_interpolation_quality(
        const std::vector<Tensor>& original_frames,
        const std::vector<Tensor>& interpolated_frames
    );

private:
    int default_num_steps_;
    
    // Linear interpolation implementation
    std::vector<Tensor> linear_interpolation(const Tensor& frame_a, const Tensor& frame_b, int num_steps);
    
    // Cubic spline interpolation implementation
    std::vector<Tensor> cubic_interpolation(const Tensor& frame_a, const Tensor& frame_b, int num_steps);
    
    // Hermite interpolation (cubic)
    Tensor hermite_interpolate(
        const Tensor& p0, const Tensor& p1,
        const Tensor& m0, const Tensor& m1, float t
    );
    
    // Compute simple frame derivative (for cubic interpolation)
    Tensor compute_frame_derivative(const Tensor& frame);
};

} // namespace temporal
} // namespace jin