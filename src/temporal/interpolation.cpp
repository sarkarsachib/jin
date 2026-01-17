#include "interpolation.h"

namespace jin {
namespace temporal {

MotionInterpolator::MotionInterpolator() : default_num_steps_(4) {}

std::vector<Tensor> MotionInterpolator::interpolate_frames(
    const Tensor& frame_a, 
    const Tensor& frame_b, 
    int num_steps,
    const std::string& method
) {
    if (frame_a.shape() != frame_b.shape()) {
        throw std::invalid_argument("Frames must have the same shape");
    }
    
    std::vector<Tensor> interpolated_frames;
    
    if (method == "linear") {
        interpolated_frames = linear_interpolation(frame_a, frame_b, num_steps);
    } else if (method == "cubic") {
        interpolated_frames = cubic_interpolation(frame_a, frame_b, num_steps);
    } else {
        throw std::invalid_argument("Unknown interpolation method: " + method);
    }
    
    return interpolated_frames;
}

Tensor MotionInterpolator::blend_motion(const Tensor& frame1, const Tensor& frame2, float t) {
    if (frame1.shape() != frame2.shape()) {
        throw std::invalid_argument("Frames must have the same shape");
    }
    
    t = std::clamp(t, 0.0f, 1.0f);
    return frame1 * (1.0f - t) + frame2 * t;
}

std::vector<Tensor> MotionInterpolator::adaptive_insert(
    const std::vector<Tensor>& frames,
    float target_fps,
    float source_fps
) {
    if (frames.empty()) return {};
    
    if (source_fps <= 0) return frames;
    
    float fps_ratio = target_fps / source_fps;
    if (fps_ratio <= 1.0f) {
        // Need to remove frames (downsampling)
        int keep_every = static_cast<int>(std::ceil(1.0f / fps_ratio));
        std::vector<Tensor> downsampled;
        for (size_t i = 0; i < frames.size(); i += keep_every) {
            downsampled.push_back(frames[i]);
        }
        return downsampled;
    } else {
        // Need to insert frames (upsampling)
        std::vector<Tensor> upsampled;
        upsampled.reserve(static_cast<size_t>(frames.size() * fps_ratio));
        
        for (size_t i = 0; i < frames.size() - 1; ++i) {
            upsampled.push_back(frames[i]);
            
            int num_insertions = static_cast<int>(std::floor(fps_ratio)) - 1;
            for (int j = 1; j <= num_insertions; ++j) {
                float t = static_cast<float>(j) / (static_cast<float>(num_insertions) + 1.0f);
                Tensor inserted = blend_motion(frames[i], frames[i + 1], t);
                upsampled.push_back(inserted);
            }
        }
        
        // Add last frame
        upsampled.push_back(frames.back());
        
        return upsampled;
    }
}

Tensor MotionInterpolator::warp_frame(const Tensor& frame, const Tensor& motion_field, float alpha) {
    if (motion_field.shape().size() != 3 || motion_field.shape()[2] != 2) {
        throw std::invalid_argument("Motion field must be [H, W, 2]");
    }
    
    auto frame_shape = frame.shape();
    if (frame_shape.size() != 3) {
        throw std::invalid_argument("Frame must be 3D tensor [C, H, W]");
    }
    
    int channels = frame_shape[0];
    int height = frame_shape[1];
    int width = frame_shape[2];
    
    if (motion_field.shape()[0] != height || motion_field.shape()[1] != width) {
        throw std::invalid_argument("Motion field spatial dimensions must match frame");
    }
    
    alpha = std::clamp(alpha, 0.0f, 1.0f);
    
    Tensor warped = zeros(frame_shape);
    
    // Apply bilinear sampling using motion vectors
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float flow_x = motion_field.data_ptr<float>()[y * width * 2 + x * 2] * alpha;
            float flow_y = motion_field.data_ptr<float>()[y * width * 2 + x * 2 + 1] * alpha;
            
            float src_x = x + flow_x;
            float src_y = y + flow_y;
            
            // Clamp to valid range
            src_x = std::clamp(src_x, 0.0f, static_cast<float>(width - 1));
            src_y = std::clamp(src_y, 0.0f, static_cast<float>(height - 1));
            
            // Bilinear interpolation
            int x0 = static_cast<int>(std::floor(src_x));
            int y0 = static_cast<int>(std::floor(src_y));
            int x1 = std::min(x0 + 1, width - 1);
            int y1 = std::min(y0 + 1, height - 1);
            
            float dx = src_x - x0;
            float dy = src_y - y0;
            
            for (int c = 0; c < channels; ++c) {
                float val00 = frame.data_ptr<float>()[c * height * width + y0 * width + x0];
                float val01 = frame.data_ptr<float>()[c * height * width + y1 * width + x0];
                float val10 = frame.data_ptr<float>()[c * height * width + y0 * width + x1];
                float val11 = frame.data_ptr<float>()[c * height * width + y1 * width + x1];
                
                float val0 = val00 * (1.0f - dx) + val10 * dx;
                float val1 = val01 * (1.0f - dx) + val11 * dx;
                float val = val0 * (1.0f - dy) + val1 * dy;
                
                warped.data_ptr<float>()[c * height * width + y * width + x] = val;
            }
        }
    }
    
    return warped;
}

float MotionInterpolator::compute_interpolation_quality(
    const std::vector<Tensor>& original_frames,
    const std::vector<Tensor>& interpolated_frames
) {
    if (original_frames.empty() || interpolated_frames.empty()) {
        return 0.0f;
    }
    
    // Compute temporal gradients
    float total_temporal_gradient = 0.0f;
    int num_gradients = 0;
    
    // Original sequence temporal gradient
    for (size_t i = 1; i < original_frames.size(); ++i) {
        Tensor diff = original_frames[i] - original_frames[i-1];
        Tensor grad_mag = jin::ops::sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)));
        total_temporal_gradient += grad_mag.item();
        num_gradients++;
    }
    
    // Interpolated sequence temporal gradient
    for (size_t i = 1; i < interpolated_frames.size(); ++i) {
        Tensor diff = interpolated_frames[i] - interpolated_frames[i-1];
        Tensor grad_mag = jin::ops::sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)));
        total_temporal_gradient += grad_mag.item();
        num_gradients++;
    }
    
    float avg_temporal_gradient = total_temporal_gradient / num_gradients;
    
    // Lower temporal gradient indicates smoother interpolation
    return avg_temporal_gradient;
}

std::vector<Tensor> MotionInterpolator::linear_interpolation(const Tensor& frame_a, const Tensor& frame_b, int num_steps) {
    std::vector<Tensor> result;
    result.reserve(num_steps + 2);
    
    // Add start frame
    result.push_back(frame_a);
    
    // Add interpolated frames
    for (int i = 1; i <= num_steps; ++i) {
        float t = static_cast<float>(i) / (num_steps + 1);
        Tensor interpolated = blend_motion(frame_a, frame_b, t);
        result.push_back(interpolated);
    }
    
    // Add end frame
    result.push_back(frame_b);
    
    return result;
}

std::vector<Tensor> MotionInterpolator::cubic_interpolation(const Tensor& frame_a, const Tensor& frame_b, int num_steps) {
    std::vector<Tensor> result;
    result.reserve(num_steps + 2);
    
    // For cubic interpolation, we need more context
    // For now, use Hermite interpolation (simplified cubic)
    Tensor derivative_a = compute_frame_derivative(frame_a);
    Tensor derivative_b = compute_frame_derivative(frame_b);
    
    // Add start frame
    result.push_back(frame_a);
    
    // Add interpolated frames using Hermite basis
    for (int i = 1; i <= num_steps; ++i) {
        float t = static_cast<float>(i) / (num_steps + 1);
        Tensor interpolated = hermite_interpolate(frame_a, frame_b, derivative_a, derivative_b, t);
        result.push_back(interpolated);
    }
    
    // Add end frame
    result.push_back(frame_b);
    
    return result;
}

Tensor MotionInterpolator::hermite_interpolate(
    const Tensor& p0, const Tensor& p1,
    const Tensor& m0, const Tensor& m1, float t
) {
    float t2 = t * t;
    float t3 = t2 * t;
    
    // Hermite basis functions
    float h00 = 2 * t3 - 3 * t2 + 1;
    float h10 = t3 - 2 * t2 + t;
    float h01 = -2 * t3 + 3 * t2;
    float h11 = t3 - t2;
    
    return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;
}

Tensor MotionInterpolator::compute_frame_derivative(const Tensor& frame) {
    auto shape = frame.shape();
    if (shape.size() != 3) return frame; // Can't compute derivative for non-3D tensors
    
    Tensor derivative = zeros(shape);
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    // Simple gradient computation
    for (int c = 0; c < channels; ++c) {
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                float center = frame.data_ptr<float>()[c * height * width + y * width + x];
                float right = frame.data_ptr<float>()[c * height * width + y * width + (x + 1)];
                float bottom = frame.data_ptr<float>()[c * height * width + (y + 1) * width + x];
                
                float grad_x = (right - center);
                float grad_y = (bottom - center);
                
                derivative.data_ptr<float>()[c * height * width + y * width + x] = 
                    std::sqrt(grad_x * grad_x + grad_y * grad_y);
            }
        }
    }
    
    return derivative;
}

} // namespace temporal
} // namespace jin