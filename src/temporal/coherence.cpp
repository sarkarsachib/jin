#include "coherence.h"

namespace jin {
namespace temporal {

TemporalCoherence::TemporalCoherence() : sigma_(1.0f) {}

Tensor TemporalCoherence::compute_motion_field(const Tensor& frame_t, const Tensor& frame_t1) {
    if (frame_t.shape() != frame_t1.shape()) {
        throw std::invalid_argument("Frames must have the same shape");
    }
    
    auto shape = frame_t.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("Frame must be 3D tensor [C, H, W]");
    }
    
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    // Compute gradients for both frames
    Tensor grad_t_x = compute_gradient_x(frame_t);
    Tensor grad_t_y = compute_gradient_y(frame_t);
    Tensor grad_t1_x = compute_gradient_x(frame_t1);
    Tensor grad_t1_y = compute_gradient_y(frame_t1);
    
    // Compute temporal gradient
    Tensor temporal_grad = frame_t1 - frame_t;
    
    // Initialize motion field [H, W, 2]
    Shape motion_shape = {static_cast<size_t>(height), static_cast<size_t>(width), 2};
    Tensor motion_field = zeros(motion_shape);
    
    // Solve Lucas-Kanade equation for each pixel
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            // 2x2 system: [Gxx Gxy; Gxy Gyy] * [vx; vy] = [-Gxt; -Gyt]
            float Gxx = 0.0f, Gxy = 0.0f, Gyy = 0.0f;
            float Gxt = 0.0f, Gyt = 0.0f;
            
            // Accumulate over neighborhood (3x3 window)
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int py = y + dy;
                    int px = x + dx;
                    
                    // Average over channels
                    float Ix_t = 0.0f, Iy_t = 0.0f, Ix_t1 = 0.0f, Iy_t1 = 0.0f, It = 0.0f;
                    
                    for (int c = 0; c < channels; ++c) {
                        Ix_t += grad_t_x.data_ptr<float>()[c * height * width + py * width + px];
                        Iy_t += grad_t_y.data_ptr<float>()[c * height * width + py * width + px];
                        Ix_t1 += grad_t1_x.data_ptr<float>()[c * height * width + py * width + px];
                        Iy_t1 += grad_t1_y.data_ptr<float>()[c * height * width + py * width + px];
                        It += temporal_grad.data_ptr<float>()[c * height * width + py * width + px];
                    }
                    
                    Ix_t /= channels;
                    Iy_t /= channels;
                    Ix_t1 /= channels;
                    Iy_t1 /= channels;
                    It /= channels;
                    
                    float Ix = 0.5f * (Ix_t + Ix_t1);
                    float Iy = 0.5f * (Iy_t + Iy_t1);
                    
                    Gxx += Ix * Ix;
                    Gxy += Ix * Iy;
                    Gyy += Iy * Iy;
                    Gxt += Ix * It;
                    Gyt += Iy * It;
                }
            }
            
            // Solve 2x2 system
            float det = Gxx * Gyy - Gxy * Gxy;
            if (std::abs(det) > 1e-6f) {
                float vx = (-Gxt * Gyy + Gxy * Gyt) / det;
                float vy = (-Gxx * Gyt + Gxy * Gxt) / det;
                
                motion_field.data_ptr<float>()[y * width * 2 + x * 2] = vx;
                motion_field.data_ptr<float>()[y * width * 2 + x * 2 + 1] = vy;
            } else {
                // No reliable flow, set to zero
                motion_field.data_ptr<float>()[y * width * 2 + x * 2] = 0.0f;
                motion_field.data_ptr<float>()[y * width * 2 + x * 2 + 1] = 0.0f;
            }
        }
    }
    
    return motion_field;
}

Tensor TemporalCoherence::estimate_optical_flow(const Tensor& frame_a, const Tensor& frame_b) {
    return compute_motion_field(frame_a, frame_b);
}

Tensor TemporalCoherence::smooth_temporal(const std::vector<Tensor>& frames, float sigma) {
    if (frames.empty()) {
        return Tensor();
    }
    
    if (frames.size() == 1) {
        return frames[0];
    }
    
    // Build 1D Gaussian kernel
    int kernel_radius = static_cast<int>(std::ceil(3.0f * sigma));
    std::vector<float> kernel(2 * kernel_radius + 1);
    
    float sum = 0.0f;
    for (int i = -kernel_radius; i <= kernel_radius; ++i) {
        float weight = std::exp(-0.5f * (i * i) / (sigma * sigma));
        kernel[i + kernel_radius] = weight;
        sum += weight;
    }
    
    // Normalize kernel
    for (float& w : kernel) {
        w /= sum;
    }
    
    // Apply temporal smoothing
    std::vector<Tensor> smoothed_frames;
    int num_frames = frames.size();
    auto frame_shape = frames[0].shape();
    
    for (int t = 0; t < num_frames; ++t) {
        Tensor smoothed = zeros(frame_shape);
        
        for (int k = -kernel_radius; k <= kernel_radius; ++k) {
            int src_idx = std::clamp(t + k, 0, num_frames - 1);
            float weight = kernel[k + kernel_radius];
            
            Tensor weighted_frame = frames[src_idx] * weight;
            smoothed = smoothed + weighted_frame;
        }
        
        smoothed_frames.push_back(smoothed);
    }
    
    // Concatenate along time dimension
    return concatenate_temporal(smoothed_frames);
}

float TemporalCoherence::compute_consistency_score(const std::vector<Tensor>& frames) {
    if (frames.size() < 2) {
        return 1.0f;
    }
    
    float total_mse = 0.0f;
    int num_pairs = 0;
    
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        Tensor mse = jin::ops::mean(jin::ops::pow(diff, 2.0f));
        total_mse += mse.item();
        num_pairs++;
    }
    
    float avg_mse = total_mse / num_pairs;
    
    // Convert MSE to consistency score using exponential decay
    // Score = exp(-k * MSE), where k controls sensitivity
    float k = 10.0f;
    return std::exp(-k * avg_mse);
}

Tensor TemporalCoherence::compute_gradient_x(const Tensor& image) {
    auto shape = image.shape();
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    Tensor grad = zeros(shape);
    
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                float left = image.data_ptr<float>()[c * height * width + y * width + (x - 1)];
                float right = image.data_ptr<float>()[c * height * width + y * width + (x + 1)];
                grad.data_ptr<float>()[c * height * width + y * width + x] = (right - left) * 0.5f;
            }
        }
    }
    
    return grad;
}

Tensor TemporalCoherence::compute_gradient_y(const Tensor& image) {
    auto shape = image.shape();
    int channels = shape[0];
    int height = shape[1];
    int width = shape[2];
    
    Tensor grad = zeros(shape);
    
    for (int c = 0; c < channels; ++c) {
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 0; x < width; ++x) {
                float top = image.data_ptr<float>()[c * height * width + (y - 1) * width + x];
                float bottom = image.data_ptr<float>()[c * height * width + (y + 1) * width + x];
                grad.data_ptr<float>()[c * height * width + y * width + x] = (bottom - top) * 0.5f;
            }
        }
    }
    
    return grad;
}

Tensor TemporalCoherence::concatenate_temporal(const std::vector<Tensor>& frames) {
    if (frames.empty()) return Tensor();
    
    auto frame_shape = frames[0].shape();
    int num_frames = frames.size();
    
    // Create [num_frames, C, H, W] tensor
    Shape new_shape = {static_cast<size_t>(num_frames)};
    new_shape.insert(new_shape.end(), frame_shape.begin(), frame_shape.end());
    Tensor result = zeros(new_shape);
    
    for (int t = 0; t < num_frames; ++t) {
        for (size_t i = 0; i < frame_shape.size(); ++i) {
            int frame_size = 1;
            for (size_t j = i + 1; j < frame_shape.size(); ++j) {
                frame_size *= frame_shape[j];
            }
            
            Shape result_slice_shape(frame_shape.begin() + i, frame_shape.end());
            result_slice_shape.insert(result_slice_shape.begin(), 1);
            
            Tensor result_slice = result.slice(
                t * frame_size,
                (t + 1) * frame_size
            ).view(result_slice_shape);
            
            result_slice = frames[t].view(frame_shape);
        }
    }
    
    return result;
}

} // namespace temporal
} // namespace jin