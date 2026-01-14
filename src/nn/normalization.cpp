#include "normalization.h"
#include "../autodiff/ops.h"
#include "../autodiff/tape.h"
#include "../core/math_ops.h"
#include <cmath>

namespace jin {
namespace nn {

// ==================== LayerNorm ====================

LayerNorm::LayerNorm(size_t normalized_shape, float eps)
    : eps_(eps), mode_(ForwardMode::Train) {
    
    normalized_shape_ = Shape{normalized_shape};
    
    weight_ = jin::ones(normalized_shape_, DType::FP32, Device::cpu());
    weight_.set_requires_grad(true);
    
    bias_ = jin::zeros(normalized_shape_, DType::FP32, Device::cpu());
    bias_.set_requires_grad(true);
}

Tensor LayerNorm::forward(const Tensor& x) {
    autodiff::TapeGuard tape_guard;
    
    size_t ndim = x.ndim();
    size_t norm_ndim = normalized_shape_.size();
    
    if (norm_ndim > ndim) {
        throw std::invalid_argument("normalized_shape cannot be larger than input dimensions");
    }
    
    int axis = static_cast<int>(ndim - norm_ndim);
    Shape input_shape = x.shape();
    
    size_t num_elements = 1;
    for (size_t i = 0; i < norm_ndim; ++i) {
        num_elements *= input_shape[axis + static_cast<int>(i)];
    }
    
    // Compute mean
    Tensor x_sum = ops::sum(x, axis, true);
    Tensor num_elem_t = jin::ones({num_elements}, DType::FP32, Device::cpu());
    float* ne_data = num_elem_t.data_ptr<float>();
    for (size_t i = 0; i < num_elements; ++i) ne_data[i] = static_cast<float>(num_elements);
    Tensor mean = ops::divide(x_sum, num_elem_t);
    
    // Compute variance: E[(x - mean)^2]
    Tensor x_centered = ops::subtract(x, mean);
    Tensor x_sq = ops::multiply(x_centered, x_centered);
    Tensor var_sum = ops::sum(x_sq, axis, true);
    Tensor var = ops::divide(var_sum, num_elem_t);
    
    // Compute std with epsilon
    Tensor eps_tensor = jin::ones({1}, DType::FP32, Device::cpu());
    eps_tensor.data_ptr<float>()[0] = eps_;
    Tensor std = ops::sqrt(ops::add(var, eps_tensor));
    
    // Normalize: (x - mean) / std
    Tensor normalized = ops::divide(x_centered, std);
    
    // Scale and shift: gamma * normalized + beta
    Tensor weighted = ops::multiply(normalized, weight_);
    Tensor out = ops::add(weighted, bias_);
    
    return out;
}

// ==================== RMSNorm ====================

RMSNorm::RMSNorm(size_t normalized_shape, float eps)
    : eps_(eps), mode_(ForwardMode::Train) {
    
    normalized_shape_ = Shape{normalized_shape};
    
    weight_ = jin::ones(normalized_shape_, DType::FP32, Device::cpu());
    weight_.set_requires_grad(true);
}

Tensor RMSNorm::forward(const Tensor& x) {
    autodiff::TapeGuard tape_guard;
    
    size_t ndim = x.ndim();
    size_t norm_ndim = normalized_shape_.size();
    
    if (norm_ndim > ndim) {
        throw std::invalid_argument("normalized_shape cannot be larger than input dimensions");
    }
    
    int axis = static_cast<int>(ndim - norm_ndim);
    Shape input_shape = x.shape();
    
    size_t num_elements = 1;
    for (size_t i = 0; i < norm_ndim; ++i) {
        num_elements *= input_shape[axis + static_cast<int>(i)];
    }
    
    // Compute mean of squares: E[x^2]
    Tensor x_sq = ops::multiply(x, x);
    Tensor mean_sq_sum = ops::sum(x_sq, axis, true);
    Tensor num_elem_t = jin::ones({num_elements}, DType::FP32, Device::cpu());
    float* ne_data = num_elem_t.data_ptr<float>();
    for (size_t i = 0; i < num_elements; ++i) ne_data[i] = static_cast<float>(num_elements);
    Tensor mean_sq = ops::divide(mean_sq_sum, num_elem_t);
    
    // Compute RMS: sqrt(mean(x^2) + eps)
    Tensor eps_tensor = jin::ones({1}, DType::FP32, Device::cpu());
    eps_tensor.data_ptr<float>()[0] = eps_;
    Tensor rms = ops::sqrt(ops::add(mean_sq, eps_tensor));
    
    // Normalize: x / rms
    Tensor normalized = ops::divide(x, rms);
    
    // Scale: gamma * normalized
    Tensor out = ops::multiply(normalized, weight_);
    
    return out;
}

// ==================== BatchNorm1d ====================

BatchNorm1d::BatchNorm1d(size_t num_features, float momentum, float eps)
    : num_features_(num_features),
      momentum_(momentum),
      eps_(eps),
      is_training_(true) {
    
    weight_ = jin::ones({num_features}, DType::FP32, Device::cpu());
    weight_.set_requires_grad(true);
    
    bias_ = jin::zeros({num_features}, DType::FP32, Device::cpu());
    bias_.set_requires_grad(true);
    
    running_mean_ = jin::zeros({num_features}, DType::FP32, Device::cpu());
    running_var_ = jin::ones({num_features}, DType::FP32, Device::cpu());
}

Tensor BatchNorm1d::forward(const Tensor& x) {
    Shape x_shape = x.shape();
    size_t dim = x_shape.size();
    
    if (dim < 2) {
        throw std::invalid_argument("BatchNorm1d requires at least 2D input");
    }
    
    if (x_shape[1] != num_features_) {
        throw std::invalid_argument("Input channel dimension must match num_features");
    }
    
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight_.data_ptr<float>();
    const float* bias_data = bias_.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    if (is_training_) {
        if (dim == 2) {
            // [N, C]
            for (size_t c = 0; c < num_features_; ++c) {
                float mean = 0.0f;
                for (size_t n = 0; n < x_shape[0]; ++n) {
                    mean += x_data[n * num_features_ + c];
                }
                mean /= static_cast<float>(x_shape[0]);
                
                float var = 0.0f;
                for (size_t n = 0; n < x_shape[0]; ++n) {
                    float diff = x_data[n * num_features_ + c] - mean;
                    var += diff * diff;
                }
                var /= static_cast<float>(x_shape[0]);
                
                float* running_mean_data = running_mean_.data_ptr<float>();
                float* running_var_data = running_var_.data_ptr<float>();
                running_mean_data[c] = momentum_ * running_mean_data[c] + (1.0f - momentum_) * mean;
                running_var_data[c] = momentum_ * running_var_data[c] + (1.0f - momentum_) * var;
                
                float std = std::sqrt(var + eps_);
                
                for (size_t n = 0; n < x_shape[0]; ++n) {
                    float val = x_data[n * num_features_ + c];
                    float norm = (val - mean) / std;
                    out_data[n * num_features_ + c] = norm * weight_data[c] + bias_data[c];
                }
            }
        } else {
            size_t L = x_shape[2];
            size_t N = x_shape[0];
            
            for (size_t c = 0; c < num_features_; ++c) {
                float mean = 0.0f;
                for (size_t n = 0; n < N; ++n) {
                    for (size_t l = 0; l < L; ++l) {
                        mean += x_data[(n * num_features_ + c) * L + l];
                    }
                }
                mean /= static_cast<float>(N * L);
                
                float var = 0.0f;
                for (size_t n = 0; n < N; ++n) {
                    for (size_t l = 0; l < L; ++l) {
                        float diff = x_data[(n * num_features_ + c) * L + l] - mean;
                        var += diff * diff;
                    }
                }
                var /= static_cast<float>(N * L);
                
                float* running_mean_data = running_mean_.data_ptr<float>();
                float* running_var_data = running_var_.data_ptr<float>();
                running_mean_data[c] = momentum_ * running_mean_data[c] + (1.0f - momentum_) * mean;
                running_var_data[c] = momentum_ * running_var_data[c] + (1.0f - momentum_) * var;
                
                float std = std::sqrt(var + eps_);
                
                for (size_t n = 0; n < N; ++n) {
                    for (size_t l = 0; l < L; ++l) {
                        float val = x_data[(n * num_features_ + c) * L + l];
                        float norm = (val - mean) / std;
                        out_data[(n * num_features_ + c) * L + l] = norm * weight_data[c] + bias_data[c];
                    }
                }
            }
        }
    } else {
        const float* running_mean_data = running_mean_.data_ptr<float>();
        const float* running_var_data = running_var_.data_ptr<float>();
        
        if (dim == 2) {
            for (size_t c = 0; c < num_features_; ++c) {
                float mean = running_mean_data[c];
                float std = std::sqrt(running_var_data[c] + eps_);
                
                for (size_t n = 0; n < x_shape[0]; ++n) {
                    float val = x_data[n * num_features_ + c];
                    float norm = (val - mean) / std;
                    out_data[n * num_features_ + c] = norm * weight_data[c] + bias_data[c];
                }
            }
        } else {
            size_t L = x_shape[2];
            size_t N = x_shape[0];
            
            for (size_t c = 0; c < num_features_; ++c) {
                float mean = running_mean_data[c];
                float std = std::sqrt(running_var_data[c] + eps_);
                
                for (size_t n = 0; n < N; ++n) {
                    for (size_t l = 0; l < L; ++l) {
                        float val = x_data[(n * num_features_ + c) * L + l];
                        float norm = (val - mean) / std;
                        out_data[(n * num_features_ + c) * L + l] = norm * weight_data[c] + bias_data[c];
                    }
                }
            }
        }
    }
    
    return out;
}

void BatchNorm1d::train() {
    is_training_ = true;
}

void BatchNorm1d::eval() {
    is_training_ = false;
}

// ==================== BatchNorm2d ====================

BatchNorm2d::BatchNorm2d(size_t num_features, float momentum, float eps)
    : num_features_(num_features),
      momentum_(momentum),
      eps_(eps),
      is_training_(true) {
    
    weight_ = jin::ones({num_features}, DType::FP32, Device::cpu());
    weight_.set_requires_grad(true);
    
    bias_ = jin::zeros({num_features}, DType::FP32, Device::cpu());
    bias_.set_requires_grad(true);
    
    running_mean_ = jin::zeros({num_features}, DType::FP32, Device::cpu());
    running_var_ = jin::ones({num_features}, DType::FP32, Device::cpu());
}

Tensor BatchNorm2d::forward(const Tensor& x) {
    Shape x_shape = x.shape();
    
    if (x_shape.size() != 4) {
        throw std::invalid_argument("BatchNorm2d requires 4D input [N, C, H, W]");
    }
    
    if (x_shape[1] != num_features_) {
        throw std::invalid_argument("Input channel dimension must match num_features");
    }
    
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight_.data_ptr<float>();
    const float* bias_data = bias_.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    size_t N = x_shape[0];
    size_t C = x_shape[1];
    size_t H = x_shape[2];
    size_t W = x_shape[3];
    size_t spatial_size = H * W;
    
    if (is_training_) {
        for (size_t c = 0; c < C; ++c) {
            float mean = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        mean += x_data[((n * C + c) * H + h) * W + w];
                    }
                }
            }
            mean /= static_cast<float>(N * spatial_size);
            
            float var = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        float diff = x_data[((n * C + c) * H + h) * W + w] - mean;
                        var += diff * diff;
                    }
                }
            }
            var /= static_cast<float>(N * spatial_size);
            
            float* running_mean_data = running_mean_.data_ptr<float>();
            float* running_var_data = running_var_.data_ptr<float>();
            running_mean_data[c] = momentum_ * running_mean_data[c] + (1.0f - momentum_) * mean;
            running_var_data[c] = momentum_ * running_var_data[c] + (1.0f - momentum_) * var;
            
            float std = std::sqrt(var + eps_);
            
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        float val = x_data[((n * C + c) * H + h) * W + w];
                        float norm = (val - mean) / std;
                        out_data[((n * C + c) * H + h) * W + w] = norm * weight_data[c] + bias_data[c];
                    }
                }
            }
        }
    } else {
        const float* running_mean_data = running_mean_.data_ptr<float>();
        const float* running_var_data = running_var_.data_ptr<float>();
        
        for (size_t c = 0; c < C; ++c) {
            float mean = running_mean_data[c];
            float std = std::sqrt(running_var_data[c] + eps_);
            
            for (size_t n = 0; n < N; ++n) {
                for (size_t h = 0; h < H; ++h) {
                    for (size_t w = 0; w < W; ++w) {
                        float val = x_data[((n * C + c) * H + h) * W + w];
                        float norm = (val - mean) / std;
                        out_data[((n * C + c) * H + h) * W + w] = norm * weight_data[c] + bias_data[c];
                    }
                }
            }
        }
    }
    
    return out;
}

void BatchNorm2d::train() {
    is_training_ = true;
}

void BatchNorm2d::eval() {
    is_training_ = false;
}

} // namespace nn
} // namespace jin
