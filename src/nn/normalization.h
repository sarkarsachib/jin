#pragma once

#include "../core/tensor.h"
#include "common.h"

namespace jin {
namespace nn {

class LayerNorm {
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-5f);
    
    Tensor forward(const Tensor& x);
    
    // Properties
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }
    float eps() const { return eps_; }
    
    // Set to training/eval mode (for future batch norm compatibility)
    void train() { mode_ = ForwardMode::Train; }
    void eval() { mode_ = ForwardMode::Eval; }
    
private:
    Shape normalized_shape_;
    Tensor weight_;
    Tensor bias_;
    float eps_;
    ForwardMode mode_;
};

// RMSNorm: Root Mean Square Layer Normalization
// More efficient than LayerNorm - no mean centering
// y = x / sqrt(mean(x^2) + eps) * gamma
class RMSNorm {
public:
    RMSNorm(size_t normalized_shape, float eps = 1e-5f);
    
    Tensor forward(const Tensor& x);
    
    // Properties
    Tensor weight() const { return weight_; }
    float eps() const { return eps_; }
    
    void train() { mode_ = ForwardMode::Train; }
    void eval() { mode_ = ForwardMode::Eval; }
    
private:
    Shape normalized_shape_;
    Tensor weight_;
    float eps_;
    ForwardMode mode_;
};

// Batch Normalization (1D, 2D, 3D)
// For now, implementing basic 1D/2D support
class BatchNorm1d {
public:
    BatchNorm1d(size_t num_features, float momentum = 0.9f, float eps = 1e-5f);
    
    Tensor forward(const Tensor& x);
    
    void train();
    void eval();
    
    // Properties
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }
    
private:
    size_t num_features_;
    float momentum_;
    float eps_;
    Tensor weight_;
    Tensor bias_;
    Tensor running_mean_;
    Tensor running_var_;
    bool is_training_;
};

class BatchNorm2d {
public:
    BatchNorm2d(size_t num_features, float momentum = 0.9f, float eps = 1e-5f);
    
    Tensor forward(const Tensor& x);
    
    void train();
    void eval();
    
private:
    size_t num_features_;
    float momentum_;
    float eps_;
    Tensor weight_;
    Tensor bias_;
    Tensor running_mean_;
    Tensor running_var_;
    bool is_training_;
};

} // namespace nn
} // namespace jin
