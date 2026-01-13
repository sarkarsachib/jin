#pragma once

#include "common.h"
#include "../core/tensor.h"

namespace jin {
namespace nn {

class Linear {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true,
           Initializer weight_init = Initializer::kaiming_normal(2.0f),
           Initializer bias_init = Initializer::zeros());
    
    Tensor forward(const Tensor& input);
    
    // Properties
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    
    // Set learning mode (affects dropout, etc.)
    void train() { mode_ = ForwardMode::Train; }
    void eval() { mode_ = ForwardMode::Eval; }
    bool is_training() const { return mode_ == ForwardMode::Train; }

private:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    Tensor weight_;
    Tensor bias_;
    ForwardMode mode_;
};

} // namespace nn
} // namespace jin
