#pragma once

#include "../core/tensor.h"
#include "common.h"

namespace jin {
namespace nn {

class Dropout {
public:
    Dropout(float p = 0.5f);
    
    Tensor forward(const Tensor& x);
    
    // Properties
    float p() const { return p_; }
    float train_p() const { return 1.0f - p_; }
    
    void train() { mode_ = ForwardMode::Train; }
    void eval() { mode_ = ForwardMode::Eval; }
    bool is_training() const { return mode_ == ForwardMode::Train; }

private:
    float p_;
    ForwardMode mode_;
};

} // namespace nn
} // namespace jin
