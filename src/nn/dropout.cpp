#include "dropout.h"
#include "../autodiff/ops.h"
#include "../autodiff/tape.h"
#include <random>
#include <cmath>

namespace jin {
namespace nn {

Dropout::Dropout(float p) : p_(p), mode_(ForwardMode::Train) {
    if (p < 0.0f || p >= 1.0f) {
        throw std::invalid_argument("Dropout probability must be in [0, 1)");
    }
}

Tensor Dropout::forward(const Tensor& x) {
    if (mode_ == ForwardMode::Eval) {
        return x;
    }
    
    // Training mode: apply dropout
    Tensor out(x.shape(), x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0f - p_);
    
    float scale = 1.0f / (1.0f - p_);
    
    for (size_t i = 0; i < x.size(); ++i) {
        if (dist(gen)) {
            out_data[i] = x_data[i] * scale;
        } else {
            out_data[i] = 0.0f;
        }
    }
    
    return out;
}

} // namespace nn
} // namespace jin
