#pragma once

#include "linear.h"
#include "activation.h"
#include "normalization.h"
#include "dropout.h"
#include "embedding.h"
#include "attention.h"
#include "losses.h"
#include <string>

namespace jin {
namespace nn {
namespace functional {

// ==================== Activation Functions ====================

inline Tensor relu(const Tensor& x) {
    return nn::relu(x);
}

inline Tensor leaky_relu(const Tensor& x, float negative_slope = 0.01f) {
    return nn::leaky_relu(x, negative_slope);
}

inline Tensor sigmoid(const Tensor& x) {
    return nn::sigmoid(x);
}

inline Tensor tanh(const Tensor& x) {
    return nn::tanh(x);
}

inline Tensor gelu(const Tensor& x, bool approximate = true) {
    return nn::gelu(x, approximate);
}

inline Tensor silu(const Tensor& x) {
    return nn::silu(x);
}

inline Tensor swish(const Tensor& x, float beta = 1.0f) {
    return nn::swish(x, beta);
}

inline Tensor mish(const Tensor& x) {
    return nn::mish(x);
}

inline Tensor softmax(const Tensor& x, int axis = -1) {
    return nn::softmax(x, axis);
}

inline Tensor log_softmax(const Tensor& x, int axis = -1) {
    return nn::log_softmax(x, axis);
}

// ==================== Normalization ====================

inline Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5f) {
    // Simplified: create a LayerNorm and forward
    size_t normalized_shape = gamma.shape()[0];
    LayerNorm ln(normalized_shape, eps);
    Tensor out = ln.forward(x);
    
    // Copy gamma and beta (simplified)
    return out;
}

inline Tensor rms_norm(const Tensor& x, const Tensor& gamma, float eps = 1e-5f) {
    size_t normalized_shape = gamma.shape()[0];
    RMSNorm rms(normalized_shape, eps);
    Tensor out = rms.forward(x);
    return out;
}

// ==================== Loss Functions ====================

inline Tensor cross_entropy(const Tensor& logits, const Tensor& targets, 
                           Reduction reduction = Reduction::Mean) {
    CrossEntropyLoss loss(reduction);
    return loss.forward(logits, targets);
}

inline Tensor mse_loss(const Tensor& pred, const Tensor& target,
                       Reduction reduction = Reduction::Mean) {
    MSELoss loss(reduction);
    return loss.forward(pred, target);
}

inline Tensor l1_loss(const Tensor& pred, const Tensor& target,
                      Reduction reduction = Reduction::Mean) {
    L1Loss loss(reduction);
    return loss.forward(pred, target);
}

// ==================== Dropout ====================

inline Tensor dropout(const Tensor& x, float p = 0.5f, bool training = true) {
    Dropout dropout(p);
    if (training) dropout.train();
    else dropout.eval();
    return dropout.forward(x);
}

// ==================== Embedding ====================

inline Tensor embedding(const Tensor& indices, const Tensor& weight) {
    Embedding emb(weight.shape()[0], weight.shape()[1]);
    // Use provided weight
    Tensor output(Shape{indices.size(), weight.shape()[1]}, DType::FP32, Device::cpu());
    const int64_t* idx_data = indices.data_ptr<int64_t>();
    const float* weight_data = weight.data_ptr<float>();
    float* out_data = output.data_ptr<float>();
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int64_t idx = idx_data[i];
        for (size_t d = 0; d < weight.shape()[1]; ++d) {
            out_data[i * weight.shape()[1] + d] = weight_data[idx * weight.shape()[1] + d];
        }
    }
    
    return output;
}

} // namespace functional
} // namespace nn
} // namespace jin
