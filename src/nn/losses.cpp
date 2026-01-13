#include "losses.h"
#include "../autodiff/ops.h"
#include "../autodiff/tape.h"
#include "../core/math_ops.h"
#include <cmath>
#include <algorithm>

namespace jin {
namespace nn {

// ==================== CrossEntropyLoss ====================

CrossEntropyLoss::CrossEntropyLoss(Reduction reduction, float label_smoothing)
    : reduction_(reduction), label_smoothing_(label_smoothing) {
    if (label_smoothing < 0.0f || label_smoothing > 1.0f) {
        throw std::invalid_argument("Label smoothing must be in [0, 1]");
    }
}

Tensor CrossEntropyLoss::forward(const Tensor& logits, const Tensor& targets) {
    autodiff::TapeGuard tape_guard;
    
    Shape logits_shape = logits.shape();
    size_t batch_size = logits_shape[0];
    size_t num_classes = logits_shape[1];
    
    // Log softmax
    Tensor max_logits = ops::sum(logits, 1, true);
    Tensor exp_logits = ops::exp(ops::subtract(logits, max_logits));
    Tensor sum_exp = ops::sum(exp_logits, 1, true);
    Tensor log_softmax = ops::subtract(ops::subtract(logits, max_logits), ops::log(sum_exp));
    
    // Get log probabilities for target classes
    Tensor target_log_probs(logits.shape(), DType::FP32, Device::cpu());
    const float* log_softmax_data = log_softmax.data_ptr<float>();
    const int64_t* target_data = targets.data_ptr<int64_t>();
    float* target_log_data = target_log_probs.data_ptr<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        int64_t target_idx = target_data[b];
        for (size_t c = 0; c < num_classes; ++c) {
            target_log_data[b * num_classes + c] = (c == target_idx) ? 
                log_softmax_data[b * num_classes + c] : 0.0f;
        }
    }
    
    Tensor log_probs = ops::sum(target_log_probs, 1, false);
    
    // Apply label smoothing if needed
    if (label_smoothing_ > 0.0f) {
        Tensor log_uniform = jin::ones({batch_size}, DType::FP32, Device::cpu());
        float* log_uniform_data = log_uniform.data_ptr<float>();
        for (size_t i = 0; i < batch_size; ++i) {
            log_uniform_data[i] = -std::log(static_cast<float>(num_classes));
        }
        
        Tensor scale = jin::ones({1}, DType::FP32, Device::cpu());
        scale.data_ptr<float>()[0] = (1.0f - label_smoothing_);
        
        Tensor smoothed = ops::add(
            ops::multiply(log_probs, scale),
            ops::multiply(log_uniform, scale)
        );
        log_probs = smoothed;
    }
    
    Tensor neg_log_likelihood = ops::neg(log_probs);
    
    Tensor loss;
    if (reduction_ == Reduction::None) {
        loss = neg_log_likelihood;
    } else if (reduction_ == Reduction::Mean) {
        loss = ops::mean(neg_log_likelihood, -1, false);
    } else {
        loss = ops::sum(neg_log_likelihood, -1, false);
    }
    
    return loss;
}

// ==================== MSELoss ====================

MSELoss::MSELoss(Reduction reduction) : reduction_(reduction) {}

Tensor MSELoss::forward(const Tensor& pred, const Tensor& target) {
    autodiff::TapeGuard tape_guard;
    
    Tensor diff = ops::subtract(pred, target);
    Tensor sq_diff = ops::multiply(diff, diff);
    
    Tensor loss;
    if (reduction_ == Reduction::None) {
        loss = sq_diff;
    } else if (reduction_ == Reduction::Mean) {
        loss = ops::mean(sq_diff, -1, false);
    } else {
        loss = ops::sum(sq_diff, -1, false);
    }
    
    return loss;
}

// ==================== L1Loss ====================

L1Loss::L1Loss(Reduction reduction) : reduction_(reduction) {}

Tensor L1Loss::forward(const Tensor& pred, const Tensor& target) {
    autodiff::TapeGuard tape_guard;
    
    Tensor diff = ops::subtract(pred, target);
    Tensor abs_diff = ops::multiply(diff, ops::sign(diff));
    
    Tensor loss;
    if (reduction_ == Reduction::None) {
        loss = abs_diff;
    } else if (reduction_ == Reduction::Mean) {
        loss = ops::mean(abs_diff, -1, false);
    } else {
        loss = ops::sum(abs_diff, -1, false);
    }
    
    return loss;
}

// ==================== BCELoss ====================

BCELoss::BCELoss(Reduction reduction) : reduction_(reduction) {}

Tensor BCELoss::forward(const Tensor& pred, const Tensor& target) {
    autodiff::TapeGuard tape_guard;
    
    Tensor pred_clamped = ops::clamp(pred, 1e-7f, 1.0f - 1e-7f);
    
    Tensor ones_t = jin::ones(target.shape(), DType::FP32, Device::cpu());
    Tensor ones_pred = jin::ones(pred_clamped.shape(), DType::FP32, Device::cpu());
    
    Tensor term1 = ops::multiply(target, ops::log(pred_clamped));
    Tensor term2 = ops::multiply(ops::subtract(ones_t, target),
                                 ops::log(ops::subtract(ones_pred, pred_clamped)));
    
    Tensor losses = ops::neg(ops::add(term1, term2));
    
    Tensor loss;
    if (reduction_ == Reduction::None) {
        loss = losses;
    } else if (reduction_ == Reduction::Mean) {
        loss = ops::mean(losses, -1, false);
    } else {
        loss = ops::sum(losses, -1, false);
    }
    
    return loss;
}

// ==================== BCEWithLogitsLoss ====================

BCEWithLogitsLoss::BCEWithLogitsLoss(Reduction reduction) : reduction_(reduction) {}

Tensor BCEWithLogitsLoss::forward(const Tensor& logits, const Tensor& targets) {
    autodiff::TapeGuard tape_guard;
    
    Tensor zeros = jin::zeros(logits.shape(), DType::FP32, Device::cpu());
    Tensor ones = jin::ones(logits.shape(), DType::FP32, Device::cpu());
    
    Tensor relu_part = ops::relu(ops::neg(logits));
    Tensor exp_part = ops::exp(ops::neg(ops::abs(logits)));
    Tensor log_sigmoid_part = ops::log(ops::add(ones, exp_part));
    
    Tensor losses = ops::add(relu_part, log_sigmoid_part);
    
    Tensor term1 = ops::multiply(targets, logits);
    losses = ops::subtract(losses, term1);
    
    Tensor loss;
    if (reduction_ == Reduction::None) {
        loss = losses;
    } else if (reduction_ == Reduction::Mean) {
        loss = ops::mean(losses, -1, false);
    } else {
        loss = ops::sum(losses, -1, false);
    }
    
    return loss;
}

// ==================== Functional API ====================

Tensor cross_entropy(const Tensor& logits, const Tensor& targets, Reduction reduction) {
    CrossEntropyLoss loss(reduction);
    return loss.forward(logits, targets);
}

Tensor mse_loss(const Tensor& pred, const Tensor& target, Reduction reduction) {
    MSELoss loss(reduction);
    return loss.forward(pred, target);
}

Tensor l1_loss(const Tensor& pred, const Tensor& target, Reduction reduction) {
    L1Loss loss(reduction);
    return loss.forward(pred, target);
}

} // namespace nn
} // namespace jin
