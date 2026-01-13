#pragma once

#include "../core/tensor.h"
#include "common.h"

namespace jin {
namespace nn {

enum class Reduction {
    None,
    Mean,
    Sum
};

class CrossEntropyLoss {
public:
    CrossEntropyLoss(Reduction reduction = Reduction::Mean, float label_smoothing = 0.0f);
    
    Tensor forward(const Tensor& logits, const Tensor& targets);
    
    // Properties
    Reduction reduction() const { return reduction_; }
    float label_smoothing() const { return label_smoothing_; }

private:
    Reduction reduction_;
    float label_smoothing_;
};

class MSELoss {
public:
    MSELoss(Reduction reduction = Reduction::Mean);
    
    Tensor forward(const Tensor& pred, const Tensor& target);
    
    Reduction reduction() const { return reduction_; }

private:
    Reduction reduction_;
};

class L1Loss {
public:
    L1Loss(Reduction reduction = Reduction::Mean);
    
    Tensor forward(const Tensor& pred, const Tensor& target);
    
    Reduction reduction() const { return reduction_; }

private:
    Reduction reduction_;
};

class BCELoss {
public:
    BCELoss(Reduction reduction = Reduction::Mean);
    
    Tensor forward(const Tensor& pred, const Tensor& target);
    
    Reduction reduction() const { return reduction_; }

private:
    Reduction reduction_;
};

class BCEWithLogitsLoss {
public:
    BCEWithLogitsLoss(Reduction reduction = Reduction::Mean);
    
    Tensor forward(const Tensor& logits, const Tensor& targets);
    
    Reduction reduction() const { return reduction_; }

private:
    Reduction reduction_;
};

// Functional API
Tensor cross_entropy(const Tensor& logits, const Tensor& targets, Reduction reduction = Reduction::Mean);
Tensor mse_loss(const Tensor& pred, const Tensor& target, Reduction reduction = Reduction::Mean);
Tensor l1_loss(const Tensor& pred, const Tensor& target, Reduction reduction = Reduction::Mean);

} // namespace nn
} // namespace jin
