#include "optim.h"
#include "../autodiff/grad_registry.h"
#include "../autodiff/backward.h"
#include "../autodiff/ops.h"
#include "../core/math_ops.h"
#include <algorithm>
#include <cmath>

namespace jin {
namespace optim {

// ==================== Optimizer ====================

Optimizer::Optimizer(float learning_rate) : learning_rate_(learning_rate) {}

void Optimizer::zero_grad() {
    for (auto& param : params_) {
        autodiff::zero_grad(param);
    }
}

void Optimizer::set_learning_rate(float lr) {
    learning_rate_ = lr;
}

void Optimizer::register_parameter(const Tensor& param) {
    params_.push_back(param);
}

Tensor& Optimizer::grad(const Tensor& param) {
    static Tensor dummy;
    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i].storage().get() == param.storage().get()) {
            return grads_[i];
        }
    }
    return dummy;
}

bool Optimizer::has_grad(const Tensor& param) const {
    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i].storage().get() == param.storage().get()) {
            return params_[i].has_grad();
        }
    }
    return false;
}

// ==================== SGD ====================

class SGDImpl : public Optimizer {
public:
    SGDImpl(float learning_rate, float momentum, float weight_decay);
    
    void step() override;
    
    void register_parameter(const Tensor& param) override;
    
private:
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocities_;
};

SGDImpl::SGDImpl(float learning_rate, float momentum, float weight_decay)
    : Optimizer(learning_rate), momentum_(momentum), weight_decay_(weight_decay) {}

void SGDImpl::register_parameter(const Tensor& param) {
    Optimizer::register_parameter(param);
    Tensor vel(param.shape(), param.dtype(), param.device());
    vel.fill(0.0f);
    velocities_.push_back(vel);
}

void SGDImpl::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        const Tensor& param = params_[i];
        Tensor& vel = velocities_[i];
        
        if (!param.has_grad()) continue;
        
        Tensor grad = param.grad();
        
        // Weight decay
        if (weight_decay_ > 0.0f) {
            Tensor decay = jin::ones({1}, DType::FP32, Device::cpu());
            decay.data_ptr<float>()[0] = weight_decay_;
            grad = autodiff::add(grad, autodiff::multiply(param, decay));
        }
        
        // Momentum
        if (momentum_ > 0.0f) {
            const float* vel_data = vel.data_ptr<float>();
            const float* grad_data = grad.data_ptr<float>();
            float* vel_out = vel.data_ptr<float>();
            
            for (size_t j = 0; j < vel.size(); ++j) {
                vel_out[j] = momentum_ * vel_data[j] + grad_data[j];
            }
            
            // Update: param = param - lr * velocity
            float* param_data = const_cast<float*>(param.data_ptr<float>());
            for (size_t j = 0; j < param.size(); ++j) {
                param_data[j] -= learning_rate_ * vel_out[j];
            }
        } else {
            // Simple SGD: param = param - lr * grad
            float* param_data = const_cast<float*>(param.data_ptr<float>());
            const float* grad_data = grad.data_ptr<float>();
            
            for (size_t j = 0; j < param.size(); ++j) {
                param_data[j] -= learning_rate_ * grad_data[j];
            }
        }
    }
}

// ==================== Adam ====================

class AdamImpl : public Optimizer {
public:
    AdamImpl(float learning_rate, float beta1, float beta2,
             float eps, float weight_decay);
    
    void step() override;
    
    void register_parameter(const Tensor& param) override;
    
private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    int step_;
    std::vector<Tensor> m_;  // First moment estimate
    std::vector<Tensor> v_;  // Second moment estimate
};

AdamImpl::AdamImpl(float learning_rate, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      eps_(eps),
      weight_decay_(weight_decay),
      step_(0) {}

void AdamImpl::register_parameter(const Tensor& param) {
    Optimizer::register_parameter(param);
    
    Tensor m(param.shape(), param.dtype(), param.device());
    m.fill(0.0f);
    m_.push_back(m);
    
    Tensor v(param.shape(), param.dtype(), param.device());
    v.fill(0.0f);
    v_.push_back(v);
}

void AdamImpl::step() {
    step_++;
    
    for (size_t i = 0; i < params_.size(); ++i) {
        const Tensor& param = params_[i];
        Tensor& m = m_[i];
        Tensor& v = v_[i];
        
        if (!param.has_grad()) continue;
        
        Tensor grad = param.grad();
        
        // Weight decay (L2 regularization)
        if (weight_decay_ > 0.0f) {
            Tensor decay = jin::ones({1}, DType::FP32, Device::cpu());
            decay.data_ptr<float>()[0] = weight_decay_;
            grad = autodiff::add(grad, autodiff::multiply(param, decay));
        }
        
        // Update biased first moment estimate
        const float* m_data = m.data_ptr<float>();
        const float* grad_data = grad.data_ptr<float>();
        float* m_out = m.data_ptr<float>();
        
        for (size_t j = 0; j < m.size(); ++j) {
            m_out[j] = beta1_ * m_data[j] + (1.0f - beta1_) * grad_data[j];
        }
        
        // Update biased second raw moment estimate
        const float* v_data = v.data_ptr<float>();
        float* v_out = v.data_ptr<float>();
        
        for (size_t j = 0; j < v.size(); ++j) {
            float g = grad_data[j];
            v_out[j] = beta2_ * v_data[j] + (1.0f - beta2_) * g * g;
        }
        
        // Compute bias-corrected first moment estimate
        float m_hat_scale = 1.0f / (1.0f - std::pow(beta1_, static_cast<float>(step_)));
        
        // Compute bias-corrected second raw moment estimate
        float v_hat_scale = 1.0f / (1.0f - std::pow(beta2_, static_cast<float>(step_)));
        
        // Update parameters
        float* param_data = const_cast<float*>(param.data_ptr<float>());
        for (size_t j = 0; j < param.size(); ++j) {
            float m_corrected = m_out[j] * m_hat_scale;
            float v_corrected = v_out[j] * v_hat_scale;
            
            // Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
            param_data[j] -= learning_rate_ * m_corrected / (std::sqrt(v_corrected) + eps_);
        }
    }
}

// ==================== Lion ====================

class LionImpl : public Optimizer {
public:
    LionImpl(float learning_rate, float beta, float weight_decay);
    
    void step() override;
    
    void register_parameter(const Tensor& param) override;
    
private:
    float beta_;
    float weight_decay_;
    std::vector<Tensor> updates_;
};

LionImpl::LionImpl(float learning_rate, float beta, float weight_decay)
    : Optimizer(learning_rate), beta_(beta), weight_decay_(weight_decay) {}

void LionImpl::register_parameter(const Tensor& param) {
    Optimizer::register_parameter(param);
    Tensor update(param.shape(), param.dtype(), param.device());
    update.fill(0.0f);
    updates_.push_back(update);
}

void LionImpl::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        const Tensor& param = params_[i];
        Tensor& update = updates_[i];
        
        if (!param.has_grad()) continue;
        
        Tensor grad = param.grad();
        
        // Weight decay (sign-based)
        if (weight_decay_ > 0.0f) {
            Tensor sign = ops::sign(param);
            Tensor decay = jin::ones({1}, DType::FP32, Device::cpu());
            decay.data_ptr<float>()[0] = weight_decay_;
            grad = autodiff::add(grad, autodiff::multiply(sign, decay));
        }
        
        // Sign-based update
        const float* update_data = update.data_ptr<float>();
        const float* grad_data = grad.data_ptr<float>();
        float* update_out = update.data_ptr<float>();
        float* param_data = const_cast<float*>(param.data_ptr<float>());
        
        for (size_t j = 0; j < param.size(); ++j) {
            // momentum update
            update_out[j] = beta_ * update_data[j] + grad_data[j];
            
            // sign update
            param_data[j] -= learning_rate_ * (update_out[j] >= 0.0f ? 1.0f : -1.0f);
        }
    }
}

// ==================== Factory Functions ====================

std::unique_ptr<Optimizer> create_sgd(float lr, float momentum, float wd) {
    return std::make_unique<SGDImpl>(lr, momentum, wd);
}

std::unique_ptr<Optimizer> create_adam(float lr, float beta1, float beta2,
                                        float eps, float wd) {
    return std::make_unique<AdamImpl>(lr, beta1, beta2, eps, wd);
}

std::unique_ptr<Optimizer> create_lion(float lr, float beta, float wd) {
    return std::make_unique<LionImpl>(lr, beta, wd);
}

// ==================== LR Schedulers ====================

StepLR::StepLR(float initial_lr, int step_size, float gamma)
    : initial_lr_(initial_lr), step_size_(step_size), gamma_(gamma) {}

float StepLR::get_lr(int step) const {
    int num_steps = step / step_size_;
    return initial_lr_ * std::pow(gamma_, static_cast<float>(num_steps));
}

MultiStepLR::MultiStepLR(float initial_lr, const std::vector<int>& milestones, float gamma)
    : initial_lr_(initial_lr), milestones_(milestones), gamma_(gamma) {}

float MultiStepLR::get_lr(int step) const {
    int num_milestones = 0;
    for (auto& m : milestones_) {
        if (step >= m) num_milestones++;
    }
    return initial_lr_ * std::pow(gamma_, static_cast<float>(num_milestones));
}

CosineAnnealingLR::CosineAnnealingLR(float initial_lr, int T_max, float eta_min)
    : initial_lr_(initial_lr), T_max_(T_max), eta_min_(eta_min) {}

float CosineAnnealingLR::get_lr(int step) const {
    return eta_min_ + (initial_lr_ - eta_min_) * 
           (1.0f + std::cos(static_cast<float>(step) / static_cast<float>(T_max_) * M_PI)) / 2.0f;
}

ReduceLROnPlateau::ReduceLROnPlateau(float initial_lr, float factor, int patience, float min_lr)
    : initial_lr_(initial_lr), factor_(factor), patience_(patience), min_lr_(min_lr),
      best_loss_(std::numeric_limits<float>::infinity()), num_bad_steps_(0) {}

float ReduceLROnPlateau::get_lr(int) const {
    return initial_lr_;
}

void ReduceLROnPlateau::step(float loss) {
    if (loss < best_loss_) {
        best_loss_ = loss;
        num_bad_steps_ = 0;
    } else {
        num_bad_steps_++;
        if (num_bad_steps_ >= patience_) {
            initial_lr_ = std::max(initial_lr_ * factor_, min_lr_);
            num_bad_steps_ = 0;
        }
    }
}

} // namespace optim
} // namespace jin
