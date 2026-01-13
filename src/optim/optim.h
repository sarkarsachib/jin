#pragma once

#include "../core/tensor.h"
#include <vector>
#include <string>
#include <memory>

namespace jin {
namespace optim {

class Optimizer {
public:
    explicit Optimizer(float learning_rate = 0.001f);
    virtual ~Optimizer() = default;
    
    virtual void step() = 0;
    virtual void zero_grad();
    
    void set_learning_rate(float lr);
    float learning_rate() const { return learning_rate_; }
    
    // Register a parameter tensor for optimization
    void register_parameter(const Tensor& param);
    
    // Get the list of registered parameters
    const std::vector<Tensor>& parameters() const { return params_; }
    
    // Get gradients for a parameter
    Tensor& grad(const Tensor& param);
    bool has_grad(const Tensor& param) const;
    
protected:
    float learning_rate_;
    std::vector<Tensor> params_;
    std::vector<Tensor> grads_;
};

class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual float get_lr(int step) const = 0;
    
    void set_optimizer(Optimizer* opt) { optimizer_ = opt; }
    
protected:
    Optimizer* optimizer_ = nullptr;
};

class StepLR : public LRScheduler {
public:
    StepLR(float initial_lr, int step_size, float gamma = 0.1f);
    float get_lr(int step) const override;
    
private:
    float initial_lr_;
    int step_size_;
    float gamma_;
};

class MultiStepLR : public LRScheduler {
public:
    MultiStepLR(float initial_lr, const std::vector<int>& milestones, float gamma = 0.1f);
    float get_lr(int step) const override;
    
private:
    float initial_lr_;
    std::vector<int> milestones_;
    float gamma_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(float initial_lr, int T_max, float eta_min = 0.0f);
    float get_lr(int step) const override;
    
private:
    float initial_lr_;
    int T_max_;
    float eta_min_;
};

class ReduceLROnPlateau : public LRScheduler {
public:
    ReduceLROnPlateau(float initial_lr, float factor = 0.1f, int patience = 10,
                      float min_lr = 1e-6f);
    float get_lr(int step) const override;
    void step(float loss);
    
private:
    float initial_lr_;
    float factor_;
    int patience_;
    float min_lr_;
    float best_loss_;
    int num_bad_steps_;
};

// Factory functions
std::unique_ptr<Optimizer> create_sgd(float lr, float momentum = 0.0f, float wd = 0.0f);
std::unique_ptr<Optimizer> create_adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f,
                                        float eps = 1e-8f, float wd = 0.0f);
std::unique_ptr<Optimizer> create_lion(float lr, float beta = 0.9f, float wd = 0.0f);

} // namespace optim
} // namespace jin
