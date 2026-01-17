#include "trainer.h"

#include "../autodiff/backward.h"
#include "../autodiff/grad_registry.h"
#include "../autodiff/ops.h"
#include "../autodiff/tape.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace jin {
namespace diffusion {

DiffusionTrainer::DiffusionTrainer(nn::UNet& model,
                                 const NoiseScheduler& scheduler,
                                 optim::Optimizer& optimizer,
                                 Device device)
    : model_(model), scheduler_(scheduler), optimizer_(optimizer), device_(device), rng_(0) {

    // Register all model parameters with optimizer and initialize EMA.
    auto params = model_.parameters();
    ema_params_.reserve(params.size());

    for (const auto& p : params) {
        optimizer_.register_parameter(p);

        Tensor ema(p.shape(), p.dtype(), p.device());
        std::memcpy(ema.data(), p.data(), p.size() * sizeof(float));
        ema_params_.push_back(ema);
    }
}

void DiffusionTrainer::set_seed(uint64_t seed) {
    rng_.seed(static_cast<uint32_t>(seed));
}

Tensor DiffusionTrainer::randn_like(const Shape& shape) {
    Tensor t(shape, DType::FP32, device_);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float* d = t.data_ptr<float>();
    for (size_t i = 0; i < t.size(); ++i) {
        d[i] = dist(rng_);
    }
    return t;
}

Tensor DiffusionTrainer::forward_diffusion_step(const Tensor& x0, size_t t, const Tensor& noise) const {
    return scheduler_.scale_sample(x0, t, noise);
}

Tensor DiffusionTrainer::compute_loss(const Tensor& x0, size_t t, const Tensor& noise) {
    Tensor xt = forward_diffusion_step(x0, t, noise);

    Tensor eps_pred = model_.forward(xt, t);

    // Record loss ops
    autodiff::GradientTape::instance().enable();

    Tensor diff = autodiff::subtract(eps_pred, noise);
    Tensor sq = autodiff::multiply(diff, diff);
    Tensor sum = autodiff::sum(sq, -1, false);

    Tensor inv_n({1}, DType::FP32, Device::cpu());
    inv_n.data_ptr<float>()[0] = 1.0f / static_cast<float>(sq.size());
    Tensor loss = autodiff::multiply(sum, inv_n);

    autodiff::GradientTape::instance().disable();

    return loss;
}

Tensor DiffusionTrainer::compute_loss(const Tensor& x0) {
    std::uniform_int_distribution<size_t> dist_t(1, scheduler_.timesteps());
    size_t t = dist_t(rng_);
    Tensor noise = randn_like(x0.shape());
    return compute_loss(x0, t, noise);
}

float DiffusionTrainer::training_step(const Tensor& batch_x) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();

    optimizer_.zero_grad();

    Tensor loss = compute_loss(batch_x);

    autodiff::backward(loss, nullptr);

    optimizer_.step();

    update_ema();

    return loss.item();
}

void DiffusionTrainer::update_ema(float decay) {
    auto params = model_.parameters();
    if (ema_params_.size() != params.size()) {
        return;
    }

    for (size_t i = 0; i < params.size(); ++i) {
        const float* p = params[i].data_ptr<float>();
        float* e = ema_params_[i].data_ptr<float>();

        for (size_t j = 0; j < params[i].size(); ++j) {
            e[j] = decay * e[j] + (1.0f - decay) * p[j];
        }
    }
}

} // namespace diffusion
} // namespace jin
