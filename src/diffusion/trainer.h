#pragma once

#include "scheduler.h"
#include "../nn/unet.h"
#include "../optim/optim.h"
#include "../core/device.h"

#include <cstddef>
#include <random>
#include <vector>

namespace jin {
namespace diffusion {

class DiffusionTrainer {
public:
    DiffusionTrainer(nn::UNet& model,
                    const NoiseScheduler& scheduler,
                    optim::Optimizer& optimizer,
                    Device device = Device::cpu());

    void set_seed(uint64_t seed);

    Tensor forward_diffusion_step(const Tensor& x0, size_t t, const Tensor& noise) const;
    Tensor compute_loss(const Tensor& x0, size_t t, const Tensor& noise);
    Tensor compute_loss(const Tensor& x0);

    float training_step(const Tensor& batch_x);

    // EMA
    void update_ema(float decay = 0.999f);
    const std::vector<Tensor>& ema_parameters() const { return ema_params_; }

private:
    nn::UNet& model_;
    const NoiseScheduler& scheduler_;
    optim::Optimizer& optimizer_;
    Device device_;

    std::mt19937 rng_;

    std::vector<Tensor> ema_params_;

    Tensor randn_like(const Shape& shape);
};

} // namespace diffusion
} // namespace jin
