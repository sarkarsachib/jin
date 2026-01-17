#pragma once

#include "scheduler.h"
#include "../nn/unet.h"
#include "../core/device.h"

#include <cstddef>
#include <random>

namespace jin {
namespace diffusion {

class DDPMSampler {
public:
    DDPMSampler(const NoiseScheduler& scheduler, nn::UNet& model, Device device = Device::cpu());

    void set_seed(uint64_t seed);

    Tensor sample(size_t batch_size, size_t height, size_t width, size_t num_steps,
                  float guidance_scale = 1.0f);

    Tensor sample_deterministic(size_t batch_size, size_t height, size_t width, size_t num_steps,
                                float guidance_scale = 1.0f);

private:
    const NoiseScheduler& scheduler_;
    nn::UNet& model_;
    Device device_;

    std::mt19937 rng_;

    Tensor randn_like(const Shape& shape);
};

} // namespace diffusion
} // namespace jin
