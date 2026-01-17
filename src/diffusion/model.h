#pragma once

#include "scheduler.h"
#include "sampler.h"
#include "../nn/unet.h"

#include <cstddef>
#include <vector>

namespace jin {
namespace diffusion {

class DiffusionModel {
public:
    DiffusionModel(const SchedulerConfig& scheduler_cfg = {},
                   size_t in_channels = 3,
                   size_t out_channels = 3,
                   size_t hidden_channels = 128,
                   std::vector<size_t> attention_at_res = {8},
                   Device device = Device::cpu())
        : scheduler_(scheduler_cfg),
          unet_(in_channels, out_channels, hidden_channels, std::move(attention_at_res)),
          sampler_(scheduler_, unet_, device) {}

    NoiseScheduler& scheduler() { return scheduler_; }
    const NoiseScheduler& scheduler() const { return scheduler_; }

    nn::UNet& unet() { return unet_; }
    const nn::UNet& unet() const { return unet_; }

    DDPMSampler& sampler() { return sampler_; }

    Tensor generate(size_t batch_size, size_t height, size_t width, size_t num_steps = 50, float guidance_scale = 1.0f) {
        return sampler_.sample_deterministic(batch_size, height, width, num_steps, guidance_scale);
    }

private:
    NoiseScheduler scheduler_;
    nn::UNet unet_;
    DDPMSampler sampler_;
};

} // namespace diffusion
} // namespace jin
