#pragma once

#include "../core/tensor.h"
#include <string>
#include <vector>

namespace jin {
namespace diffusion {

struct SchedulerConfig {
    size_t timesteps = 1000;
    std::string schedule_type = "linear"; // "linear" or "cosine"
    float beta_start = 1e-4f;
    float beta_end = 2e-2f;
};

class NoiseScheduler {
public:
    explicit NoiseScheduler(const SchedulerConfig& config = {});

    size_t timesteps() const { return config_.timesteps; }
    const SchedulerConfig& config() const { return config_; }

    float get_beta(size_t t) const;
    float get_alpha(size_t t) const; // alpha_cumprod(t)
    float get_posterior_variance(size_t t) const;

    Tensor scale_sample(const Tensor& x0, size_t t, const Tensor& eps) const;

private:
    SchedulerConfig config_;

    std::vector<float> betas_;              // [0..T]
    std::vector<float> alphas_;             // 1 - beta
    std::vector<float> alpha_cumprod_;      // prod alphas
    std::vector<float> log_alpha_cumprod_;  // log(alpha_cumprod)
    std::vector<float> posterior_variance_; // [0..T]

    void build_schedule();
    void build_linear();
    void build_cosine();
};

} // namespace diffusion
} // namespace jin
