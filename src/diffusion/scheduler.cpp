#include "scheduler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jin {
namespace diffusion {

namespace {

inline float clampf(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

} // namespace

NoiseScheduler::NoiseScheduler(const SchedulerConfig& config) : config_(config) {
    if (config_.timesteps == 0) {
        throw std::invalid_argument("Scheduler timesteps must be > 0");
    }
    if (config_.beta_start <= 0.0f || config_.beta_end <= 0.0f) {
        throw std::invalid_argument("beta_start and beta_end must be > 0");
    }
    if (config_.beta_start >= 1.0f || config_.beta_end >= 1.0f) {
        throw std::invalid_argument("beta_start and beta_end must be < 1");
    }

    build_schedule();
}

float NoiseScheduler::get_beta(size_t t) const {
    if (t > config_.timesteps) {
        throw std::out_of_range("timestep out of range");
    }
    return betas_[t];
}

float NoiseScheduler::get_alpha(size_t t) const {
    if (t > config_.timesteps) {
        throw std::out_of_range("timestep out of range");
    }
    return alpha_cumprod_[t];
}

float NoiseScheduler::get_posterior_variance(size_t t) const {
    if (t > config_.timesteps) {
        throw std::out_of_range("timestep out of range");
    }
    return posterior_variance_[t];
}

Tensor NoiseScheduler::scale_sample(const Tensor& x0, size_t t, const Tensor& eps) const {
    if (x0.shape() != eps.shape()) {
        throw std::invalid_argument("scale_sample: x0 and eps must have the same shape");
    }
    float alpha_bar = get_alpha(t);
    alpha_bar = clampf(alpha_bar, 0.0f, 1.0f);

    float s1 = std::sqrt(alpha_bar);
    float s2 = std::sqrt(std::max(0.0f, 1.0f - alpha_bar));

    Tensor out(x0.shape(), x0.dtype(), x0.device());
    const float* x_data = x0.data_ptr<float>();
    const float* e_data = eps.data_ptr<float>();
    float* o_data = out.data_ptr<float>();

    for (size_t i = 0; i < x0.size(); ++i) {
        o_data[i] = s1 * x_data[i] + s2 * e_data[i];
    }

    return out;
}

void NoiseScheduler::build_schedule() {
    betas_.assign(config_.timesteps + 1, 0.0f);
    alphas_.assign(config_.timesteps + 1, 1.0f);
    alpha_cumprod_.assign(config_.timesteps + 1, 1.0f);
    log_alpha_cumprod_.assign(config_.timesteps + 1, 0.0f);
    posterior_variance_.assign(config_.timesteps + 1, 0.0f);

    if (config_.schedule_type == "linear") {
        build_linear();
    } else if (config_.schedule_type == "cosine") {
        build_cosine();
    } else {
        throw std::invalid_argument("Unknown schedule_type: " + config_.schedule_type);
    }

    // Compute alpha cumprod using log-space for stability.
    log_alpha_cumprod_[0] = 0.0f;
    alpha_cumprod_[0] = 1.0f;
    for (size_t t = 1; t <= config_.timesteps; ++t) {
        float a = clampf(alphas_[t], 1e-12f, 1.0f);
        log_alpha_cumprod_[t] = log_alpha_cumprod_[t - 1] + std::log(a);
        alpha_cumprod_[t] = std::exp(log_alpha_cumprod_[t]);
        alpha_cumprod_[t] = clampf(alpha_cumprod_[t], 0.0f, 1.0f);
    }

    // Posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    posterior_variance_[0] = 0.0f;
    for (size_t t = 1; t <= config_.timesteps; ++t) {
        float beta_t = betas_[t];
        float a_bar_t = alpha_cumprod_[t];
        float a_bar_prev = alpha_cumprod_[t - 1];

        float denom = std::max(1e-20f, 1.0f - a_bar_t);
        float pv = beta_t * (1.0f - a_bar_prev) / denom;
        posterior_variance_[t] = std::max(0.0f, pv);
    }
}

void NoiseScheduler::build_linear() {
    size_t T = config_.timesteps;
    betas_[0] = 0.0f;
    alphas_[0] = 1.0f;

    for (size_t t = 1; t <= T; ++t) {
        float frac = static_cast<float>(t - 1) / static_cast<float>(std::max<size_t>(1, T - 1));
        float beta_t = config_.beta_start + frac * (config_.beta_end - config_.beta_start);
        beta_t = clampf(beta_t, 1e-8f, 0.999f);
        betas_[t] = beta_t;
        alphas_[t] = 1.0f - beta_t;
    }
}

void NoiseScheduler::build_cosine() {
    // Nichol & Dhariwal cosine schedule
    // alpha_bar(t) = cos(((t/T)+s)/(1+s) * pi/2)^2
    const float s = 0.008f;
    size_t T = config_.timesteps;

    auto alpha_bar = [T, s](size_t t) {
        float ft = static_cast<float>(t) / static_cast<float>(T);
        float x = (ft + s) / (1.0f + s);
        float c = std::cos(x * static_cast<float>(M_PI) * 0.5f);
        return c * c;
    };

    betas_[0] = 0.0f;
    alphas_[0] = 1.0f;

    float ab_prev = alpha_bar(0);
    for (size_t t = 1; t <= T; ++t) {
        float ab_t = alpha_bar(t);
        float beta_t = 1.0f - (ab_t / std::max(1e-20f, ab_prev));
        beta_t = clampf(beta_t, 1e-8f, 0.999f);
        betas_[t] = beta_t;
        alphas_[t] = 1.0f - beta_t;
        ab_prev = ab_t;
    }
}

} // namespace diffusion
} // namespace jin
