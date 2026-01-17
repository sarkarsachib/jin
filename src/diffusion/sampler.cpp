#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace jin {
namespace diffusion {

DDPMSampler::DDPMSampler(const NoiseScheduler& scheduler, nn::UNet& model, Device device)
    : scheduler_(scheduler), model_(model), device_(device), rng_(0) {}

void DDPMSampler::set_seed(uint64_t seed) {
    rng_.seed(static_cast<uint32_t>(seed));
}

Tensor DDPMSampler::randn_like(const Shape& shape) {
    Tensor t(shape, DType::FP32, device_);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float* d = t.data_ptr<float>();
    for (size_t i = 0; i < t.size(); ++i) {
        d[i] = dist(rng_);
    }
    return t;
}

static std::vector<size_t> make_timesteps(size_t T, size_t num_steps) {
    num_steps = std::max<size_t>(1, std::min(num_steps, T));

    std::vector<size_t> steps;
    steps.reserve(num_steps);

    if (num_steps == T) {
        for (size_t t = T; t >= 1; --t) {
            steps.push_back(t);
            if (t == 1) break;
        }
        return steps;
    }

    double stride = static_cast<double>(T) / static_cast<double>(num_steps);
    for (size_t i = 0; i < num_steps; ++i) {
        size_t t = static_cast<size_t>(std::round(static_cast<double>(T) - i * stride));
        t = std::max<size_t>(1, std::min(t, T));
        if (!steps.empty() && steps.back() == t) {
            if (t > 1) t -= 1;
        }
        steps.push_back(t);
    }

    if (steps.back() != 1) steps.back() = 1;

    // Ensure strictly descending
    for (size_t i = 1; i < steps.size(); ++i) {
        if (steps[i] >= steps[i - 1]) {
            steps[i] = (steps[i - 1] > 1) ? (steps[i - 1] - 1) : 1;
        }
    }

    return steps;
}

Tensor DDPMSampler::sample(size_t batch_size, size_t height, size_t width, size_t num_steps, float guidance_scale) {
    size_t C = model_.in_channels();
    Tensor x = randn_like({batch_size, C, height, width});

    size_t T = scheduler_.timesteps();
    std::vector<size_t> steps = make_timesteps(T, num_steps);

    for (size_t idx = 0; idx < steps.size(); ++idx) {
        size_t t = steps[idx];

        Tensor eps_pred = model_.forward(x, t);
        if (guidance_scale != 1.0f) {
            // Classifier-free guidance with implicit unconditional=0
            float* ep = eps_pred.data_ptr<float>();
            for (size_t i = 0; i < eps_pred.size(); ++i) {
                ep[i] *= guidance_scale;
            }
        }

        float beta_t = scheduler_.get_beta(t);
        float alpha_t = 1.0f - beta_t;
        float alpha_bar_t = scheduler_.get_alpha(t);

        float sqrt_one_minus_ab = std::sqrt(std::max(1e-20f, 1.0f - alpha_bar_t));
        float sqrt_alpha_t = std::sqrt(std::max(1e-20f, alpha_t));

        // mu_t = (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) / sqrt(alpha_t)
        float coef = beta_t / sqrt_one_minus_ab;

        Tensor mu(x.shape(), x.dtype(), x.device());
        const float* x_data = x.data_ptr<float>();
        const float* e_data = eps_pred.data_ptr<float>();
        float* m_data = mu.data_ptr<float>();

        for (size_t i = 0; i < x.size(); ++i) {
            m_data[i] = (x_data[i] - coef * e_data[i]) / sqrt_alpha_t;
        }

        if (t > 1) {
            float var = scheduler_.get_posterior_variance(t);
            float std = std::sqrt(std::max(0.0f, var));
            Tensor z = randn_like(x.shape());
            float* z_data = z.data_ptr<float>();

            float* out_data = x.data_ptr<float>();
            for (size_t i = 0; i < x.size(); ++i) {
                out_data[i] = m_data[i] + std * z_data[i];
            }
        } else {
            x = mu;
        }
    }

    return x;
}

Tensor DDPMSampler::sample_deterministic(size_t batch_size, size_t height, size_t width, size_t num_steps, float guidance_scale) {
    size_t C = model_.in_channels();
    Tensor x = randn_like({batch_size, C, height, width});

    size_t T = scheduler_.timesteps();
    std::vector<size_t> steps = make_timesteps(T, num_steps);

    for (size_t idx = 0; idx < steps.size(); ++idx) {
        size_t t = steps[idx];
        size_t t_prev = (idx + 1 < steps.size()) ? steps[idx + 1] : 0;

        Tensor eps_pred = model_.forward(x, t);
        if (guidance_scale != 1.0f) {
            float* ep = eps_pred.data_ptr<float>();
            for (size_t i = 0; i < eps_pred.size(); ++i) {
                ep[i] *= guidance_scale;
            }
        }

        float alpha_bar_t = scheduler_.get_alpha(t);
        float alpha_bar_prev = (t_prev == 0) ? 1.0f : scheduler_.get_alpha(t_prev);

        float sqrt_ab_t = std::sqrt(std::max(1e-20f, alpha_bar_t));
        float sqrt_one_minus_ab_t = std::sqrt(std::max(1e-20f, 1.0f - alpha_bar_t));

        float sqrt_ab_prev = std::sqrt(std::max(1e-20f, alpha_bar_prev));
        float sqrt_one_minus_ab_prev = std::sqrt(std::max(1e-20f, 1.0f - alpha_bar_prev));

        Tensor x0_pred(x.shape(), x.dtype(), x.device());
        const float* x_data = x.data_ptr<float>();
        const float* e_data = eps_pred.data_ptr<float>();
        float* x0_data = x0_pred.data_ptr<float>();

        for (size_t i = 0; i < x.size(); ++i) {
            x0_data[i] = (x_data[i] - sqrt_one_minus_ab_t * e_data[i]) / sqrt_ab_t;
        }

        Tensor x_prev(x.shape(), x.dtype(), x.device());
        float* xp = x_prev.data_ptr<float>();
        for (size_t i = 0; i < x.size(); ++i) {
            xp[i] = sqrt_ab_prev * x0_data[i] + sqrt_one_minus_ab_prev * e_data[i];
        }

        x = x_prev;
    }

    return x;
}

} // namespace diffusion
} // namespace jin
