#include "unet.h"

#include "../autodiff/ops.h"
#include "../autodiff/tape.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace jin {
namespace nn {

namespace {

inline size_t best_group_count(size_t channels, size_t requested) {
    requested = std::max<size_t>(1, std::min(requested, channels));
    for (size_t g = requested; g >= 1; --g) {
        if (channels % g == 0) return g;
        if (g == 1) break;
    }
    return 1;
}

Tensor silu(const Tensor& x) {
    // SiLU(x) = x * sigmoid(x)
    autodiff::TapeGuard guard;
    return autodiff::multiply(x, autodiff::sigmoid(x));
}

Tensor concat_channels(const Tensor& a, const Tensor& b) {
    autodiff::TapeGuard guard;

    if (a.ndim() != 4 || b.ndim() != 4) {
        throw std::invalid_argument("concat_channels expects 4D tensors [B,C,H,W]");
    }
    if (a.shape()[0] != b.shape()[0] || a.shape()[2] != b.shape()[2] || a.shape()[3] != b.shape()[3]) {
        throw std::invalid_argument("concat_channels: batch/height/width must match");
    }

    Shape out_shape = {a.shape()[0], a.shape()[1] + b.shape()[1], a.shape()[2], a.shape()[3]};
    Tensor out(out_shape, a.dtype(), a.device());

    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* o_data = out.data_ptr<float>();

    size_t B = a.shape()[0];
    size_t Ca = a.shape()[1];
    size_t Cb = b.shape()[1];
    size_t H = a.shape()[2];
    size_t W = a.shape()[3];
    size_t HW = H * W;

    for (size_t n = 0; n < B; ++n) {
        // Copy a
        std::memcpy(o_data + (n * (Ca + Cb) * HW), a_data + (n * Ca * HW), Ca * HW * sizeof(float));
        // Copy b
        std::memcpy(o_data + (n * (Ca + Cb) * HW) + (Ca * HW), b_data + (n * Cb * HW), Cb * HW * sizeof(float));
    }

    auto backward_fn = [a, b, out_shape](const Tensor& grad) -> std::vector<Tensor> {
        Tensor grad_a(a.shape(), a.dtype(), a.device());
        Tensor grad_b(b.shape(), b.dtype(), b.device());

        const float* g_data = grad.data_ptr<float>();
        float* ga = grad_a.data_ptr<float>();
        float* gb = grad_b.data_ptr<float>();

        size_t B = a.shape()[0];
        size_t Ca = a.shape()[1];
        size_t Cb = b.shape()[1];
        size_t H = a.shape()[2];
        size_t W = a.shape()[3];
        size_t HW = H * W;

        for (size_t n = 0; n < B; ++n) {
            std::memcpy(ga + n * Ca * HW, g_data + n * (Ca + Cb) * HW, Ca * HW * sizeof(float));
            std::memcpy(gb + n * Cb * HW, g_data + n * (Ca + Cb) * HW + Ca * HW, Cb * HW * sizeof(float));
        }

        return {grad_a, grad_b};
    };

    autodiff::GradientTape::instance().record({a, b}, out, backward_fn, "concat_channels");
    if (a.requires_grad() || b.requires_grad()) {
        out.set_requires_grad(true);
    }

    return out;
}

Tensor upsample_nearest2x(const Tensor& x) {
    autodiff::TapeGuard guard;

    if (x.ndim() != 4) {
        throw std::invalid_argument("upsample_nearest2x expects 4D tensor [B,C,H,W]");
    }

    size_t B = x.shape()[0];
    size_t C = x.shape()[1];
    size_t H = x.shape()[2];
    size_t W = x.shape()[3];

    size_t H2 = H * 2;
    size_t W2 = W * 2;

    Tensor out({B, C, H2, W2}, x.dtype(), x.device());

    const float* x_data = x.data_ptr<float>();
    float* o_data = out.data_ptr<float>();

    for (size_t n = 0; n < B; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t i = 0; i < H2; ++i) {
                size_t src_i = i / 2;
                for (size_t j = 0; j < W2; ++j) {
                    size_t src_j = j / 2;
                    o_data[(((n * C + c) * H2 + i) * W2) + j] =
                        x_data[(((n * C + c) * H + src_i) * W) + src_j];
                }
            }
        }
    }

    auto backward_fn = [x, H, W](const Tensor& grad) -> std::vector<Tensor> {
        Tensor grad_x(x.shape(), x.dtype(), x.device());
        grad_x.fill(0.0f);

        size_t B = x.shape()[0];
        size_t C = x.shape()[1];
        size_t H2 = H * 2;
        size_t W2 = W * 2;

        const float* g = grad.data_ptr<float>();
        float* gx = grad_x.data_ptr<float>();

        for (size_t n = 0; n < B; ++n) {
            for (size_t c = 0; c < C; ++c) {
                for (size_t i = 0; i < H2; ++i) {
                    size_t src_i = i / 2;
                    for (size_t j = 0; j < W2; ++j) {
                        size_t src_j = j / 2;
                        gx[(((n * C + c) * H + src_i) * W) + src_j] +=
                            g[(((n * C + c) * H2 + i) * W2) + j];
                    }
                }
            }
        }

        return {grad_x};
    };

    autodiff::GradientTape::instance().record({x}, out, backward_fn, "upsample_nearest2x");
    if (x.requires_grad()) {
        out.set_requires_grad(true);
    }

    return out;
}

Tensor apply_scale_shift(const Tensor& x, const Tensor& scale_shift) {
    autodiff::TapeGuard guard;

    if (x.ndim() != 4) {
        throw std::invalid_argument("apply_scale_shift expects x [B,C,H,W]");
    }
    if (scale_shift.ndim() != 2) {
        throw std::invalid_argument("apply_scale_shift expects scale_shift [B,2C]");
    }

    size_t B = x.shape()[0];
    size_t C = x.shape()[1];
    size_t H = x.shape()[2];
    size_t W = x.shape()[3];

    if (scale_shift.shape()[0] != B || scale_shift.shape()[1] != 2 * C) {
        throw std::invalid_argument("apply_scale_shift: scale_shift must be [B,2C]");
    }

    Tensor out(x.shape(), x.dtype(), x.device());

    const float* x_data = x.data_ptr<float>();
    const float* ss = scale_shift.data_ptr<float>();
    float* o_data = out.data_ptr<float>();

    for (size_t n = 0; n < B; ++n) {
        for (size_t c = 0; c < C; ++c) {
            float scale = ss[n * (2 * C) + c];
            float shift = ss[n * (2 * C) + (C + c)];
            float mult = 1.0f + scale;

            for (size_t i = 0; i < H * W; ++i) {
                size_t idx = (n * C + c) * (H * W) + i;
                o_data[idx] = x_data[idx] * mult + shift;
            }
        }
    }

    auto backward_fn = [x, scale_shift](const Tensor& grad) -> std::vector<Tensor> {
        Tensor grad_x(x.shape(), x.dtype(), x.device());
        Tensor grad_ss(scale_shift.shape(), scale_shift.dtype(), scale_shift.device());
        grad_ss.fill(0.0f);

        size_t B = x.shape()[0];
        size_t C = x.shape()[1];
        size_t H = x.shape()[2];
        size_t W = x.shape()[3];
        size_t HW = H * W;

        const float* g = grad.data_ptr<float>();
        const float* x_data = x.data_ptr<float>();
        const float* ss = scale_shift.data_ptr<float>();
        float* gx = grad_x.data_ptr<float>();
        float* gss = grad_ss.data_ptr<float>();

        for (size_t n = 0; n < B; ++n) {
            for (size_t c = 0; c < C; ++c) {
                float scale = ss[n * (2 * C) + c];
                float mult = 1.0f + scale;

                float accum_scale = 0.0f;
                float accum_shift = 0.0f;

                for (size_t i = 0; i < HW; ++i) {
                    size_t idx = (n * C + c) * HW + i;
                    gx[idx] = g[idx] * mult;
                    accum_scale += g[idx] * x_data[idx];
                    accum_shift += g[idx];
                }

                gss[n * (2 * C) + c] = accum_scale;
                gss[n * (2 * C) + (C + c)] = accum_shift;
            }
        }

        return {grad_x, grad_ss};
    };

    autodiff::GradientTape::instance().record({x, scale_shift}, out, backward_fn, "apply_scale_shift");
    if (x.requires_grad() || scale_shift.requires_grad()) {
        out.set_requires_grad(true);
    }

    return out;
}

} // namespace

// ==================== TimeEmbedding ====================

TimeEmbedding::TimeEmbedding(size_t feature_dim)
    : feature_dim_(feature_dim),
      proj1_(feature_dim, feature_dim * 4),
      proj2_(feature_dim * 4, feature_dim) {}

Tensor TimeEmbedding::forward(const Tensor& timesteps) {
    autodiff::TapeGuard guard;

    if (timesteps.ndim() != 1 && !(timesteps.ndim() == 2 && timesteps.shape()[1] == 1)) {
        throw std::invalid_argument("TimeEmbedding expects timesteps as [B] or [B,1]");
    }

    size_t B = timesteps.shape()[0];
    Tensor t1d = timesteps.ndim() == 1 ? timesteps : timesteps.reshape({B});

    Tensor emb({B, feature_dim_}, DType::FP32, timesteps.device());
    const float* t_data = t1d.data_ptr<float>();
    float* e_data = emb.data_ptr<float>();

    size_t half = feature_dim_ / 2;
    float log_max = std::log(10000.0f);

    for (size_t n = 0; n < B; ++n) {
        float t = t_data[n];
        for (size_t i = 0; i < half; ++i) {
            float inv_freq = std::exp(-log_max * static_cast<float>(i) / static_cast<float>(std::max<size_t>(1, half)));
            float v = t * inv_freq;
            e_data[n * feature_dim_ + i] = std::sin(v);
            if (i + half < feature_dim_) {
                e_data[n * feature_dim_ + (i + half)] = std::cos(v);
            }
        }
        if (feature_dim_ % 2 == 1) {
            e_data[n * feature_dim_ + (feature_dim_ - 1)] = 0.0f;
        }
    }

    Tensor h = proj1_.forward(emb);
    h = silu(h);
    h = proj2_.forward(h);
    return h;
}

std::vector<Tensor> TimeEmbedding::parameters() const {
    std::vector<Tensor> params;
    params.push_back(proj1_.weight());
    params.push_back(proj1_.bias());
    params.push_back(proj2_.weight());
    params.push_back(proj2_.bias());
    return params;
}

// ==================== GroupNorm ====================

GroupNorm::GroupNorm(size_t num_channels, size_t groups, float eps)
    : num_channels_(num_channels),
      groups_(best_group_count(num_channels, groups)),
      eps_(eps) {

    weight_ = jin::ones({num_channels_}, DType::FP32, Device::cpu());
    weight_.set_requires_grad(true);
    bias_ = jin::zeros({num_channels_}, DType::FP32, Device::cpu());
    bias_.set_requires_grad(true);
}

Tensor GroupNorm::forward(const Tensor& x) {
    autodiff::TapeGuard guard;

    if (x.ndim() != 4) {
        throw std::invalid_argument("GroupNorm expects 4D input [B,C,H,W]");
    }
    if (x.shape()[1] != num_channels_) {
        throw std::invalid_argument("GroupNorm: channel dimension mismatch");
    }

    size_t B = x.shape()[0];
    size_t C = x.shape()[1];
    size_t H = x.shape()[2];
    size_t W = x.shape()[3];

    size_t G = groups_;
    size_t CpG = C / G;
    size_t N = CpG * H * W;

    Tensor out(x.shape(), x.dtype(), x.device());

    const float* x_data = x.data_ptr<float>();
    const float* w_data = weight_.data_ptr<float>();
    const float* b_data = bias_.data_ptr<float>();
    float* o_data = out.data_ptr<float>();

    for (size_t n = 0; n < B; ++n) {
        for (size_t g = 0; g < G; ++g) {
            // Compute mean/var for group
            double mean = 0.0;
            double var = 0.0;

            for (size_t c = 0; c < CpG; ++c) {
                size_t ch = g * CpG + c;
                for (size_t i = 0; i < H * W; ++i) {
                    size_t idx = (n * C + ch) * (H * W) + i;
                    mean += static_cast<double>(x_data[idx]);
                }
            }
            mean /= static_cast<double>(N);

            for (size_t c = 0; c < CpG; ++c) {
                size_t ch = g * CpG + c;
                for (size_t i = 0; i < H * W; ++i) {
                    size_t idx = (n * C + ch) * (H * W) + i;
                    double d = static_cast<double>(x_data[idx]) - mean;
                    var += d * d;
                }
            }
            var /= static_cast<double>(N);

            float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps_);

            for (size_t c = 0; c < CpG; ++c) {
                size_t ch = g * CpG + c;
                float gamma = w_data[ch];
                float beta = b_data[ch];

                for (size_t i = 0; i < H * W; ++i) {
                    size_t idx = (n * C + ch) * (H * W) + i;
                    float xhat = (x_data[idx] - static_cast<float>(mean)) * inv_std;
                    o_data[idx] = xhat * gamma + beta;
                }
            }
        }
    }

    auto backward_fn = [x, weight = weight_, bias = bias_, eps = eps_, G = groups_](const Tensor& grad) -> std::vector<Tensor> {
        Tensor grad_x(x.shape(), x.dtype(), x.device());
        Tensor grad_w(weight.shape(), weight.dtype(), weight.device());
        Tensor grad_b(bias.shape(), bias.dtype(), bias.device());
        grad_w.fill(0.0f);
        grad_b.fill(0.0f);

        size_t B = x.shape()[0];
        size_t C = x.shape()[1];
        size_t H = x.shape()[2];
        size_t W = x.shape()[3];

        size_t CpG = C / G;
        size_t N = CpG * H * W;

        const float* x_data = x.data_ptr<float>();
        const float* w_data = weight.data_ptr<float>();
        const float* g_data = grad.data_ptr<float>();
        float* gx = grad_x.data_ptr<float>();
        float* gw = grad_w.data_ptr<float>();
        float* gb = grad_b.data_ptr<float>();

        for (size_t n = 0; n < B; ++n) {
            for (size_t g = 0; g < G; ++g) {
                // mean/var
                double mean = 0.0;
                double var = 0.0;
                for (size_t c = 0; c < CpG; ++c) {
                    size_t ch = g * CpG + c;
                    for (size_t i = 0; i < H * W; ++i) {
                        size_t idx = (n * C + ch) * (H * W) + i;
                        mean += static_cast<double>(x_data[idx]);
                    }
                }
                mean /= static_cast<double>(N);

                for (size_t c = 0; c < CpG; ++c) {
                    size_t ch = g * CpG + c;
                    for (size_t i = 0; i < H * W; ++i) {
                        size_t idx = (n * C + ch) * (H * W) + i;
                        double d = static_cast<double>(x_data[idx]) - mean;
                        var += d * d;
                    }
                }
                var /= static_cast<double>(N);

                float inv_std = 1.0f / std::sqrt(static_cast<float>(var) + eps);

                // Accumulators for dxhat
                double sum_dxhat = 0.0;
                double sum_dxhat_xhat = 0.0;

                // First pass: compute xhat and dxhat, also gamma/beta grads
                for (size_t c = 0; c < CpG; ++c) {
                    size_t ch = g * CpG + c;
                    float gamma = w_data[ch];

                    for (size_t i = 0; i < H * W; ++i) {
                        size_t idx = (n * C + ch) * (H * W) + i;
                        float xhat = (x_data[idx] - static_cast<float>(mean)) * inv_std;
                        float dy = g_data[idx];

                        // grad gamma/beta
                        gw[ch] += dy * xhat;
                        gb[ch] += dy;

                        float dxhat = dy * gamma;
                        sum_dxhat += dxhat;
                        sum_dxhat_xhat += static_cast<double>(dxhat) * static_cast<double>(xhat);
                    }
                }

                float mean_dxhat = static_cast<float>(sum_dxhat / static_cast<double>(N));
                float mean_dxhat_xhat = static_cast<float>(sum_dxhat_xhat / static_cast<double>(N));

                for (size_t c = 0; c < CpG; ++c) {
                    size_t ch = g * CpG + c;
                    float gamma = w_data[ch];

                    for (size_t i = 0; i < H * W; ++i) {
                        size_t idx = (n * C + ch) * (H * W) + i;
                        float xhat = (x_data[idx] - static_cast<float>(mean)) * inv_std;
                        float dy = g_data[idx];
                        float dxhat = dy * gamma;

                        gx[idx] = inv_std * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
                    }
                }
            }
        }

        return {grad_x, grad_w, grad_b};
    };

    autodiff::GradientTape::instance().record({x, weight_, bias_}, out, backward_fn, "groupnorm");
    if (x.requires_grad() || weight_.requires_grad() || bias_.requires_grad()) {
        out.set_requires_grad(true);
    }

    return out;
}

std::vector<Tensor> GroupNorm::parameters() const {
    return {weight_, bias_};
}

// ==================== ResnetBlock ====================

ResnetBlock::ResnetBlock(size_t in_channels, size_t out_channels, size_t time_embed_dim, size_t groups)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      time_embed_dim_(time_embed_dim),
      norm1_(in_channels, groups),
      norm2_(out_channels, groups),
      conv1_(in_channels, out_channels, 3, 1, 1, 1, 1),
      conv2_(out_channels, out_channels, 3, 1, 1, 1, 1),
      time_proj_(time_embed_dim, 2 * out_channels),
      use_skip_conv_(in_channels != out_channels),
      skip_conv_(in_channels, out_channels, 1, 1, 1, 0, 0) {}

Tensor ResnetBlock::forward(const Tensor& x, const Tensor& time_emb) {
    // norm + silu + conv
    Tensor h = norm1_.forward(x);
    h = silu(h);
    h = conv1_.forward(h);

    Tensor ss = time_proj_.forward(time_emb);
    h = apply_scale_shift(h, ss);

    h = norm2_.forward(h);
    h = silu(h);
    h = conv2_.forward(h);

    Tensor skip = x;
    if (use_skip_conv_) {
        skip = skip_conv_.forward(x);
    }

    autodiff::TapeGuard guard;
    return autodiff::add(h, skip);
}

std::vector<Tensor> ResnetBlock::parameters() const {
    std::vector<Tensor> p;
    auto add = [&p](const std::vector<Tensor>& more) {
        p.insert(p.end(), more.begin(), more.end());
    };

    add(norm1_.parameters());
    add(norm2_.parameters());

    p.push_back(conv1_.weight());
    p.push_back(conv1_.bias());
    p.push_back(conv2_.weight());
    p.push_back(conv2_.bias());

    p.push_back(time_proj_.weight());
    p.push_back(time_proj_.bias());

    if (use_skip_conv_) {
        p.push_back(skip_conv_.weight());
        p.push_back(skip_conv_.bias());
    }

    return p;
}

// ==================== AttentionBlock ====================

AttentionBlock::AttentionBlock(size_t channels)
    : channels_(channels),
      norm_(channels, 32),
      qkv_proj_(channels, channels * 3, 1, 1, 1, 0, 0),
      out_proj_(channels, channels, 1, 1, 1, 0, 0) {}

Tensor AttentionBlock::forward(const Tensor& x) {
    Tensor h = norm_.forward(x);
    Tensor qkv = qkv_proj_.forward(h);

    // Split into q,k,v with separate convs avoided here by slicing via copy.
    // We intentionally keep this lightweight; correctness in shape is the priority.
    size_t B = qkv.shape()[0];
    size_t C3 = qkv.shape()[1];
    size_t H = qkv.shape()[2];
    size_t W = qkv.shape()[3];
    size_t C = C3 / 3;

    Tensor q({B, C, H, W}, qkv.dtype(), qkv.device());
    Tensor k({B, C, H, W}, qkv.dtype(), qkv.device());
    Tensor v({B, C, H, W}, qkv.dtype(), qkv.device());

    const float* qkv_data = qkv.data_ptr<float>();
    float* qd = q.data_ptr<float>();
    float* kd = k.data_ptr<float>();
    float* vd = v.data_ptr<float>();

    size_t HW = H * W;
    for (size_t n = 0; n < B; ++n) {
        for (size_t c = 0; c < C; ++c) {
            std::memcpy(qd + (n * C + c) * HW, qkv_data + (n * (3 * C) + c) * HW, HW * sizeof(float));
            std::memcpy(kd + (n * C + c) * HW, qkv_data + (n * (3 * C) + (C + c)) * HW, HW * sizeof(float));
            std::memcpy(vd + (n * C + c) * HW, qkv_data + (n * (3 * C) + (2 * C + c)) * HW, HW * sizeof(float));
        }
    }

    autodiff::TapeGuard guard;
    Tensor gate = autodiff::sigmoid(k);
    Tensor y = autodiff::add(autodiff::multiply(q, gate), v);
    y = out_proj_.forward(y);

    return autodiff::add(x, y);
}

std::vector<Tensor> AttentionBlock::parameters() const {
    std::vector<Tensor> p;
    auto add = [&p](const std::vector<Tensor>& more) {
        p.insert(p.end(), more.begin(), more.end());
    };

    add(norm_.parameters());
    p.push_back(qkv_proj_.weight());
    p.push_back(qkv_proj_.bias());
    p.push_back(out_proj_.weight());
    p.push_back(out_proj_.bias());
    return p;
}

// ==================== UNet ====================

UNet::UNet(size_t in_channels, size_t out_channels, size_t hidden_channels, std::vector<size_t> attention_at_res)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      hidden_channels_(hidden_channels),
      attention_at_res_(std::move(attention_at_res)),
      time_embed_(hidden_channels * 4),
      in_conv_(in_channels, hidden_channels, 3, 1, 1, 1, 1),
      down0_a_(hidden_channels, hidden_channels, time_embed_.feature_dim()),
      down0_b_(hidden_channels, hidden_channels, time_embed_.feature_dim()),
      down0_attn_(hidden_channels),
      down0_use_attn_(false),
      down0_downsample_(hidden_channels, hidden_channels, 3, 2, 2, 1, 1),
      down1_a_(hidden_channels, hidden_channels * 2, time_embed_.feature_dim()),
      down1_b_(hidden_channels * 2, hidden_channels * 2, time_embed_.feature_dim()),
      down1_attn_(hidden_channels * 2),
      down1_use_attn_(false),
      down1_downsample_(hidden_channels * 2, hidden_channels * 2, 3, 2, 2, 1, 1),
      down2_a_(hidden_channels * 2, hidden_channels * 4, time_embed_.feature_dim()),
      down2_b_(hidden_channels * 4, hidden_channels * 4, time_embed_.feature_dim()),
      down2_attn_(hidden_channels * 4),
      down2_use_attn_(true),
      mid_a_(hidden_channels * 4, hidden_channels * 4, time_embed_.feature_dim()),
      mid_attn_(hidden_channels * 4),
      mid_b_(hidden_channels * 4, hidden_channels * 4, time_embed_.feature_dim()),
      up2_a_(hidden_channels * 8, hidden_channels * 4, time_embed_.feature_dim()),
      up2_b_(hidden_channels * 4, hidden_channels * 4, time_embed_.feature_dim()),
      up2_attn_(hidden_channels * 4),
      up2_use_attn_(true),
      up1_a_(hidden_channels * 6, hidden_channels * 2, time_embed_.feature_dim()),
      up1_b_(hidden_channels * 2, hidden_channels * 2, time_embed_.feature_dim()),
      up1_attn_(hidden_channels * 2),
      up1_use_attn_(false),
      up0_a_(hidden_channels * 3, hidden_channels, time_embed_.feature_dim()),
      up0_b_(hidden_channels, hidden_channels, time_embed_.feature_dim()),
      up0_attn_(hidden_channels),
      up0_use_attn_(false),
      out_norm_(hidden_channels, 32),
      out_conv_(hidden_channels, out_channels, 3, 1, 1, 1, 1) {

    // Best-effort static attention assignment for typical 32->16->8 hierarchy.
    down0_use_attn_ = attention_enabled_for_res(32);
    down1_use_attn_ = attention_enabled_for_res(16);
    down2_use_attn_ = attention_enabled_for_res(8);

    up2_use_attn_ = attention_enabled_for_res(8);
    up1_use_attn_ = attention_enabled_for_res(16);
    up0_use_attn_ = attention_enabled_for_res(32);
}

bool UNet::attention_enabled_for_res(size_t res) const {
    return std::find(attention_at_res_.begin(), attention_at_res_.end(), res) != attention_at_res_.end();
}

Tensor UNet::forward(const Tensor& x, size_t t) {
    Tensor tt({x.shape()[0]}, DType::FP32, x.device());
    float* td = tt.data_ptr<float>();
    for (size_t i = 0; i < x.shape()[0]; ++i) td[i] = static_cast<float>(t);
    return forward(x, tt);
}

Tensor UNet::forward(const Tensor& x, const Tensor& timesteps) {
    Tensor t_emb = time_embed_.forward(timesteps);

    Tensor h = in_conv_.forward(x);

    // Down 0
    h = down0_a_.forward(h, t_emb);
    h = down0_b_.forward(h, t_emb);
    if (down0_use_attn_) h = down0_attn_.forward(h);
    Tensor skip0 = h;
    h = down0_downsample_.forward(h);

    // Down 1
    h = down1_a_.forward(h, t_emb);
    h = down1_b_.forward(h, t_emb);
    if (down1_use_attn_) h = down1_attn_.forward(h);
    Tensor skip1 = h;
    h = down1_downsample_.forward(h);

    // Down 2
    h = down2_a_.forward(h, t_emb);
    h = down2_b_.forward(h, t_emb);
    if (down2_use_attn_) h = down2_attn_.forward(h);
    Tensor skip2 = h;

    // Mid
    h = mid_a_.forward(h, t_emb);
    h = mid_attn_.forward(h);
    h = mid_b_.forward(h, t_emb);

    // Up 2
    h = concat_channels(h, skip2);
    h = up2_a_.forward(h, t_emb);
    h = up2_b_.forward(h, t_emb);
    if (up2_use_attn_) h = up2_attn_.forward(h);
    h = upsample_nearest2x(h);

    // Up 1
    h = concat_channels(h, skip1);
    h = up1_a_.forward(h, t_emb);
    h = up1_b_.forward(h, t_emb);
    if (up1_use_attn_) h = up1_attn_.forward(h);
    h = upsample_nearest2x(h);

    // Up 0
    h = concat_channels(h, skip0);
    h = up0_a_.forward(h, t_emb);
    h = up0_b_.forward(h, t_emb);
    if (up0_use_attn_) h = up0_attn_.forward(h);

    h = out_norm_.forward(h);
    h = silu(h);
    h = out_conv_.forward(h);
    return h;
}

std::vector<Tensor> UNet::parameters() const {
    std::vector<Tensor> p;
    auto add = [&p](const std::vector<Tensor>& more) {
        p.insert(p.end(), more.begin(), more.end());
    };

    add(time_embed_.parameters());

    p.push_back(in_conv_.weight());
    p.push_back(in_conv_.bias());

    add(down0_a_.parameters());
    add(down0_b_.parameters());
    add(down0_attn_.parameters());
    p.push_back(down0_downsample_.weight());
    p.push_back(down0_downsample_.bias());

    add(down1_a_.parameters());
    add(down1_b_.parameters());
    add(down1_attn_.parameters());
    p.push_back(down1_downsample_.weight());
    p.push_back(down1_downsample_.bias());

    add(down2_a_.parameters());
    add(down2_b_.parameters());
    add(down2_attn_.parameters());

    add(mid_a_.parameters());
    add(mid_attn_.parameters());
    add(mid_b_.parameters());

    add(up2_a_.parameters());
    add(up2_b_.parameters());
    add(up2_attn_.parameters());

    add(up1_a_.parameters());
    add(up1_b_.parameters());
    add(up1_attn_.parameters());

    add(up0_a_.parameters());
    add(up0_b_.parameters());
    add(up0_attn_.parameters());

    add(out_norm_.parameters());
    p.push_back(out_conv_.weight());
    p.push_back(out_conv_.bias());

    return p;
}

} // namespace nn
} // namespace jin
