#pragma once

#include "../core/tensor.h"
#include "conv.h"
#include "linear.h"

#include <cstddef>
#include <vector>

namespace jin {
namespace nn {

class TimeEmbedding {
public:
    explicit TimeEmbedding(size_t feature_dim);

    Tensor forward(const Tensor& timesteps); // [B] or [1] -> [B, feature_dim]

    size_t feature_dim() const { return feature_dim_; }

    std::vector<Tensor> parameters() const;

private:
    size_t feature_dim_;
    Linear proj1_;
    Linear proj2_;
};

class GroupNorm {
public:
    GroupNorm(size_t num_channels, size_t groups = 32, float eps = 1e-5f);

    Tensor forward(const Tensor& x);

    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }

    std::vector<Tensor> parameters() const;

private:
    size_t num_channels_;
    size_t groups_;
    float eps_;

    Tensor weight_;
    Tensor bias_;
};

class ResnetBlock {
public:
    ResnetBlock(size_t in_channels, size_t out_channels, size_t time_embed_dim,
                size_t groups = 32);

    Tensor forward(const Tensor& x, const Tensor& time_emb); // time_emb: [B, time_embed_dim]

    size_t in_channels() const { return in_channels_; }
    size_t out_channels() const { return out_channels_; }

    std::vector<Tensor> parameters() const;

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t time_embed_dim_;

    GroupNorm norm1_;
    GroupNorm norm2_;

    Conv2d conv1_;
    Conv2d conv2_;

    // Projects time embedding to [B, 2*out_channels] (scale, shift)
    Linear time_proj_;

    // Optional 1x1 conv for skip path
    bool use_skip_conv_;
    Conv2d skip_conv_;
};

class AttentionBlock {
public:
    explicit AttentionBlock(size_t channels);

    Tensor forward(const Tensor& x);

    std::vector<Tensor> parameters() const;

private:
    size_t channels_;

    // Lightweight attention-like gating.
    GroupNorm norm_;
    Conv2d qkv_proj_;
    Conv2d out_proj_;
};

class UNet {
public:
    UNet(size_t in_channels = 3,
         size_t out_channels = 3,
         size_t hidden_channels = 128,
         std::vector<size_t> attention_at_res = {8});

    Tensor forward(const Tensor& x, size_t t);
    Tensor forward(const Tensor& x, const Tensor& timesteps);

    size_t in_channels() const { return in_channels_; }
    size_t out_channels() const { return out_channels_; }

    std::vector<Tensor> parameters() const;

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t hidden_channels_;
    std::vector<size_t> attention_at_res_;

    TimeEmbedding time_embed_;

    Conv2d in_conv_;

    // 3-stage UNet with channel multipliers [1,2,4]
    ResnetBlock down0_a_;
    ResnetBlock down0_b_;
    AttentionBlock down0_attn_;
    bool down0_use_attn_;
    Conv2d down0_downsample_;

    ResnetBlock down1_a_;
    ResnetBlock down1_b_;
    AttentionBlock down1_attn_;
    bool down1_use_attn_;
    Conv2d down1_downsample_;

    ResnetBlock down2_a_;
    ResnetBlock down2_b_;
    AttentionBlock down2_attn_;
    bool down2_use_attn_;

    ResnetBlock mid_a_;
    AttentionBlock mid_attn_;
    ResnetBlock mid_b_;

    // Upsampling
    ResnetBlock up2_a_;
    ResnetBlock up2_b_;
    AttentionBlock up2_attn_;
    bool up2_use_attn_;

    ResnetBlock up1_a_;
    ResnetBlock up1_b_;
    AttentionBlock up1_attn_;
    bool up1_use_attn_;

    ResnetBlock up0_a_;
    ResnetBlock up0_b_;
    AttentionBlock up0_attn_;
    bool up0_use_attn_;

    GroupNorm out_norm_;
    Conv2d out_conv_;

    bool attention_enabled_for_res(size_t res) const;
};

} // namespace nn
} // namespace jin
