#pragma once

#include "../core/tensor.h"
#include "common.h"

namespace jin {
namespace nn {

enum class PaddingMode {
    Zeros,
    Reflect,
    Circular
};

class Conv1d {
public:
    Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, size_t padding = 0, size_t dilation = 1,
           PaddingMode padding_mode = PaddingMode::Zeros,
           Initializer weight_init = Initializer::kaiming_normal(2.0f),
           Initializer bias_init = Initializer::zeros());
    
    Tensor forward(const Tensor& x);
    
    // Properties
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }
    size_t in_channels() const { return in_channels_; }
    size_t out_channels() const { return out_channels_; }
    size_t kernel_size() const { return kernel_size_; }
    size_t stride() const { return stride_; }
    size_t padding() const { return padding_; }
    size_t dilation() const { return dilation_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t dilation_;
    PaddingMode padding_mode_;
    Tensor weight_;
    Tensor bias_;
};

class Conv2d {
public:
    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride_h = 1, size_t stride_w = 1,
           size_t padding_h = 0, size_t padding_w = 0,
           size_t dilation_h = 1, size_t dilation_w = 1,
           PaddingMode padding_mode = PaddingMode::Zeros,
           Initializer weight_init = Initializer::kaiming_normal(2.0f),
           Initializer bias_init = Initializer::zeros());
    
    Tensor forward(const Tensor& x);
    
    // Properties
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }
    size_t in_channels() const { return in_channels_; }
    size_t out_channels() const { return out_channels_; }
    size_t kernel_size() const { return kernel_size_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;
    size_t dilation_h_;
    size_t dilation_w_;
    PaddingMode padding_mode_;
    Tensor weight_;
    Tensor bias_;
};

// Transposed convolution (deconvolution)
class ConvTranspose1d {
public:
    ConvTranspose1d(size_t in_channels, size_t out_channels, size_t kernel_size,
                    size_t stride = 1, size_t padding = 0, size_t output_padding = 0,
                    size_t dilation = 1);
    
    Tensor forward(const Tensor& x);

private:
    // Simplified implementation
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t output_padding_;
    size_t dilation_;
    Tensor weight_;
    Tensor bias_;
};

class ConvTranspose2d {
public:
    ConvTranspose2d(size_t in_channels, size_t out_channels, size_t kernel_size,
                    size_t stride_h = 1, size_t stride_w = 1,
                    size_t padding_h = 0, size_t padding_w = 0,
                    size_t output_padding_h = 0, size_t output_padding_w = 0,
                    size_t dilation_h = 1, size_t dilation_w = 1);
    
    Tensor forward(const Tensor& x);

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;
    size_t output_padding_h_;
    size_t output_padding_w_;
    size_t dilation_h_;
    size_t dilation_w_;
    Tensor weight_;
    Tensor bias_;
};

} // namespace nn
} // namespace jin
