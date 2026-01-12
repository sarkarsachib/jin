#pragma once

#include "../core/tensor.h"
#include "common.h"

namespace jin {
namespace nn {

class MaxPool1d {
public:
    MaxPool1d(size_t kernel_size, size_t stride = 0, size_t padding = 0);
    
    Tensor forward(const Tensor& x);
    
    size_t kernel_size() const { return kernel_size_; }
    size_t stride() const { return stride_; }
    size_t padding() const { return padding_; }

private:
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
};

class MaxPool2d {
public:
    MaxPool2d(size_t kernel_size, size_t stride_h = 0, size_t stride_w = 0,
              size_t padding_h = 0, size_t padding_w = 0);
    
    Tensor forward(const Tensor& x);
    
    size_t kernel_size() const { return kernel_size_; }
    size_t stride_h() const { return stride_h_; }
    size_t stride_w() const { return stride_w_; }
    size_t padding_h() const { return padding_h_; }
    size_t padding_w() const { return padding_w_; }

private:
    size_t kernel_size_;
    size_t stride_h_;
    size_t stride_w_;
    size_t padding_h_;
    size_t padding_w_;
};

class AvgPool1d {
public:
    AvgPool1d(size_t kernel_size, size_t stride = 0, bool count_include_pad = true);
    
    Tensor forward(const Tensor& x);
    
    size_t kernel_size() const { return kernel_size_; }
    size_t stride() const { return stride_; }
    bool count_include_pad() const { return count_include_pad_; }

private:
    size_t kernel_size_;
    size_t stride_;
    bool count_include_pad_;
};

class AvgPool2d {
public:
    AvgPool2d(size_t kernel_size, size_t stride_h = 0, size_t stride_w = 0,
              bool count_include_pad = true);
    
    Tensor forward(const Tensor& x);
    
    size_t kernel_size() const { return kernel_size_; }
    size_t stride_h() const { return stride_h_; }
    size_t stride_w() const { return stride_w_; }
    bool count_include_pad() const { return count_include_pad_; }

private:
    size_t kernel_size_;
    size_t stride_h_;
    size_t stride_w_;
    bool count_include_pad_;
};

// Adaptive pooling - output size is specified directly
class AdaptiveAvgPool1d {
public:
    AdaptiveAvgPool1d(size_t output_size);
    
    Tensor forward(const Tensor& x);
    
    size_t output_size() const { return output_size_; }

private:
    size_t output_size_;
};

class AdaptiveAvgPool2d {
public:
    AdaptiveAvgPool2d(size_t output_h, size_t output_w);
    
    Tensor forward(const Tensor& x);
    
    size_t output_h() const { return output_h_; }
    size_t output_w() const { return output_w_; }

private:
    size_t output_h_;
    size_t output_w_;
};

} // namespace nn
} // namespace jin
