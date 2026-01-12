#include "conv.h"
#include "../autodiff/ops.h"
#include "../autodiff/tape.h"
#include "../core/math_ops.h"
#include <algorithm>
#include <cmath>

namespace jin {
namespace nn {

// ==================== Conv1d ====================

Conv1d::Conv1d(size_t in_channels, size_t out_channels, size_t kernel_size,
               size_t stride, size_t padding, size_t dilation,
               PaddingMode padding_mode, Initializer weight_init, Initializer bias_init)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      padding_mode_(padding_mode) {
    
    weight_ = initialize_tensor({out_channels, in_channels, kernel_size}, weight_init);
    weight_.set_requires_grad(true);
    
    bias_ = initialize_tensor({out_channels}, bias_init);
    bias_.set_requires_grad(true);
}

Tensor Conv1d::forward(const Tensor& x) {
    autodiff::TapeGuard tape_guard;
    
    Shape x_shape = x.shape();  // [batch, in_channels, length]
    
    if (x_shape.size() != 3) {
        throw std::invalid_argument("Conv1d expects 3D input [batch, in_channels, length]");
    }
    
    if (x_shape[1] != in_channels_) {
        throw std::invalid_argument("Input channel dimension must match in_channels");
    }
    
    size_t batch = x_shape[0];
    size_t in_len = x_shape[2];
    
    // Calculate output length
    size_t effective_kernel = (kernel_size_ - 1) * dilation_ + 1;
    size_t out_len = (in_len + 2 * padding_ - effective_kernel) / stride_ + 1;
    
    Shape out_shape = {batch, out_channels_, out_len};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight_.data_ptr<float>();
    const float* bias_data = bias_.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    // For each output position
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            for (size_t out_i = 0; out_i < out_len; ++out_i) {
                float sum = bias_data[oc];
                
                // Compute input position
                size_t in_start = out_i * stride_;
                
                // Convolve
                for (size_t ic = 0; ic < in_channels_; ++ic) {
                    for (size_t k = 0; k < kernel_size_; ++k) {
                        size_t dil_k = k * dilation_;
                        size_t in_pos = in_start + dil_k;
                        
                        // Handle padding
                        float x_val;
                        if (in_pos < padding_ || in_pos >= in_len + padding_) {
                            // Outside input (padding)
                            x_val = 0.0f;
                        } else if (in_pos < in_len + 2 * padding_ && 
                                   (padding_mode_ == PaddingMode::Zeros || 
                                    padding_mode_ == PaddingMode::Zeros)) {
                            x_val = 0.0f;
                        } else if (in_pos < padding_) {
                            x_val = 0.0f;
                        } else if (in_pos >= in_len + padding_) {
                            x_val = 0.0f;
                        } else {
                            x_val = x_data[((b * in_channels_ + ic) * in_len) + (in_pos - padding_)];
                        }
                        
                        float w_val = weight_data[(oc * in_channels_ + ic) * kernel_size_ + k];
                        sum += x_val * w_val;
                    }
                }
                
                out_data[((b * out_channels_ + oc) * out_len) + out_i] = sum;
            }
        }
    }
    
    return out;
}

// ==================== Conv2d ====================

Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
               size_t stride_h, size_t stride_w,
               size_t padding_h, size_t padding_w,
               size_t dilation_h, size_t dilation_w,
               PaddingMode padding_mode, Initializer weight_init, Initializer bias_init)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_h_(stride_h),
      stride_w_(stride_w),
      padding_h_(padding_h),
      padding_w_(padding_w),
      dilation_h_(dilation_h),
      dilation_w_(dilation_w),
      padding_mode_(padding_mode) {
    
    weight_ = initialize_tensor({out_channels, in_channels, kernel_size, kernel_size}, weight_init);
    weight_.set_requires_grad(true);
    
    bias_ = initialize_tensor({out_channels}, bias_init);
    bias_.set_requires_grad(true);
}

Tensor Conv2d::forward(const Tensor& x) {
    autodiff::TapeGuard tape_guard;
    
    Shape x_shape = x.shape();  // [batch, in_channels, height, width]
    
    if (x_shape.size() != 4) {
        throw std::invalid_argument("Conv2d expects 4D input [batch, in_channels, height, width]");
    }
    
    if (x_shape[1] != in_channels_) {
        throw std::invalid_argument("Input channel dimension must match in_channels");
    }
    
    size_t batch = x_shape[0];
    size_t in_h = x_shape[2];
    size_t in_w = x_shape[3];
    
    // Calculate output dimensions
    size_t effective_h = (kernel_size_ - 1) * dilation_h_ + 1;
    size_t effective_w = (kernel_size_ - 1) * dilation_w_ + 1;
    size_t out_h = (in_h + 2 * padding_h_ - effective_h) / stride_h_ + 1;
    size_t out_w = (in_w + 2 * padding_w_ - effective_w) / stride_w_ + 1;
    
    Shape out_shape = {batch, out_channels_, out_h, out_w};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight_.data_ptr<float>();
    const float* bias_data = bias_.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    // For each output position
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            for (size_t out_i = 0; out_i < out_h; ++out_i) {
                size_t in_start_h = out_i * stride_h_;
                
                for (size_t out_j = 0; out_j < out_w; ++out_j) {
                    size_t in_start_w = out_j * stride_w_;
                    
                    float sum = bias_data[oc];
                    
                    // Convolve over kernel
                    for (size_t ic = 0; ic < in_channels_; ++ic) {
                        for (size_t kh = 0; kh < kernel_size_; ++kh) {
                            size_t dil_kh = kh * dilation_h_;
                            size_t in_h_pos = in_start_h + dil_kh;
                            
                            for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                size_t dil_kw = kw * dilation_w_;
                                size_t in_w_pos = in_start_w + dil_kw;
                                
                                // Get input value with padding
                                float x_val = 0.0f;
                                if (in_h_pos >= padding_h_ && in_h_pos < in_h + padding_h_ &&
                                    in_w_pos >= padding_w_ && in_w_pos < in_w + padding_w_) {
                                    size_t x_h = in_h_pos - padding_h_;
                                    size_t x_w = in_w_pos - padding_w_;
                                    x_val = x_data[(((b * in_channels_ + ic) * in_h + x_h) * in_w) + x_w];
                                }
                                
                                float w_val = weight_data[(((oc * in_channels_ + ic) * kernel_size_ + kh) * kernel_size_) + kw];
                                sum += x_val * w_val;
                            }
                        }
                    }
                    
                    out_data[(((b * out_channels_ + oc) * out_h + out_i) * out_w) + out_j] = sum;
                }
            }
        }
    }
    
    return out;
}

// ==================== ConvTranspose1d ====================

ConvTranspose1d::ConvTranspose1d(size_t in_channels, size_t out_channels, size_t kernel_size,
                                 size_t stride, size_t padding, size_t output_padding, size_t dilation)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      output_padding_(output_padding),
      dilation_(dilation) {
    
    weight_ = initialize_tensor({in_channels, out_channels, kernel_size}, Initializer::kaiming_normal(2.0f));
    weight_.set_requires_grad(true);
}

Tensor ConvTranspose1d::forward(const Tensor& x) {
    Shape x_shape = x.shape();  // [batch, in_channels, length]
    
    if (x_shape.size() != 3) {
        throw std::invalid_argument("ConvTranspose1d expects 3D input [batch, in_channels, length]");
    }
    
    size_t batch = x_shape[0];
    size_t in_len = x_shape[2];
    
    // Output length calculation
    size_t effective_kernel = (kernel_size_ - 1) * dilation_ + 1;
    size_t out_len = (in_len - 1) * stride_ - 2 * padding_ + effective_kernel + output_padding_;
    
    Shape out_shape = {batch, out_channels_, out_len};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight_.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    // Initialize output to zeros
    for (size_t i = 0; i < out.size(); ++i) {
        out_data[i] = 0.0f;
    }
    
    // Transposed convolution (gradient of conv)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t ic = 0; ic < in_channels_; ++ic) {
            for (size_t in_i = 0; in_i < in_len; ++in_i) {
                for (size_t oc = 0; oc < out_channels_; ++oc) {
                    for (size_t k = 0; k < kernel_size_; ++k) {
                        size_t dil_k = k * dilation_;
                        size_t out_pos = in_i * stride_ + dil_k - padding_;
                        
                        if (out_pos < out_len) {
                            float x_val = x_data[((b * in_channels_ + ic) * in_len) + in_i];
                            float w_val = weight_data[((ic * out_channels_ + oc) * kernel_size_) + k];
                            out_data[((b * out_channels_ + oc) * out_len) + out_pos] += x_val * w_val;
                        }
                    }
                }
            }
        }
    }
    
    return out;
}

// ==================== ConvTranspose2d ====================

ConvTranspose2d::ConvTranspose2d(size_t in_channels, size_t out_channels, size_t kernel_size,
                                 size_t stride_h, size_t stride_w,
                                 size_t padding_h, size_t padding_w,
                                 size_t output_padding_h, size_t output_padding_w,
                                 size_t dilation_h, size_t dilation_w)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_h_(stride_h),
      stride_w_(stride_w),
      padding_h_(padding_h),
      padding_w_(padding_w),
      output_padding_h_(output_padding_h),
      output_padding_w_(output_padding_w),
      dilation_h_(dilation_h),
      dilation_w_(dilation_w) {
    
    weight_ = initialize_tensor({in_channels, out_channels, kernel_size, kernel_size}, Initializer::kaiming_normal(2.0f));
    weight_.set_requires_grad(true);
}

Tensor ConvTranspose2d::forward(const Tensor& x) {
    Shape x_shape = x.shape();  // [batch, in_channels, height, width]
    
    if (x_shape.size() != 4) {
        throw std::invalid_argument("ConvTranspose2d expects 4D input [batch, in_channels, height, width]");
    }
    
    size_t batch = x_shape[0];
    size_t in_h = x_shape[2];
    size_t in_w = x_shape[3];
    
    // Output dimensions
    size_t effective_h = (kernel_size_ - 1) * dilation_h_ + 1;
    size_t effective_w = (kernel_size_ - 1) * dilation_w_ + 1;
    size_t out_h = (in_h - 1) * stride_h_ - 2 * padding_h_ + effective_h + output_padding_h_;
    size_t out_w = (in_w - 1) * stride_w_ - 2 * padding_w_ + effective_w + output_padding_w_;
    
    Shape out_shape = {batch, out_channels_, out_h, out_w};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    const float* weight_data = weight_.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    // Initialize output to zeros
    for (size_t i = 0; i < out.size(); ++i) {
        out_data[i] = 0.0f;
    }
    
    // Transposed convolution
    for (size_t b = 0; b < batch; ++b) {
        for (size_t ic = 0; ic < in_channels_; ++ic) {
            for (size_t in_i = 0; in_i < in_h; ++in_i) {
                for (size_t in_j = 0; in_j < in_w; ++in_j) {
                    for (size_t oc = 0; oc < out_channels_; ++oc) {
                        for (size_t kh = 0; kh < kernel_size_; ++kh) {
                            size_t dil_kh = kh * dilation_h_;
                            size_t out_h_pos = in_i * stride_h_ + dil_kh - padding_h_;
                            
                            for (size_t kw = 0; kw < kernel_size_; ++kw) {
                                size_t dil_kw = kw * dilation_w_;
                                size_t out_w_pos = in_j * stride_w_ + dil_kw - padding_w_;
                                
                                if (out_h_pos < out_h && out_w_pos < out_w) {
                                    float x_val = x_data[(((b * in_channels_ + ic) * in_h + in_i) * in_w) + in_j];
                                    float w_val = weight_data[((((ic * out_channels_ + oc) * kernel_size_ + kh) * kernel_size_) + kw)];
                                    out_data[(((b * out_channels_ + oc) * out_h + out_h_pos) * out_w) + out_w_pos] += x_val * w_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return out;
}

} // namespace nn
} // namespace jin
