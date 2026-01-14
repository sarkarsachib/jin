#include "pooling.h"
#include "../autodiff/tape.h"
#include <algorithm>
#include <cmath>

namespace jin {
namespace nn {

// ==================== MaxPool1d ====================

MaxPool1d::MaxPool1d(size_t kernel_size, size_t stride, size_t padding)
    : kernel_size_(kernel_size),
      stride_(stride == 0 ? kernel_size : stride),
      padding_(padding) {}

Tensor MaxPool1d::forward(const Tensor& x) {
    Shape x_shape = x.shape();  // [batch, channels, length]
    
    if (x_shape.size() != 3) {
        throw std::invalid_argument("MaxPool1d expects 3D input [batch, channels, length]");
    }
    
    size_t batch = x_shape[0];
    size_t channels = x_shape[1];
    size_t in_length = x_shape[2];
    
    size_t out_length = (in_length + 2 * padding_ - kernel_size_) / stride_ + 1;
    
    Shape out_shape = {batch, channels, out_length};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t out_i = 0; out_i < out_length; ++out_i) {
                float max_val = -std::numeric_limits<float>::infinity();
                
                size_t in_start = out_i * stride_ < padding_ ? 0 : out_i * stride_ - padding_;
                size_t in_end = std::min(in_start + kernel_size_, in_length + padding_);
                
                // Adjust for padding
                for (size_t in_i = in_start; in_i < in_end; ++in_i) {
                    if (in_i < in_length) {
                        float val = x_data[(b * channels + c) * in_length + in_i];
                        if (val > max_val) max_val = val;
                    }
                }
                
                out_data[(b * channels + c) * out_length + out_i] = max_val;
            }
        }
    }
    
    return out;
}

// ==================== MaxPool2d ====================

MaxPool2d::MaxPool2d(size_t kernel_size, size_t stride_h, size_t stride_w,
                     size_t padding_h, size_t padding_w)
    : kernel_size_(kernel_size),
      stride_h_(stride_h == 0 ? kernel_size : stride_h),
      stride_w_(stride_w == 0 ? kernel_size : stride_w),
      padding_h_(padding_h),
      padding_w_(padding_w) {}

Tensor MaxPool2d::forward(const Tensor& x) {
    Shape x_shape = x.shape();  // [batch, channels, height, width]
    
    if (x_shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d expects 4D input [batch, channels, height, width]");
    }
    
    size_t batch = x_shape[0];
    size_t channels = x_shape[1];
    size_t in_h = x_shape[2];
    size_t in_w = x_shape[3];
    
    size_t out_h = (in_h + 2 * padding_h_ - kernel_size_) / stride_h_ + 1;
    size_t out_w = (in_w + 2 * padding_w_ - kernel_size_) / stride_w_ + 1;
    
    Shape out_shape = {batch, channels, out_h, out_w};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t out_i = 0; out_i < out_h; ++out_i) {
                size_t in_start_h = out_i * stride_h_ < padding_h_ ? 0 : out_i * stride_h_ - padding_h_;
                
                for (size_t out_j = 0; out_j < out_w; ++out_j) {
                    size_t in_start_w = out_j * stride_w_ < padding_w_ ? 0 : out_j * stride_w_ - padding_w_;
                    
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (size_t ki = 0; ki < kernel_size_; ++ki) {
                        size_t in_i = in_start_h + ki;
                        if (in_i >= in_h + padding_h_) break;
                        if (in_i < in_h) {
                            for (size_t kj = 0; kj < kernel_size_; ++kj) {
                                size_t in_j = in_start_w + kj;
                                if (in_j >= in_w + padding_w_) break;
                                if (in_j < in_w) {
                                    float val = x_data[((b * channels + c) * in_h + in_i) * in_w + in_j];
                                    if (val > max_val) max_val = val;
                                }
                            }
                        }
                    }
                    
                    out_data[((b * channels + c) * out_h + out_i) * out_w + out_j] = max_val;
                }
            }
        }
    }
    
    return out;
}

// ==================== AvgPool1d ====================

AvgPool1d::AvgPool1d(size_t kernel_size, size_t stride, bool count_include_pad)
    : kernel_size_(kernel_size),
      stride_(stride == 0 ? kernel_size : stride),
      count_include_pad_(count_include_pad) {}

Tensor AvgPool1d::forward(const Tensor& x) {
    Shape x_shape = x.shape();  // [batch, channels, length]
    
    if (x_shape.size() != 3) {
        throw std::invalid_argument("AvgPool1d expects 3D input [batch, channels, length]");
    }
    
    size_t batch = x_shape[0];
    size_t channels = x_shape[1];
    size_t in_length = x_shape[2];
    
    size_t out_length = (in_length + kernel_size_ - 1) / stride_;  // Same padding approximation
    
    Shape out_shape = {batch, channels, out_length};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t out_i = 0; out_i < out_length; ++out_i) {
                size_t in_start = out_i * stride_;
                size_t in_end = std::min(in_start + kernel_size_, in_length);
                
                float sum = 0.0f;
                size_t count = count_include_pad_ ? kernel_size_ : (in_end - in_start);
                
                for (size_t in_i = in_start; in_i < in_end; ++in_i) {
                    sum += x_data[(b * channels + c) * in_length + in_i];
                }
                
                out_data[(b * channels + c) * out_length + out_i] = sum / static_cast<float>(count);
            }
        }
    }
    
    return out;
}

// ==================== AvgPool2d ====================

AvgPool2d::AvgPool2d(size_t kernel_size, size_t stride_h, size_t stride_w,
                     bool count_include_pad)
    : kernel_size_(kernel_size),
      stride_h_(stride_h == 0 ? kernel_size : stride_h),
      stride_w_(stride_w == 0 ? kernel_size : stride_w),
      count_include_pad_(count_include_pad) {}

Tensor AvgPool2d::forward(const Tensor& x) {
    Shape x_shape = x.shape();  // [batch, channels, height, width]
    
    if (x_shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d expects 4D input [batch, channels, height, width]");
    }
    
    size_t batch = x_shape[0];
    size_t channels = x_shape[1];
    size_t in_h = x_shape[2];
    size_t in_w = x_shape[3];
    
    size_t out_h = (in_h + kernel_size_ - 1) / stride_h_;
    size_t out_w = (in_w + kernel_size_ - 1) / stride_w_;
    
    Shape out_shape = {batch, channels, out_h, out_w};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t out_i = 0; out_i < out_h; ++out_i) {
                size_t in_start_h = out_i * stride_h_;
                
                for (size_t out_j = 0; out_j < out_w; ++out_j) {
                    size_t in_start_w = out_j * stride_w_;
                    
                    float sum = 0.0f;
                    size_t count = count_include_pad_ ? kernel_size_ * kernel_size_ : 0;
                    
                    for (size_t ki = 0; ki < kernel_size_; ++ki) {
                        size_t in_i = in_start_h + ki;
                        if (in_i >= in_h) break;
                        
                        for (size_t kj = 0; kj < kernel_size_; ++kj) {
                            size_t in_j = in_start_w + kj;
                            if (in_j >= in_w) break;
                            
                            sum += x_data[((b * channels + c) * in_h + in_i) * in_w + in_j];
                            count++;
                        }
                    }
                    
                    out_data[((b * channels + c) * out_h + out_i) * out_w + out_j] = 
                        sum / static_cast<float>(count);
                }
            }
        }
    }
    
    return out;
}

// ==================== AdaptiveAvgPool1d ====================

AdaptiveAvgPool1d::AdaptiveAvgPool1d(size_t output_size)
    : output_size_(output_size) {}

Tensor AdaptiveAvgPool1d::forward(const Tensor& x) {
    Shape x_shape = x.shape();  // [batch, channels, length]
    
    if (x_shape.size() != 3) {
        throw std::invalid_argument("AdaptiveAvgPool1d expects 3D input [batch, channels, length]");
    }
    
    size_t batch = x_shape[0];
    size_t channels = x_shape[1];
    size_t in_length = x_shape[2];
    
    Shape out_shape = {batch, channels, output_size_};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    float stride = static_cast<float>(in_length) / static_cast<float>(output_size_);
    float start = stride / 2.0f;
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t out_i = 0; out_i < output_size_; ++out_i) {
                float center = start + out_i * stride;
                float in_start = center - stride / 2.0f;
                float in_end = center + stride / 2.0f;
                
                // Clamp to valid range
                size_t i_start = static_cast<size_t>(std::max(0.0f, in_start));
                size_t i_end = static_cast<size_t>(std::min(static_cast<float>(in_length), in_end));
                
                float sum = 0.0f;
                size_t count = 0;
                for (size_t in_i = i_start; in_i < i_end && in_i < in_length; ++in_i) {
                    sum += x_data[(b * channels + c) * in_length + in_i];
                    count++;
                }
                
                out_data[(b * channels + c) * output_size_ + out_i] = 
                    count > 0 ? sum / static_cast<float>(count) : 0.0f;
            }
        }
    }
    
    return out;
}

// ==================== AdaptiveAvgPool2d ====================

AdaptiveAvgPool2d::AdaptiveAvgPool2d(size_t output_h, size_t output_w)
    : output_h_(output_h), output_w_(output_w) {}

Tensor AdaptiveAvgPool2d::forward(const Tensor& x) {
    Shape x_shape = x.shape();  // [batch, channels, height, width]
    
    if (x_shape.size() != 4) {
        throw std::invalid_argument("AdaptiveAvgPool2d expects 4D input [batch, channels, height, width]");
    }
    
    size_t batch = x_shape[0];
    size_t channels = x_shape[1];
    size_t in_h = x_shape[2];
    size_t in_w = x_shape[3];
    
    Shape out_shape = {batch, channels, output_h_, output_w_};
    Tensor out(out_shape, x.dtype(), x.device());
    
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    float stride_h = static_cast<float>(in_h) / static_cast<float>(output_h_);
    float stride_w = static_cast<float>(in_w) / static_cast<float>(output_w_);
    float start_h = stride_h / 2.0f;
    float start_w = stride_w / 2.0f;
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t out_i = 0; out_i < output_h_; ++out_i) {
                float center_h = start_h + out_i * stride_h;
                float in_start_h = center_h - stride_h / 2.0f;
                float in_end_h = center_h + stride_h / 2.0f;
                
                size_t i_start_h = static_cast<size_t>(std::max(0.0f, in_start_h));
                size_t i_end_h = static_cast<size_t>(std::min(static_cast<float>(in_h), in_end_h));
                
                for (size_t out_j = 0; out_j < output_w_; ++out_j) {
                    float center_w = start_w + out_j * stride_w;
                    float in_start_w = center_w - stride_w / 2.0f;
                    float in_end_w = center_w + stride_w / 2.0f;
                    
                    size_t i_start_w = static_cast<size_t>(std::max(0.0f, in_start_w));
                    size_t i_end_w = static_cast<size_t>(std::min(static_cast<float>(in_w), in_end_w));
                    
                    float sum = 0.0f;
                    size_t count = 0;
                    
                    for (size_t in_i = i_start_h; in_i < i_end_h && in_i < in_h; ++in_i) {
                        for (size_t in_j = i_start_w; in_j < i_end_w && in_j < in_w; ++in_j) {
                            sum += x_data[((b * channels + c) * in_h + in_i) * in_w + in_j];
                            count++;
                        }
                    }
                    
                    out_data[((b * channels + c) * output_h_ + out_i) * output_w_ + out_j] =
                        count > 0 ? sum / static_cast<float>(count) : 0.0f;
                }
            }
        }
    }
    
    return out;
}

} // namespace nn
} // namespace jin
