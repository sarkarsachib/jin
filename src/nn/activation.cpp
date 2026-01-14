#include "activation.h"
#include "../core/math_ops.h"
#include "../autodiff/ops.h"
#include <cmath>

namespace jin {
namespace nn {

// ==================== ReLU ====================

Tensor relu(const Tensor& x) {
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < x.size(); ++i) {
        out_data[i] = std::max(0.0f, x_data[i]);
    }
    
    return out;
}

Tensor relu_backward(const Tensor& grad, const Tensor& x) {
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* grad_data = grad.data_ptr<float>();
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < x.size(); ++i) {
        out_data[i] = (x_data[i] > 0.0f) ? grad_data[i] : 0.0f;
    }
    
    return out;
}

// ==================== Leaky ReLU ====================

Tensor leaky_relu(const Tensor& x, float negative_slope) {
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < x.size(); ++i) {
        out_data[i] = (x_data[i] > 0.0f) ? x_data[i] : negative_slope * x_data[i];
    }
    
    return out;
}

Tensor leaky_relu_backward(const Tensor& grad, const Tensor& x, float negative_slope) {
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* grad_data = grad.data_ptr<float>();
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < x.size(); ++i) {
        out_data[i] = (x_data[i] > 0.0f) ? grad_data[i] : negative_slope * grad_data[i];
    }
    
    return out;
}

// ==================== Sigmoid ====================

Tensor sigmoid(const Tensor& x) {
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < x.size(); ++i) {
        out_data[i] = 1.0f / (1.0f + std::exp(-x_data[i]));
    }
    
    return out;
}

Tensor sigmoid_backward(const Tensor& grad, const Tensor& x) {
    Tensor sig = sigmoid(x);
    Tensor one_minus_sig = ops::subtract(jin::ones(sig.shape(), sig.dtype(), sig.device()), sig);
    Tensor factor = ops::multiply(sig, one_minus_sig);
    return ops::multiply(grad, factor);
}

// ==================== Tanh ====================

Tensor tanh(const Tensor& x) {
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < x.size(); ++i) {
        out_data[i] = std::tanh(x_data[i]);
    }
    
    return out;
}

Tensor tanh_backward(const Tensor& grad, const Tensor& x) {
    Tensor t = tanh(x);
    Tensor t_sq = ops::multiply(t, t);
    Tensor ones = jin::ones(t.shape(), t.dtype(), t.device());
    Tensor factor = ops::subtract(ones, t_sq);
    return ops::multiply(grad, factor);
}

// ==================== GELU ====================

Tensor gelu(const Tensor& x, bool approximate) {
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    if (approximate) {
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        const float coeff = 0.044715f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            float xi = x_data[i];
            float cube = xi * xi * xi;
            float inner = sqrt_2_over_pi * (xi + coeff * cube);
            float tanh_inner = std::tanh(inner);
            out_data[i] = 0.5f * xi * (1.0f + tanh_inner);
        }
    } else {
        for (size_t i = 0; i < x.size(); ++i) {
            float xi = x_data[i];
            float cdf = 0.5f * (1.0f + std::erf(xi / std::sqrt(2.0f)));
            out_data[i] = xi * cdf;
        }
    }
    
    return out;
}

Tensor gelu_backward(const Tensor& grad, const Tensor& x, bool approximate) {
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* grad_data = grad.data_ptr<float>();
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float coeff = 0.044715f;
    
    for (size_t i = 0; i < x.size(); ++i) {
        float xi = x_data[i];
        float cube = xi * xi * xi;
        float inner = sqrt_2_over_pi * (xi + coeff * cube);
        float tanh_inner = std::tanh(inner);
        float sech_sq = 1.0f - tanh_inner * tanh_inner;
        
        float term1 = 0.5f * (1.0f + tanh_inner);
        float term2 = 0.5f * xi * sech_sq * sqrt_2_over_pi * (1.0f + 3.0f * coeff * xi * xi);
        
        out_data[i] = grad_data[i] * (term1 + term2);
    }
    
    return out;
}

// ==================== SiLU (Swish-1) ====================

Tensor silu(const Tensor& x) {
    Tensor sig = sigmoid(x);
    return ops::multiply(x, sig);
}

Tensor silu_backward(const Tensor& grad, const Tensor& x) {
    Tensor sig = sigmoid(x);
    Tensor one = jin::ones(sig.shape(), sig.dtype(), sig.device());
    Tensor x_minus = ops::subtract(x, ops::multiply(x, sig));
    Tensor factor = ops::multiply(sig, ops::add(one, x_minus));
    return ops::multiply(grad, factor);
}

// ==================== Swish ====================

Tensor swish(const Tensor& x, float beta) {
    Tensor scaled = ops::multiply(x, jin::ones({1}, x.dtype(), x.device()));
    float* s_data = scaled.data_ptr<float>();
    for (size_t i = 0; i < scaled.size(); ++i) s_data[i] *= beta;
    Tensor sig = sigmoid(scaled);
    return ops::multiply(x, sig);
}

Tensor swish_backward(const Tensor& grad, const Tensor& x, float beta) {
    Tensor scaled = ops::multiply(x, jin::ones({1}, x.dtype(), x.device()));
    float* s_data = scaled.data_ptr<float>();
    for (size_t i = 0; i < scaled.size(); ++i) s_data[i] *= beta;
    Tensor sig = sigmoid(scaled);
    Tensor sig_term = ops::multiply(x, sig);
    Tensor one = jin::ones(x.shape(), x.dtype(), x.device());
    Tensor factor = ops::add(sig, ops::multiply(ops::subtract(one, sig), sig_term));
    return ops::multiply(grad, factor);
}

// ==================== Mish ====================

Tensor mish(const Tensor& x) {
    Tensor one = jin::ones(x.shape(), x.dtype(), x.device());
    Tensor sp = ops::log(ops::add(ops::exp(x), one));
    Tensor tanh_sp = tanh(sp);
    return ops::multiply(x, tanh_sp);
}

Tensor mish_backward(const Tensor& grad, const Tensor& x) {
    Tensor one = jin::ones(x.shape(), x.dtype(), x.device());
    Tensor sp = ops::log(ops::add(ops::exp(x), one));
    Tensor tanh_sp = tanh(sp);
    Tensor sech_sq = ops::subtract(one, ops::multiply(tanh_sp, tanh_sp));
    Tensor exp_x = ops::exp(x);
    Tensor factor = ops::add(tanh_sp, ops::multiply(ops::multiply(x, sech_sq), ops::divide(exp_x, ops::add(exp_x, one))));
    return ops::multiply(grad, factor);
}

// ==================== Softmax ====================

Tensor softmax(const Tensor& x, int axis) {
    if (axis < 0) axis += x.ndim();
    
    Shape input_shape = x.shape();
    size_t axis_size = input_shape[axis];
    
    Tensor out(x.shape(), x.dtype(), x.device());
    const float* x_data = x.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= input_shape[i];
    }
    
    size_t inner_size = 1;
    for (size_t i = axis + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }
    
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < inner_size; ++j) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t k = 0; k < axis_size; ++k) {
                size_t idx = (i * axis_size + k) * inner_size + j;
                if (x_data[idx] > max_val) max_val = x_data[idx];
            }
            
            float sum = 0.0f;
            for (size_t k = 0; k < axis_size; ++k) {
                size_t idx = (i * axis_size + k) * inner_size + j;
                out_data[idx] = std::exp(x_data[idx] - max_val);
                sum += out_data[idx];
            }
            
            for (size_t k = 0; k < axis_size; ++k) {
                size_t idx = (i * axis_size + k) * inner_size + j;
                out_data[idx] /= sum;
            }
        }
    }
    
    return out;
}

Tensor softmax_backward(const Tensor& grad, const Tensor& x, int axis) {
    Tensor softmax_x = softmax(x, axis);
    Tensor prod = ops::multiply(grad, softmax_x);
    Tensor sum_axis = ops::sum(prod, axis, false);
    
    Shape shape(softmax_x.shape().size(), 1);
    shape[axis] = softmax_x.shape()[axis];
    sum_axis = sum_axis.reshape(shape);
    
    Tensor factor = ops::multiply(softmax_x, ops::subtract(grad, sum_axis));
    return ops::multiply(softmax_x, ops::sum(factor, axis, false));
}

// ==================== Log Softmax ====================

Tensor log_softmax(const Tensor& x, int axis) {
    if (axis < 0) axis += x.ndim();
    
    Tensor exp_x = ops::exp(x);
    Tensor sum_exp = ops::sum(exp_x, axis, true);
    Tensor log_sum = ops::log(sum_exp);
    return ops::subtract(x, log_sum);
}

} // namespace nn
} // namespace jin
