#include "math_ops.h"
#include "../utils/shapes.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace jin {
namespace ops {

// Helper functions for element-wise operations
void add_kernel(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void mul_kernel(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void sub_kernel(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void div_kernel(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}

void broadcast_binary_op(const Tensor& a, const Tensor& b, Tensor& out,
                         void (*op)(const float*, const float*, float*, size_t)) {
    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    // Simple case: same shape
    if (a.shape() == b.shape()) {
        op(a_data, b_data, out_data, a.size());
        return;
    }
    
    // Broadcasting case - simplified for common patterns
    size_t out_size = out.size();
    size_t a_size = a.size();
    size_t b_size = b.size();
    
    for (size_t i = 0; i < out_size; ++i) {
        float a_val = a_data[i % a_size];
        float b_val = b_data[i % b_size];
        op(&a_val, &b_val, &out_data[i], 1);
    }
}

Tensor add(const Tensor& a, const Tensor& b) {
    Shape out_shape = broadcast_shapes(a.shape(), b.shape());
    Tensor out(out_shape, a.dtype(), a.device());
    broadcast_binary_op(a, b, out, add_kernel);
    return out;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    Shape out_shape = broadcast_shapes(a.shape(), b.shape());
    Tensor out(out_shape, a.dtype(), a.device());
    broadcast_binary_op(a, b, out, mul_kernel);
    return out;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    Shape out_shape = broadcast_shapes(a.shape(), b.shape());
    Tensor out(out_shape, a.dtype(), a.device());
    broadcast_binary_op(a, b, out, sub_kernel);
    return out;
}

Tensor divide(const Tensor& a, const Tensor& b) {
    Shape out_shape = broadcast_shapes(a.shape(), b.shape());
    Tensor out(out_shape, a.dtype(), a.device());
    broadcast_binary_op(a, b, out, div_kernel);
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.ndim() < 2 || b.ndim() < 2) {
        throw std::invalid_argument("matmul requires at least 2D tensors");
    }
    
    size_t m = a.shape()[a.ndim() - 2];
    size_t k = a.shape()[a.ndim() - 1];
    size_t k2 = b.shape()[b.ndim() - 2];
    size_t n = b.shape()[b.ndim() - 1];
    
    if (k != k2) {
        throw std::invalid_argument("Matrix dimensions must match for matmul");
    }
    
    // For now, only support 2D matrix multiplication
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("Only 2D matmul currently supported");
    }
    
    Shape out_shape = {m, n};
    Tensor out(out_shape, a.dtype(), a.device());
    
    const float* a_data = a.data_ptr<float>();
    const float* b_data = b.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    // Simple matrix multiplication (not optimized)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }
    
    return out;
}

Tensor exp(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = std::exp(a_data[i]);
    }
    
    return out;
}

Tensor log(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = std::log(a_data[i]);
    }
    
    return out;
}

Tensor relu(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = std::max(0.0f, a_data[i]);
    }
    
    return out;
}

Tensor tanh(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = std::tanh(a_data[i]);
    }
    
    return out;
}

Tensor sigmoid(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = 1.0f / (1.0f + std::exp(-a_data[i]));
    }
    
    return out;
}

Tensor neg(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = -a_data[i];
    }
    
    return out;
}

Tensor pow(const Tensor& a, float exponent) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = std::pow(a_data[i], exponent);
    }
    
    return out;
}

Tensor sqrt(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = std::sqrt(a_data[i]);
    }
    
    return out;
}

Tensor abs(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = std::abs(a_data[i]);
    }
    
    return out;
}

Tensor sum(const Tensor& a, int axis, bool keepdims) {
    if (axis == -1) {
        // Sum all elements
        const float* a_data = a.data_ptr<float>();
        
        float total = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            total += a_data[i];
        }
        
        // Return scalar tensor (size 1) regardless of keepdims for now
        Tensor out({1}, a.dtype(), a.device());
        float* out_data = out.data_ptr<float>();
        out_data[0] = total;
        
        return out;
    }
    
    // Sum along specific axis
    if (axis < 0) axis += a.ndim();
    if (axis < 0 || axis >= static_cast<int>(a.ndim())) {
        throw std::out_of_range("Axis out of range");
    }
    
    Shape out_shape = infer_shape_after_sum(a.shape(), axis, keepdims);
    Tensor out(out_shape, a.dtype(), a.device());
    
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    // Initialize output to zero
    for (size_t i = 0; i < out.size(); ++i) {
        out_data[i] = 0.0f;
    }
    
    // Compute strides for reduction
    size_t outer_size = 1;
    for (int i = 0; i < axis; ++i) {
        outer_size *= a.shape()[i];
    }
    
    size_t reduce_size = a.shape()[axis];
    
    size_t inner_size = 1;
    for (size_t i = axis + 1; i < a.ndim(); ++i) {
        inner_size *= a.shape()[i];
    }
    
    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t j = 0; j < inner_size; ++j) {
            float sum_val = 0.0f;
            for (size_t k = 0; k < reduce_size; ++k) {
                size_t idx = (i * reduce_size + k) * inner_size + j;
                sum_val += a_data[idx];
            }
            out_data[i * inner_size + j] = sum_val;
        }
    }
    
    return out;
}

Tensor mean(const Tensor& a, int axis, bool keepdims) {
    Tensor sum_result = sum(a, axis, keepdims);
    
    size_t count;
    if (axis == -1) {
        count = a.size();
    } else {
        if (axis < 0) axis += a.ndim();
        count = a.shape()[axis];
    }
    
    float* data = sum_result.data_ptr<float>();
    for (size_t i = 0; i < sum_result.size(); ++i) {
        data[i] /= static_cast<float>(count);
    }
    
    return sum_result;
}

Tensor clamp(const Tensor& a, float min_val, float max_val) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = std::max(min_val, std::min(max_val, a_data[i]));
    }
    
    return out;
}

Tensor sign(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        out_data[i] = (a_data[i] > 0.0f) ? 1.0f : (a_data[i] < 0.0f) ? -1.0f : 0.0f;
    }
    
    return out;
}

Tensor square(const Tensor& a) {
    Tensor out(a.shape(), a.dtype(), a.device());
    const float* a_data = a.data_ptr<float>();
    float* out_data = out.data_ptr<float>();
    
    for (size_t i = 0; i < a.size(); ++i) {
        float v = a_data[i];
        out_data[i] = v * v;
    }
    
    return out;
}

} // namespace ops
} // namespace jin
