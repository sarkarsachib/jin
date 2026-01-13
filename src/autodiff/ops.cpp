#include "ops.h"
#include "tape.h"
#include "../utils/shapes.h"
#include <cmath>

namespace jin {
namespace autodiff {

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor result = ops::add(a, b);
    
    auto backward_fn = [a, b](const Tensor& grad) -> std::vector<Tensor> {
        // d(a+b)/da = 1, d(a+b)/db = 1
        Tensor grad_a = grad;
        Tensor grad_b = grad;
        
        // Handle broadcasting - sum out added dimensions
        if (a.shape() != grad.shape()) {
            // Simplified: assume grad has same shape as output
            grad_a = grad;
        }
        if (b.shape() != grad.shape()) {
            grad_b = grad;
        }
        
        return {grad_a, grad_b};
    };
    
    GradientTape::instance().record({a, b}, result, backward_fn, "add");
    
    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor multiply(const Tensor& a, const Tensor& b) {
    Tensor result = ops::multiply(a, b);
    
    auto backward_fn = [a, b](const Tensor& grad) -> std::vector<Tensor> {
        // d(a*b)/da = b, d(a*b)/db = a
        Tensor grad_a = ops::multiply(grad, b);
        Tensor grad_b = ops::multiply(grad, a);
        
        return {grad_a, grad_b};
    };
    
    GradientTape::instance().record({a, b}, result, backward_fn, "multiply");
    
    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor subtract(const Tensor& a, const Tensor& b) {
    Tensor result = ops::subtract(a, b);
    
    auto backward_fn = [a, b](const Tensor& grad) -> std::vector<Tensor> {
        // d(a-b)/da = 1, d(a-b)/db = -1
        Tensor grad_a = grad;
        Tensor grad_b = ops::neg(grad);
        
        return {grad_a, grad_b};
    };
    
    GradientTape::instance().record({a, b}, result, backward_fn, "subtract");
    
    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor divide(const Tensor& a, const Tensor& b) {
    Tensor result = ops::divide(a, b);
    
    auto backward_fn = [a, b, result](const Tensor& grad) -> std::vector<Tensor> {
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        Tensor grad_a = ops::divide(grad, b);
        Tensor grad_b = ops::neg(ops::multiply(grad, ops::divide(result, b)));
        
        return {grad_a, grad_b};
    };
    
    GradientTape::instance().record({a, b}, result, backward_fn, "divide");
    
    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    Tensor result = ops::matmul(a, b);
    
    auto backward_fn = [a, b](const Tensor& grad) -> std::vector<Tensor> {
        // d(AB)/dA = grad @ B^T
        // d(AB)/dB = A^T @ grad
        Tensor b_t = b.transpose({1, 0});
        Tensor a_t = a.transpose({1, 0});
        
        Tensor grad_a = ops::matmul(grad, b_t);
        Tensor grad_b = ops::matmul(a_t, grad);
        
        return {grad_a, grad_b};
    };
    
    GradientTape::instance().record({a, b}, result, backward_fn, "matmul");
    
    if (a.requires_grad() || b.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor exp(const Tensor& a) {
    Tensor result = ops::exp(a);
    
    auto backward_fn = [result](const Tensor& grad) -> std::vector<Tensor> {
        // d(exp(a))/da = exp(a)
        return {ops::multiply(grad, result)};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "exp");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor log(const Tensor& a) {
    Tensor result = ops::log(a);
    
    auto backward_fn = [a](const Tensor& grad) -> std::vector<Tensor> {
        // d(log(a))/da = 1/a
        return {ops::divide(grad, a)};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "log");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor relu(const Tensor& a) {
    Tensor result = ops::relu(a);
    
    auto backward_fn = [a](const Tensor& grad) -> std::vector<Tensor> {
        // d(relu(a))/da = a > 0 ? 1 : 0
        Tensor grad_a(a.shape(), a.dtype(), a.device());
        const float* a_data = a.data_ptr<float>();
        const float* grad_data = grad.data_ptr<float>();
        float* grad_a_data = grad_a.data_ptr<float>();
        
        for (size_t i = 0; i < a.size(); ++i) {
            grad_a_data[i] = a_data[i] > 0 ? grad_data[i] : 0.0f;
        }
        
        return {grad_a};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "relu");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor tanh(const Tensor& a) {
    Tensor result = ops::tanh(a);
    
    auto backward_fn = [result](const Tensor& grad) -> std::vector<Tensor> {
        // d(tanh(a))/da = 1 - tanh(a)^2
        Tensor ones_t = jin::ones(result.shape(), result.dtype(), result.device());
        Tensor tanh_sq = ops::multiply(result, result);
        Tensor grad_factor = ops::subtract(ones_t, tanh_sq);
        
        return {ops::multiply(grad, grad_factor)};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "tanh");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor sigmoid(const Tensor& a) {
    Tensor result = ops::sigmoid(a);
    
    auto backward_fn = [result](const Tensor& grad) -> std::vector<Tensor> {
        // d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a))
        Tensor ones_t = jin::ones(result.shape(), result.dtype(), result.device());
        Tensor one_minus_sig = ops::subtract(ones_t, result);
        Tensor grad_factor = ops::multiply(result, one_minus_sig);
        
        return {ops::multiply(grad, grad_factor)};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "sigmoid");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor neg(const Tensor& a) {
    Tensor result = ops::neg(a);
    
    auto backward_fn = [](const Tensor& grad) -> std::vector<Tensor> {
        // d(-a)/da = -1
        return {ops::neg(grad)};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "neg");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor sum(const Tensor& a, int axis, bool keepdims) {
    Tensor result = ops::sum(a, axis, keepdims);
    
    auto backward_fn = [a, axis, keepdims](const Tensor& grad) -> std::vector<Tensor> {
        // Gradient is broadcast back to input shape
        Tensor grad_expanded = grad;
        
        // If keepdims=false, we need to unsqueeze the dimension
        if (!keepdims && axis != -1) {
            // For now, return gradient with same shape
            // Full implementation would require shape manipulation
        }
        
        // Broadcast gradient to input shape
        Tensor grad_a(a.shape(), a.dtype(), a.device());
        const float* grad_data = grad_expanded.data_ptr<float>();
        float* grad_a_data = grad_a.data_ptr<float>();
        
        // Simple broadcasting - copy gradient to all positions
        for (size_t i = 0; i < a.size(); ++i) {
            grad_a_data[i] = grad_data[0];  // Simplified
        }
        
        return {grad_a};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "sum");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor mean(const Tensor& a, int axis, bool keepdims) {
    Tensor result = ops::mean(a, axis, keepdims);
    
    auto backward_fn = [a, axis, keepdims](const Tensor& grad) -> std::vector<Tensor> {
        // Gradient is broadcast back and divided by count
        size_t count;
        if (axis == -1) {
            count = a.size();
        } else {
            int ax = axis;
            if (ax < 0) ax += a.ndim();
            count = a.shape()[ax];
        }
        
        Tensor grad_a(a.shape(), a.dtype(), a.device());
        const float* grad_data = grad.data_ptr<float>();
        float* grad_a_data = grad_a.data_ptr<float>();
        
        float grad_val = grad_data[0] / static_cast<float>(count);
        for (size_t i = 0; i < a.size(); ++i) {
            grad_a_data[i] = grad_val;
        }
        
        return {grad_a};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "mean");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor sqrt(const Tensor& a) {
    Tensor result = ops::sqrt(a);
    
    auto backward_fn = [result](const Tensor& grad) -> std::vector<Tensor> {
        // d(sqrt(a))/da = 1 / (2 * sqrt(a))
        Tensor half = ops::multiply(jin::ones(result.shape(), result.dtype(), result.device()),
                                    jin::ones({1}, DType::FP32, Device::cpu()));
        // Scale by 0.5
        float* half_data = half.data_ptr<float>();
        for (size_t i = 0; i < half.size(); ++i) {
            half_data[i] *= 0.5f;
        }
        Tensor inv_two_sqrt = ops::divide(half, result);
        return {ops::multiply(grad, inv_two_sqrt)};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "sqrt");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor abs(const Tensor& a) {
    Tensor result = ops::abs(a);
    
    auto backward_fn = [a](const Tensor& grad) -> std::vector<Tensor> {
        // d(abs(a))/da = sign(a)
        Tensor sign = ops::sign(a);
        return {ops::multiply(grad, sign)};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "abs");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor clamp(const Tensor& a, float min_val, float max_val) {
    Tensor result = ops::clamp(a, min_val, max_val);
    
    // Clamp has subgradient at boundaries
    // For simplicity, we use a straight-through estimator
    auto backward_fn = [a, min_val, max_val](const Tensor& grad) -> std::vector<Tensor> {
        Tensor grad_a(a.shape(), a.dtype(), a.device());
        const float* a_data = a.data_ptr<float>();
        const float* grad_data = grad.data_ptr<float>();
        float* grad_a_data = grad_a.data_ptr<float>();
        
        for (size_t i = 0; i < a.size(); ++i) {
            if (a_data[i] < min_val || a_data[i] > max_val) {
                grad_a_data[i] = 0.0f;  // No gradient outside clamp range
            } else {
                grad_a_data[i] = grad_data[i];
            }
        }
        
        return {grad_a};
    };
    
    GradientTape::instance().record({a}, result, backward_fn, "clamp");
    
    if (a.requires_grad()) {
        result.set_requires_grad(true);
    }
    
    return result;
}

Tensor ones(const Shape& shape, DType dtype, Device device) {
    return jin::ones(shape, dtype, device);
}

Tensor zeros(const Shape& shape, DType dtype, Device device) {
    return jin::zeros(shape, dtype, device);
}

} // namespace autodiff
} // namespace jin
