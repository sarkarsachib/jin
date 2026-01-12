#pragma once

#include "tensor.h"

namespace jin {
namespace ops {

// Binary operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);

// Matrix operations
Tensor matmul(const Tensor& a, const Tensor& b);

// Unary operations
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor relu(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor neg(const Tensor& a);

// Reduction operations
Tensor sum(const Tensor& a, int axis = -1, bool keepdims = false);
Tensor mean(const Tensor& a, int axis = -1, bool keepdims = false);

// Shape operations (these are in tensor.h but we declare them here for consistency)
inline Tensor reshape(const Tensor& a, const Shape& new_shape) {
    return a.reshape(new_shape);
}

inline Tensor transpose(const Tensor& a, const std::vector<int>& axes = {}) {
    return a.transpose(axes);
}

// Broadcasting helper
void broadcast_binary_op(const Tensor& a, const Tensor& b, Tensor& out,
                         void (*op)(const float*, const float*, float*, size_t));

} // namespace ops
} // namespace jin
