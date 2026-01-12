#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"

namespace jin {
namespace autodiff {

// Autodiff-enabled operations
Tensor add(const Tensor& a, const Tensor& b);
Tensor multiply(const Tensor& a, const Tensor& b);
Tensor subtract(const Tensor& a, const Tensor& b);
Tensor divide(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor exp(const Tensor& a);
Tensor log(const Tensor& a);
Tensor relu(const Tensor& a);
Tensor tanh(const Tensor& a);
Tensor sigmoid(const Tensor& a);
Tensor neg(const Tensor& a);
Tensor sum(const Tensor& a, int axis = -1, bool keepdims = false);
Tensor mean(const Tensor& a, int axis = -1, bool keepdims = false);

} // namespace autodiff
} // namespace jin
