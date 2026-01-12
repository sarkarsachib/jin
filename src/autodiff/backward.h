#pragma once

#include "../core/tensor.h"

namespace jin {
namespace autodiff {

void backward(Tensor& output, const Tensor* grad_output = nullptr);

void zero_grad(Tensor& tensor);

} // namespace autodiff
} // namespace jin
