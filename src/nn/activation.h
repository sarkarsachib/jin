#pragma once

#include "../core/tensor.h"

namespace jin {
namespace nn {

// ==================== Activation Functions ====================

Tensor relu(const Tensor& x);
Tensor relu_backward(const Tensor& grad, const Tensor& x);

Tensor leaky_relu(const Tensor& x, float negative_slope = 0.01f);
Tensor leaky_relu_backward(const Tensor& grad, const Tensor& x, float negative_slope = 0.01f);

Tensor sigmoid(const Tensor& x);
Tensor sigmoid_backward(const Tensor& grad, const Tensor& x);

Tensor tanh(const Tensor& x);
Tensor tanh_backward(const Tensor& grad, const Tensor& x);

Tensor gelu(const Tensor& x, bool approximate = true);
Tensor gelu_backward(const Tensor& grad, const Tensor& x, bool approximate = true);

Tensor silu(const Tensor& x);
Tensor silu_backward(const Tensor& grad, const Tensor& x);

Tensor swish(const Tensor& x, float beta = 1.0f);
Tensor swish_backward(const Tensor& grad, const Tensor& x, float beta = 1.0f);

Tensor mish(const Tensor& x);
Tensor mish_backward(const Tensor& grad, const Tensor& x);

// Softmax with numerical stability
Tensor softmax(const Tensor& x, int axis = -1);
Tensor softmax_backward(const Tensor& grad, const Tensor& x, int axis = -1);

// Log softmax (numerically stable)
Tensor log_softmax(const Tensor& x, int axis = -1);

// ==================== Activation Classes ====================

class ReLU {
public:
    static Tensor forward(const Tensor& x) { return relu(x); }
};

class LeakyReLU {
public:
    LeakyReLU(float negative_slope = 0.01f) : negative_slope_(negative_slope) {}
    Tensor forward(const Tensor& x) { return leaky_relu(x, negative_slope_); }
    
private:
    float negative_slope_;
};

class GELU {
public:
    GELU(bool approximate = true) : approximate_(approximate) {}
    Tensor forward(const Tensor& x) { return gelu(x, approximate_); }
    
private:
    bool approximate_;
};

class SiLU {
public:
    static Tensor forward(const Tensor& x) { return silu(x); }
};

class Swish {
public:
    Swish(float beta = 1.0f) : beta_(beta) {}
    Tensor forward(const Tensor& x) { return swish(x, beta_); }
    
private:
    float beta_;
};

class Mish {
public:
    static Tensor forward(const Tensor& x) { return mish(x); }
};

class Sigmoid {
public:
    static Tensor forward(const Tensor& x) { return sigmoid(x); }
};

class TanhLayer {
public:
    static Tensor forward(const Tensor& x) { return tanh(x); }
};

} // namespace nn
} // namespace jin
