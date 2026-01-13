#pragma once

#include "../core/tensor.h"
#include <string>

namespace jin {
namespace nn {

// Forward declarations
class Linear;
class Conv1d;
class Conv2d;
class LayerNorm;
class RMSNorm;
class Dropout;
class Embedding;
class MultiHeadAttention;
class Pooling;

// ==================== Initialization ====================

struct Initializer {
    enum Type {
        Default,
        XavierUniform,
        XavierNormal,
        KaimingUniform,
        KaimingNormal,
        Normal,
        Zeros,
        Ones
    };
    
    Type type;
    float param1;  // std for Normal, bound for Uniform
    float param2;  // Not used for Normal, unused for Uniform
    
    static Initializer default_init() { return {Default, 0.0f, 0.0f}; }
    static Initializer xavier_uniform(float gain = 1.0f) { return {XavierUniform, gain, 0.0f}; }
    static Initializer xavier_normal(float gain = 1.0f) { return {XavierNormal, gain, 0.0f}; }
    static Initializer kaiming_uniform(float scale = 2.0f) { return {KaimingUniform, scale, 0.0f}; }
    static Initializer kaiming_normal(float scale = 2.0f) { return {KaimingNormal, scale, 0.0f}; }
    static Initializer normal(float mean = 0.0f, float std = 1.0f) { return {Normal, mean, std}; }
    static Initializer zeros() { return {Zeros, 0.0f, 0.0f}; }
    static Initializer ones() { return {Ones, 0.0f, 0.0f}; }
};

Tensor initialize_tensor(const Shape& shape, Initializer init);

// ==================== Forward Mode ====================

enum class ForwardMode {
    Train,
    Eval
};

class ForwardModeGuard {
public:
    ForwardModeGuard(ForwardMode mode);
    ~ForwardModeGuard();
    
    static ForwardMode get() { return mode_; }
    static void set(ForwardMode mode) { mode_ = mode; }
    
private:
    ForwardMode prev_mode_;
    static ForwardMode mode_;
};

// ==================== Common Utilities ====================

Shape compute_output_shape(Shape input, size_t dim, int padding, int stride, int dilation, int kernel_size);

} // namespace nn
} // namespace jin
