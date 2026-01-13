#pragma once

#include "../core/tensor.h"
#include "linear.h"
#include "dropout.h"

namespace jin {
namespace nn {

class MultiHeadAttention {
public:
    MultiHeadAttention(size_t d_model, size_t num_heads, size_t d_k = 0, size_t d_v = 0,
                       float dropout = 0.0f);
    
    // Self-attention: Q, K, V all from the same source
    Tensor forward(const Tensor& x);
    
    // Cross-attention: separate Q, K, V
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v);
    
    // Full attention with mask support
    Tensor forward(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& mask = Tensor());
    
    // Properties
    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    
    void train() { mode_ = ForwardMode::Train; dropout_.train(); }
    void eval() { mode_ = ForwardMode::Eval; dropout_.eval(); }
    bool is_training() const { return mode_ == ForwardMode::Train; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;
    size_t d_k_;
    size_t d_v_;
    
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear out_proj_;
    Dropout dropout_;
    ForwardMode mode_;
};

// Positional Encoding
Tensor sinusoidal_positional_encoding(size_t seq_len, size_t d_model);
Tensor relative_positional_encoding(size_t seq_len, size_t d_model);

// rotary positional encoding (RoPE)
void apply_rotary_embeddings(Tensor& q, Tensor& k, const Tensor& positions);

} // namespace nn
} // namespace jin
