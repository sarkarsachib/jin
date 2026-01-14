#include "attention.h"
#include "../autodiff/ops.h"
#include "../autodiff/tape.h"
#include "../core/math_ops.h"
#include "activation.h"
#include <cmath>

namespace jin {
namespace nn {

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads, size_t d_k, size_t d_v,
                                       float dropout)
    : d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(d_model / num_heads),
      d_k_(d_k == 0 ? head_dim_ : d_k),
      d_v_(d_v == 0 ? head_dim_ : d_v),
      q_proj_(d_model, d_model),
      k_proj_(d_model, d_model),
      v_proj_(d_model, d_model),
      out_proj_(d_model, d_model),
      dropout_(dropout),
      mode_(ForwardMode::Train) {
    
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    return forward(x, x, x, Tensor());
}

Tensor MultiHeadAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v) {
    return forward(q, k, v, Tensor());
}

Tensor MultiHeadAttention::forward(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& mask) {
    autodiff::TapeGuard tape_guard;
    
    Shape q_shape = q.shape();
    Shape k_shape = k.shape();
    Shape v_shape = v.shape();
    
    size_t batch_size = q_shape[0];
    size_t q_len = q_shape[1];
    size_t k_len = k_shape.size() > 1 ? k_shape[1] : 1;
    size_t v_len = v_shape.size() > 1 ? v_shape[1] : 1;
    
    // Project Q, K, V
    Tensor Q = q_proj_.forward(q);
    Tensor K = k_proj_.forward(k);
    Tensor V = v_proj_.forward(v);
    
    // Reshape for multi-head attention
    size_t new_batch = batch_size * num_heads_;
    
    // Final output accumulator
    Shape out_shape = {batch_size, q_len, d_model_};
    Tensor output(out_shape, DType::FP32, Device::cpu());
    output.set_requires_grad(true);
    
    // Process each head (sequential for now)
    for (size_t h = 0; h < num_heads_; ++h) {
        Tensor Q_h = Q.reshape({batch_size * num_heads_, q_len, head_dim_});
        Tensor K_h = K.reshape({batch_size * num_heads_, k_len, head_dim_});
        Tensor V_h = V.reshape({batch_size * num_heads_, v_len, head_dim_});
        
        // Compute attention scores
        Tensor K_h_t = K_h.transpose({0, 2, 1});
        Tensor scores = ops::matmul(Q_h, K_h_t);
        
        // Scale
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        Tensor scale_t = jin::ones({1}, DType::FP32, Device::cpu());
        scale_t.data_ptr<float>()[0] = scale;
        Tensor scores_scaled = ops::multiply(scores, scale_t);
        
        // Softmax
        Tensor attn = nn::softmax(scores_scaled, -1);
        
        // Apply dropout during training
        if (mode_ == ForwardMode::Train) {
            attn = dropout_.forward(attn);
        }
        
        // Apply attention to V
        Tensor attn_v = ops::matmul(attn, V_h);
        
        // Store in output
        const float* attn_v_data = attn_v.data_ptr<float>();
        float* out_data = output.data_ptr<float>();
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t q = 0; q < q_len; ++q) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    out_data[(b * q_len + q) * d_model_ + h * head_dim_ + d] =
                        attn_v_data[(b * q_len + q) * head_dim_ + d];
                }
            }
        }
    }
    
    // Output projection
    output = out_proj_.forward(output);
    
    // Apply dropout during training
    if (mode_ == ForwardMode::Train) {
        output = dropout_.forward(output);
    }
    
    return output;
}

// ==================== Positional Encoding ====================

Tensor sinusoidal_positional_encoding(size_t seq_len, size_t d_model) {
    Tensor pe(Shape{seq_len, d_model}, DType::FP32, Device::cpu());
    float* data = pe.data_ptr<float>();
    
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
            float angle = static_cast<float>(pos) / std::pow(10000.0f, static_cast<float>(i % (d_model / 2)) * 2.0f / static_cast<float>(d_model));
            if (i % 2 == 0) {
                data[pos * d_model + i] = std::sin(angle);
            } else {
                data[pos * d_model + i] = std::cos(angle);
            }
        }
    }
    
    return pe;
}

Tensor relative_positional_encoding(size_t seq_len, size_t d_model) {
    Tensor pe(Shape{seq_len, d_model}, DType::FP32, Device::cpu());
    float* data = pe.data_ptr<float>();
    
    size_t half = d_model / 2;
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            int rel_pos = static_cast<int>(j) - static_cast<int>(i);
            float angle = static_cast<float>(rel_pos) / std::pow(10000.0f, static_cast<float>(i % half) * 2.0f / static_cast<float>(d_model));
            if (i % 2 == 0) {
                data[i * d_model + j] = std::sin(angle);
            } else {
                data[i * d_model + j] = std::cos(angle);
            }
        }
    }
    
    return pe;
}

void apply_rotary_embeddings(Tensor& q, Tensor& k, const Tensor& positions) {
    Shape q_shape = q.shape();
    size_t d_model = q_shape[q_shape.size() - 1];
    size_t half = d_model / 2;
    
    const float* pos_data = positions.data_ptr<float>();
    const float* q_data = q.data_ptr<float>();
    const float* k_data = k.data_ptr<float>();
    float* q_out = q.data_ptr<float>();
    float* k_out = k.data_ptr<float>();
    
    size_t n_elements = 1;
    for (size_t s : q_shape) n_elements *= s;
    size_t seq_len = n_elements / d_model;
    
    for (size_t seq = 0; seq < seq_len; ++seq) {
        float theta = pos_data[seq];
        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);
        
        for (size_t d = 0; d < half; ++d) {
            size_t idx = seq * d_model + d;
            size_t idx_rot = seq * d_model + d + half;
            
            float q_rot = q_data[idx_rot];
            q_out[idx] = q_data[idx] * cos_theta - q_rot * sin_theta;
            q_out[idx_rot] = q_data[idx] * sin_theta + q_rot * cos_theta;
            
            float k_rot = k_data[idx_rot];
            k_out[idx] = k_data[idx] * cos_theta - k_rot * sin_theta;
            k_out[idx_rot] = k_data[idx] * sin_theta + k_rot * cos_theta;
        }
    }
}

} // namespace nn
} // namespace jin
