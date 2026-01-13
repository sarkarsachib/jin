#pragma once

#include "../core/tensor.h"
#include "../core/math_ops.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace jin {
namespace temporal {

class TemporalAttention {
public:
    TemporalAttention(int d_model, int max_frames = 16) 
        : d_model_(d_model), max_frames_(max_frames), dropout_(0.1f) {}
    
    // Forward pass with recency bias
    Tensor forward(
        const Tensor& current_frame_feat,  // [d_model] current frame features
        const std::vector<Tensor>& past_frames, // [num_past, d_model]
        bool use_positional_bias = true    // Closer frames = higher weight
    ) {
        if (past_frames.empty()) {
            return current_frame_feat;  // No attention if no past frames
        }
        
        int num_past = std::min(static_cast<int>(past_frames.size()), max_frames_);
        if (num_past == 0) {
            return current_frame_feat;
        }
        
        // Keep only last K frames
        int start_idx = past_frames.size() - num_past;
        std::vector<Tensor> recent_past(past_frames.begin() + start_idx, past_frames.end());
        
        // 1. Compute attention scores with recency bias
        Tensor attention_scores = compute_attention_scores(current_frame_feat, recent_past, use_positional_bias);
        
        // 2. Apply softmax to get attention weights
        Tensor attention_weights = softmax(attention_scores);
        
        // 3. Aggregate: sum(attention_weights * past_frame_features)
        Tensor attended_features = aggregate_features(recent_past, attention_weights);
        
        // 4. Fuse with current frame via gating
        Tensor fused_features = fuse_with_current(current_frame_feat, attended_features);
        
        return fused_features;
    }
    
    // Multi-head temporal attention
    Tensor multi_head_attention(
        const Tensor& current_frame_feat,
        const std::vector<Tensor>& past_frames,
        int num_heads = 8,
        bool use_positional_bias = true
    ) {
        if (past_frames.empty()) {
            return current_frame_feat;
        }
        
        int head_dim = d_model_ / num_heads;
        if (head_dim * num_heads != d_model_) {
            // Fallback to single head if dimensions don't divide evenly
            return forward(current_frame_feat, past_frames, use_positional_bias);
        }
        
        std::vector<Tensor> head_outputs;
        
        for (int h = 0; h < num_heads; ++h) {
            // Extract head features (simplified - would need proper projection in real implementation)
            std::vector<Tensor> head_past;
            for (const auto& frame : past_frames) {
                // For simplicity, use same features for all heads
                // In practice, would project to different subspaces
                head_past.push_back(frame);
            }
            
            Tensor head_output = forward(current_frame_feat, head_past, use_positional_bias);
            head_outputs.push_back(head_output);
        }
        
        // Concatenate heads
        return concatenate(head_outputs);
    }
    
    // Get attention weights for analysis/debugging
    std::vector<float> get_attention_weights(
        const Tensor& current_frame_feat,
        const std::vector<Tensor>& past_frames,
        bool use_positional_bias = true
    ) {
        if (past_frames.empty()) return {};
        
        int num_past = std::min(static_cast<int>(past_frames.size()), max_frames_);
        int start_idx = past_frames.size() - num_past;
        std::vector<Tensor> recent_past(past_frames.begin() + start_idx, past_frames.end());
        
        Tensor attention_scores = compute_attention_scores(current_frame_feat, recent_past, use_positional_bias);
        Tensor attention_weights = softmax(attention_scores);
        
        std::vector<float> weights;
        const float* weight_data = attention_weights.data_ptr<float>();
        for (int i = 0; i < attention_weights.size(); ++i) {
            weights.push_back(weight_data[i]);
        }
        
        return weights;
    }

private:
    int d_model_;        // Model dimension
    int max_frames_;     // Maximum frames to attend to
    float dropout_;      // Dropout rate
    
    // Compute attention scores with recency bias
    Tensor compute_attention_scores(
        const Tensor& current_frame_feat,
        const std::vector<Tensor>& past_frames,
        bool use_positional_bias
    ) {
        int num_past = past_frames.size();
        Tensor scores({num_past});
        
        for (int i = 0; i < num_past; ++i) {
            // Simple dot product attention
            float score = dot_product(current_frame_feat, past_frames[i]);
            scores.data_ptr<float>()[i] = score;
        }
        
        if (use_positional_bias) {
            // Apply recency bias: recent frames get higher weight
            for (int i = 0; i < num_past; ++i) {
                float recency_bias = static_cast<float>(i + 1) / num_past;  // 0 to 1
                scores.data_ptr<float>()[i] *= recency_bias;
            }
        }
        
        return scores;
    }
    
    // Compute dot product between two vectors
    float dot_product(const Tensor& a, const Tensor& b) {
        if (a.size() != b.size() || a.size() == 0) return 0.0f;
        
        Tensor elementwise_prod = a * b;
        return jin::ops::sum(elementwise_prod).item();
    }
    
    // Apply softmax for numerical stability
    Tensor softmax(const Tensor& scores) {
        if (scores.size() == 0) return scores;
        
        // Find max for numerical stability
        const float* scores_data = scores.data_ptr<float>();
        float max_score = scores_data[0];
        for (size_t i = 1; i < scores.size(); ++i) {
            max_score = std::max(max_score, scores_data[i]);
        }
        
        // Compute softmax
        Tensor softmax_scores({static_cast<size_t>(scores.size())});
        float sum_exp = 0.0f;
        
        for (size_t i = 0; i < scores.size(); ++i) {
            float exp_score = std::exp(scores_data[i] - max_score);
            softmax_scores.data_ptr<float>()[i] = exp_score;
            sum_exp += exp_score;
        }
        
        // Normalize
        if (sum_exp > 0) {
            for (size_t i = 0; i < softmax_scores.size(); ++i) {
                softmax_scores.data_ptr<float>()[i] /= sum_exp;
            }
        }
        
        return softmax_scores;
    }
    
    // Aggregate features using attention weights
    Tensor aggregate_features(
        const std::vector<Tensor>& past_frames,
        const Tensor& attention_weights
    ) {
        if (past_frames.size() != attention_weights.size() || past_frames.empty()) {
            return Tensor();
        }
        
        // Weighted sum of past features
        Tensor aggregated = zeros(past_frames[0].shape());
        const float* weights = attention_weights.data_ptr<float>();
        
        for (size_t i = 0; i < past_frames.size(); ++i) {
            Tensor weighted_frame = past_frames[i] * weights[i];
            aggregated = aggregated + weighted_frame;
        }
        
        return aggregated;
    }
    
    // Fuse attended features with current frame
    Tensor fuse_with_current(
        const Tensor& current_frame_feat,
        const Tensor& attended_features
    ) {
        // Simple gating mechanism
        float gate_value = sigmoid(compute_gate_value(current_frame_feat, attended_features));
        
        // Fuse: result = current * gate + attended * (1 - gate)
        Tensor fused = current_frame_feat * gate_value + attended_features * (1.0f - gate_value);
        
        return fused;
    }
    
    // Compute gate value
    float compute_gate_value(
        const Tensor& current_frame_feat,
        const Tensor& attended_features
    ) {
        // Simple dot product to determine how much to attend to past
        return dot_product(current_frame_feat, attended_features);
    }
    
    // Sigmoid activation
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    // Concatenate tensors
    Tensor concatenate(const std::vector<Tensor>& tensors) {
        if (tensors.empty()) return Tensor();
        
        size_t total_size = 0;
        for (const auto& tensor : tensors) {
            total_size += tensor.size();
        }
        
        Tensor result({static_cast<size_t>(total_size)});
        size_t offset = 0;
        
        for (const auto& tensor : tensors) {
            size_t tensor_size = tensor.size() * sizeof(float);
            std::memcpy(
                static_cast<char*>(result.data()) + offset,
                tensor.data(),
                tensor_size
            );
            offset += tensor.size();
        }
        
        return result;
    }
};

} // namespace temporal
} // namespace jin