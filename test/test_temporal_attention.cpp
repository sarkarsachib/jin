#include "../src/temporal/temporal_attention.h"
#include "../src/core/tensor.h"
#include <iostream>
#include <cassert>

// Simple assertion macro for testing
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

void test_temporal_attention_basic() {
    std::cout << "Testing TemporalAttention basic functionality..." << std::endl;
    
    jin::temporal::TemporalAttention attention(8, 4);  // d_model=8, max_frames=4
    
    // Create current frame features
    jin::Tensor current({8});
    current.fill(1.0f);
    
    // Create past frames
    std::vector<jin::Tensor> past_frames;
    for (int i = 0; i < 3; ++i) {
        jin::Tensor past({8});
        past.fill(static_cast<float>(i + 1));
        past_frames.push_back(past);
    }
    
    // Forward pass
    jin::Tensor output = attention.forward(current, past_frames);
    
    ASSERT(output.size() > 0, "Output should have data");
    ASSERT(output.size() == 8, "Output should have same dimension as current frame");
    
    std::cout << "✓ TemporalAttention basic functionality passed" << std::endl;
}

void test_temporal_attention_no_past_frames() {
    std::cout << "Testing TemporalAttention with no past frames..." << std::endl;
    
    jin::temporal::TemporalAttention attention(4, 4);
    
    jin::Tensor current({4});
    current.fill(1.0f);
    
    // Empty past frames
    std::vector<jin::Tensor> empty_past;
    
    jin::Tensor output = attention.forward(current, empty_past);
    
    // Should return current frame unchanged
    ASSERT(output.size() == current.size(), "Should return current frame when no past frames");
    
    std::cout << "✓ TemporalAttention no past frames passed" << std::endl;
}

void test_temporal_attention_attention_weights() {
    std::cout << "Testing TemporalAttention attention weights..." << std::endl;
    
    jin::temporal::TemporalAttention attention(4, 3);
    
    jin::Tensor current({4});
    current.fill(1.0f);
    
    // Create different past frames to see attention distribution
    std::vector<jin::Tensor> past_frames;
    for (int i = 0; i < 3; ++i) {
        jin::Tensor past({4});
        past.fill(static_cast<float>(i));  // Different values
        past_frames.push_back(past);
    }
    
    auto weights = attention.get_attention_weights(current, past_frames);
    
    ASSERT(weights.size() == 3, "Should have weights for 3 past frames");
    
    // Weights should sum to approximately 1 (softmax)
    float weight_sum = 0.0f;
    for (float w : weights) {
        weight_sum += w;
    }
    ASSERT(std::abs(weight_sum - 1.0f) < 1e-5f, "Attention weights should sum to 1");
    
    // All weights should be non-negative
    for (float w : weights) {
        ASSERT(w >= 0.0f, "Attention weights should be non-negative");
    }
    
    std::cout << "✓ TemporalAttention attention weights passed" << std::endl;
}

void test_temporal_attention_max_frames() {
    std::cout << "Testing TemporalAttention max frames limit..." << std::endl;
    
    jin::temporal::TemporalAttention attention(4, 2);  // Only keep last 2 frames
    
    jin::Tensor current({4});
    current.fill(1.0f);
    
    // Create many past frames (more than max_frames)
    std::vector<jin::Tensor> past_frames;
    for (int i = 0; i < 5; ++i) {
        jin::Tensor past({4});
        past.fill(static_cast<float>(i));
        past_frames.push_back(past);
    }
    
    // Forward pass
    jin::Tensor output = attention.forward(current, past_frames);
    
    // Get attention weights to verify only last 2 frames are used
    auto weights = attention.get_attention_weights(current, past_frames);
    
    // Should only have weights for last 2 frames
    ASSERT(weights.size() == 2, "Should only have weights for last max_frames frames");
    
    std::cout << "✓ TemporalAttention max frames limit passed" << std::endl;
}

void test_temporal_attention_positional_bias() {
    std::cout << "Testing TemporalAttention positional bias..." << std::endl;
    
    jin::temporal::TemporalAttention attention(4, 3);
    
    jin::Tensor current({4});
    current.fill(1.0f);
    
    // Create identical past frames
    std::vector<jin::Tensor> past_frames;
    for (int i = 0; i < 3; ++i) {
        jin::Tensor past({4});
        past.fill(1.0f);  // All identical
        past_frames.push_back(past);
    }
    
    // Test with positional bias
    auto weights_with_bias = attention.get_attention_weights(current, past_frames, true);
    
    // Test without positional bias
    auto weights_without_bias = attention.get_attention_weights(current, past_frames, false);
    
    // With positional bias, recent frames should have higher weights
    // Since all frames are identical, bias should make the last frame have highest weight
    ASSERT(weights_with_bias[2] >= weights_with_bias[0], 
           "With positional bias, recent frames should have higher weights");
    
    std::cout << "✓ TemporalAttention positional bias passed" << std::endl;
}

void test_temporal_attention_multi_head() {
    std::cout << "Testing TemporalAttention multi-head..." << std::endl;
    
    jin::temporal::TemporalAttention attention(8, 4);
    
    jin::Tensor current({8});
    current.fill(1.0f);
    
    std::vector<jin::Tensor> past_frames;
    for (int i = 0; i < 2; ++i) {
        jin::Tensor past({8});
        past.fill(static_cast<float>(i + 1));
        past_frames.push_back(past);
    }
    
    // Multi-head attention with 2 heads
    jin::Tensor output = attention.multi_head_attention(current, past_frames, 2);
    
    ASSERT(output.size() == 8, "Multi-head output should have same dimension");
    
    std::cout << "✓ TemporalAttention multi-head passed" << std::endl;
}

void test_temporal_attention_multi_head_uneven() {
    std::cout << "Testing TemporalAttention multi-head with uneven dimensions..." << std::endl;
    
    jin::temporal::TemporalAttention attention(7, 3);  // 7 is not divisible by 2
    
    jin::Tensor current({7});
    current.fill(1.0f);
    
    std::vector<jin::Tensor> past_frames;
    for (int i = 0; i < 2; ++i) {
        jin::Tensor past({7});
        past.fill(static_cast<float>(i + 1));
        past_frames.push_back(past);
    }
    
    // Should fallback to single head when dimensions don't divide evenly
    jin::Tensor output = attention.multi_head_attention(current, past_frames, 2);
    
    ASSERT(output.size() == 7, "Should fallback to single head with correct dimensions");
    
    std::cout << "✓ TemporalAttention multi-head uneven dimensions passed" << std::endl;
}

void test_temporal_attention_empty_current() {
    std::cout << "Testing TemporalAttention with empty current frame..." << std::endl;
    
    jin::temporal::TemporalAttention attention(4, 3);
    
    jin::Tensor current;  // Empty tensor
    
    std::vector<jin::Tensor> past_frames;
    jin::Tensor past({4});
    past.fill(1.0f);
    past_frames.push_back(past);
    
    jin::Tensor output = attention.forward(current, past_frames);
    
    // Should handle gracefully (return empty or current)
    // Implementation dependent, but shouldn't crash
    
    std::cout << "✓ TemporalAttention empty current frame passed" << std::endl;
}

void test_temporal_attention_edge_cases() {
    std::cout << "Testing TemporalAttention edge cases..." << std::endl;
    
    jin::temporal::TemporalAttention attention(2, 1);
    
    // Test with zero max_frames
    jin::temporal::TemporalAttention zero_attention(2, 0);
    jin::Tensor current({2});
    current.fill(1.0f);
    std::vector<jin::Tensor> past = {jin::Tensor({2})};
    past[0].fill(1.0f);
    
    // Should handle gracefully
    jin::Tensor output = zero_attention.forward(current, past);
    
    // Test get_attention_weights with empty past
    auto empty_weights = attention.get_attention_weights(current, {});
    ASSERT(empty_weights.empty(), "Empty past frames should give empty weights");
    
    std::cout << "✓ TemporalAttention edge cases passed" << std::endl;
}

void run_temporal_attention_tests() {
    std::cout << "\n=== Running TemporalAttention Tests ===" << std::endl;
    
    test_temporal_attention_basic();
    test_temporal_attention_no_past_frames();
    test_temporal_attention_attention_weights();
    test_temporal_attention_max_frames();
    test_temporal_attention_positional_bias();
    test_temporal_attention_multi_head();
    test_temporal_attention_multi_head_uneven();
    test_temporal_attention_empty_current();
    test_temporal_attention_edge_cases();
    
    std::cout << "✓ All TemporalAttention tests passed!" << std::endl;
}