#include "../src/temporal/consistency_loss.h"
#include "../src/temporal/frame_state.h"
#include "../src/core/tensor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <limits>

// Simple assertion macro for testing
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

void test_consistency_loss_smoothness() {
    std::cout << "Testing consistency_loss smoothness..." << std::endl;
    
    // Create frame features with smooth transition
    std::vector<jin::Tensor> frame_features;
    for (int i = 0; i < 3; ++i) {
        jin::Tensor feature({2});
        feature.fill(static_cast<float>(i));
        frame_features.push_back(feature);
    }
    
    jin::Tensor smoothness = jin::temporal::smoothness_loss(frame_features, 0.1f);
    
    ASSERT(smoothness.size() > 0, "Smoothness loss should produce output");
    float loss_value = smoothness.item();
    ASSERT(loss_value >= 0.0f, "Smoothness loss should be non-negative");
    
    // Smooth sequence should have low loss
    ASSERT(loss_value < 1.0f, "Smooth sequence should have low smoothness loss");
    
    std::cout << "✓ Consistency loss smoothness passed" << std::endl;
}

void test_consistency_loss_drift() {
    std::cout << "Testing consistency_loss drift..." << std::endl;
    
    // Create frame latents with drift
    std::vector<jin::Tensor> frame_latents;
    for (int i = 0; i < 3; ++i) {
        jin::Tensor latent({2});
        latent.fill(static_cast<float>(i));  // Gradual drift
        frame_latents.push_back(latent);
    }
    
    // Reference latent (no drift)
    jin::Tensor reference({2});
    reference.fill(0.0f);
    
    jin::Tensor drift = jin::temporal::drift_loss(frame_latents, reference, 0.05f);
    
    ASSERT(drift.size() > 0, "Drift loss should produce output");
    float loss_value = drift.item();
    ASSERT(loss_value >= 0.0f, "Drift loss should be non-negative");
    
    std::cout << "✓ Consistency loss drift passed" << std::endl;
}

void test_consistency_loss_motion_continuity() {
    std::cout << "Testing consistency_loss motion continuity..." << std::endl;
    
    // Create motion codes with continuity
    std::vector<jin::Tensor> motion_codes;
    for (int i = 0; i < 4; ++i) {
        jin::Tensor code({3});
        code.fill(static_cast<float>(i) * 0.5f);  // Gradual change
        motion_codes.push_back(code);
    }
    
    jin::Tensor continuity = jin::temporal::motion_continuity_loss(motion_codes, 0.1f);
    
    ASSERT(continuity.size() > 0, "Motion continuity loss should produce output");
    float loss_value = continuity.item();
    ASSERT(loss_value >= 0.0f, "Motion continuity loss should be non-negative");
    
    std::cout << "✓ Consistency loss motion continuity passed" << std::endl;
}

void test_consistency_loss_scene_stability() {
    std::cout << "Testing consistency_loss scene stability..." << std::endl;
    
    // Create scene states with stability
    std::vector<jin::Tensor> scene_states;
    for (int i = 0; i < 3; ++i) {
        jin::Tensor scene({2});
        scene.fill(1.0f);  // Stable scene
        scene_states.push_back(scene);
    }
    
    jin::Tensor scene_loss = jin::temporal::scene_loss(scene_states, 0.05f);
    
    ASSERT(scene_loss.size() > 0, "Scene stability loss should produce output");
    float loss_value = scene_loss.item();
    ASSERT(loss_value >= 0.0f, "Scene stability loss should be non-negative");
    
    // Stable scene should have low loss
    ASSERT(loss_value < 0.1f, "Stable scene should have low scene loss");
    
    std::cout << "✓ Consistency loss scene stability passed" << std::endl;
}

void test_consistency_loss_identity_consistency() {
    std::cout << "Testing consistency_loss identity consistency..." << std::endl;
    
    // Create FrameStates with identity
    std::vector<jin::temporal::FrameState> frame_states;
    
    for (int i = 0; i < 3; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        
        // Character latent that drifts slightly
        jin::Tensor char_latent({2});
        char_latent.fill(1.0f + static_cast<float>(i) * 0.1f);
        state.character_latent = char_latent;
        
        frame_states.push_back(state);
    }
    
    // Reference character latent
    jin::Tensor reference({2});
    reference.fill(1.0f);
    
    jin::Tensor identity_loss = jin::temporal::identity_consistency_loss(
        frame_states, reference, 0.15f);
    
    ASSERT(identity_loss.size() > 0, "Identity consistency loss should produce output");
    float loss_value = identity_loss.item();
    ASSERT(loss_value >= 0.0f, "Identity consistency loss should be non-negative");
    
    std::cout << "✓ Consistency loss identity consistency passed" << std::endl;
}

void test_consistency_loss_attention_consistency() {
    std::cout << "Testing consistency_loss attention consistency..." << std::endl;
    
    // Create attention weights that change gradually
    std::vector<std::vector<float>> attention_weights;
    
    for (int i = 0; i < 3; ++i) {
        std::vector<float> weights = {0.3f + i * 0.1f, 0.4f + i * 0.05f, 0.3f - i * 0.15f};
        attention_weights.push_back(weights);
    }
    
    jin::Tensor attention_loss = jin::temporal::attention_consistency_loss(
        attention_weights, 0.05f);
    
    ASSERT(attention_loss.size() > 0, "Attention consistency loss should produce output");
    float loss_value = attention_loss.item();
    ASSERT(loss_value >= 0.0f, "Attention consistency loss should be non-negative");
    
    std::cout << "✓ Consistency loss attention consistency passed" << std::endl;
}

void test_consistency_loss_combined() {
    std::cout << "Testing combined temporal loss..." << std::endl;
    
    // Create frame states
    std::vector<jin::temporal::FrameState> frames;
    
    for (int i = 0; i < 3; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        
        // Add character latent
        jin::Tensor char_latent({2});
        char_latent.fill(1.0f);
        state.character_latent = char_latent;
        
        // Add motion code
        jin::Tensor motion_code({2});
        motion_code.fill(static_cast<float>(i) * 0.2f);
        state.motion_code = motion_code;
        
        // Add scene latent
        jin::Tensor scene_latent({2});
        scene_latent.fill(0.8f);
        state.scene_latent = scene_latent;
        
        frames.push_back(state);
    }
    
    // Reference character latent
    jin::Tensor reference({2});
    reference.fill(1.0f);
    
    // Empty attention weights
    std::vector<std::vector<float>> empty_attention;
    
    auto losses = jin::temporal::compute_temporal_loss(
        frames, reference, empty_attention);
    
    // Check all loss components
    ASSERT(losses.smoothness.size() > 0, "Smoothness loss should be computed");
    ASSERT(losses.drift.size() > 0, "Drift loss should be computed");
    ASSERT(losses.motion_continuity.size() > 0, "Motion continuity loss should be computed");
    ASSERT(losses.scene_stability.size() > 0, "Scene stability loss should be computed");
    ASSERT(losses.identity_consistency.size() > 0, "Identity consistency loss should be computed");
    ASSERT(losses.attention_consistency.size() > 0, "Attention consistency loss should be computed");
    ASSERT(losses.total.size() > 0, "Total loss should be computed");
    
    // Check that total is sum of components
    float component_sum = losses.smoothness.item() + losses.drift.item() + 
                         losses.motion_continuity.item() + losses.scene_stability.item() +
                         losses.identity_consistency.item() + losses.attention_consistency.item();
    float total_loss = losses.total.item();
    
    ASSERT(std::abs(component_sum - total_loss) < 1e-5f, 
           "Total loss should be sum of components");
    
    std::cout << "✓ Combined temporal loss passed" << std::endl;
}

void test_consistency_loss_numerical_stability() {
    std::cout << "Testing consistency_loss numerical stability..." << std::endl;
    
    // Test with empty sequences
    std::vector<jin::Tensor> empty_frames;
    std::vector<jin::Tensor> empty_motion;
    std::vector<std::vector<float>> empty_attention;
    
    jin::Tensor empty_smoothness = jin::temporal::smoothness_loss(empty_frames);
    jin::Tensor empty_drift = jin::temporal::drift_loss(empty_frames, jin::Tensor({2}));
    jin::Tensor empty_continuity = jin::temporal::motion_continuity_loss(empty_motion);
    jin::Tensor empty_scene = jin::temporal::scene_loss(empty_frames);
    
    // Should handle empty sequences gracefully (return zeros)
    ASSERT(empty_smoothness.item() == 0.0f, "Empty smoothness should be zero");
    ASSERT(empty_drift.item() == 0.0f, "Empty drift should be zero");
    ASSERT(empty_continuity.item() == 0.0f, "Empty continuity should be zero");
    ASSERT(empty_scene.item() == 0.0f, "Empty scene should be zero");
    
    // Test with single frame
    std::vector<jin::Tensor> single_frame = {jin::Tensor({2})};
    single_frame[0].fill(1.0f);
    
    jin::Tensor single_smoothness = jin::temporal::smoothness_loss(single_frame);
    ASSERT(single_smoothness.item() == 0.0f, "Single frame smoothness should be zero");
    
    // Test nan/inf detection
    jin::Tensor test_tensor({2});
    test_tensor.data_ptr<float>()[0] = 1.0f;
    test_tensor.data_ptr<float>()[1] = std::numeric_limits<float>::infinity();
    
    ASSERT(jin::temporal::has_nan_or_inf(test_tensor), 
           "Should detect infinity in tensor");
    
    test_tensor.data_ptr<float>()[1] = std::numeric_limits<float>::quiet_NaN();
    ASSERT(jin::temporal::has_nan_or_inf(test_tensor), 
           "Should detect NaN in tensor");
    
    std::cout << "✓ Consistency loss numerical stability passed" << std::endl;
}

void test_consistency_loss_normalization() {
    std::cout << "Testing consistency_loss normalization..." << std::endl;
    
    // Create a loss with very large values
    jin::temporal::TemporalLosses large_losses;
    large_losses.smoothness.fill(100.0f);
    large_losses.drift.fill(200.0f);
    large_losses.motion_continuity.fill(150.0f);
    large_losses.scene_stability.fill(50.0f);
    large_losses.identity_consistency.fill(300.0f);
    large_losses.attention_consistency.fill(25.0f);
    large_losses.total.fill(825.0f);
    
    auto normalized = jin::temporal::normalize_losses(large_losses);
    
    // All normalized losses should be clamped to reasonable range
    ASSERT(normalized.smoothness.item() <= 10.0f, "Normalized smoothness should be clamped");
    ASSERT(normalized.drift.item() <= 10.0f, "Normalized drift should be clamped");
    ASSERT(normalized.motion_continuity.item() <= 10.0f, "Normalized motion continuity should be clamped");
    ASSERT(normalized.scene_stability.item() <= 10.0f, "Normalized scene stability should be clamped");
    ASSERT(normalized.identity_consistency.item() <= 10.0f, "Normalized identity consistency should be clamped");
    ASSERT(normalized.attention_consistency.item() <= 10.0f, "Normalized attention consistency should be clamped");
    
    // All should be non-negative
    ASSERT(normalized.smoothness.item() >= 0.0f, "Normalized losses should be non-negative");
    ASSERT(normalized.drift.item() >= 0.0f, "Normalized losses should be non-negative");
    ASSERT(normalized.motion_continuity.item() >= 0.0f, "Normalized losses should be non-negative");
    ASSERT(normalized.scene_stability.item() >= 0.0f, "Normalized losses should be non-negative");
    ASSERT(normalized.identity_consistency.item() >= 0.0f, "Normalized losses should be non-negative");
    ASSERT(normalized.attention_consistency.item() >= 0.0f, "Normalized losses should be non-negative");
    
    std::cout << "✓ Consistency loss normalization passed" << std::endl;
}

void run_consistency_loss_tests() {
    std::cout << "\n=== Running Consistency Loss Tests ===" << std::endl;
    
    test_consistency_loss_smoothness();
    test_consistency_loss_drift();
    test_consistency_loss_motion_continuity();
    test_consistency_loss_scene_stability();
    test_consistency_loss_identity_consistency();
    test_consistency_loss_attention_consistency();
    test_consistency_loss_combined();
    test_consistency_loss_numerical_stability();
    test_consistency_loss_normalization();
    
    std::cout << "✓ All Consistency Loss tests passed!" << std::endl;
}