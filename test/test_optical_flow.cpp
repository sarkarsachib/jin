#include "../src/temporal/optical_flow.h"
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

void test_optical_flow_basic() {
    std::cout << "Testing LatentOpticalFlow basic functionality..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Create test latents
    jin::Tensor latent_t({3});
    latent_t.fill(1.0f);
    
    jin::Tensor latent_t1({3});
    latent_t1.fill(1.5f);
    
    // Estimate flow
    jin::Tensor flow_vector = flow.estimate_flow(latent_t, latent_t1);
    
    ASSERT(flow_vector.size() > 0, "Flow vector should have data");
    ASSERT(flow_vector.size() == latent_t.size(), "Flow should have same dimension as input");
    
    // Flow should be (latent_t1 - latent_t) = 0.5
    const float* flow_data = flow_vector.data_ptr<float>();
    for (size_t i = 0; i < flow_vector.size(); ++i) {
        ASSERT(std::abs(flow_data[i] - 0.5f) < 1e-5f, "Flow should be 0.5 (difference between latents)");
    }
    
    std::cout << "✓ LatentOpticalFlow basic functionality passed" << std::endl;
}

void test_optical_flow_frame_state() {
    std::cout << "Testing LatentOpticalFlow with FrameState..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Create FrameStates
    jin::temporal::FrameState state_t, state_t1;
    
    state_t.character_latent = jin::Tensor({2});
    state_t.character_latent.fill(1.0f);
    
    state_t1.character_latent = jin::Tensor({2});
    state_t1.character_latent.fill(2.0f);
    
    // Estimate flow between frame states
    jin::Tensor flow_vector = flow.estimate_flow_frame_state(state_t, state_t1);
    
    ASSERT(flow_vector.size() > 0, "Flow vector should be computed from FrameState");
    ASSERT(flow_vector.size() == 2, "Flow should have same dimension as character latent");
    
    std::cout << "✓ LatentOpticalFlow FrameState processing passed" << std::endl;
}

void test_optical_flow_magnitude() {
    std::cout << "Testing LatentOpticalFlow magnitude computation..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Test zero flow
    jin::Tensor zero_flow({3});
    zero_flow.fill(0.0f);
    float zero_magnitude = flow.flow_magnitude(zero_flow);
    ASSERT(std::abs(zero_magnitude) < 1e-5f, "Zero flow should have zero magnitude");
    
    // Test known flow
    jin::Tensor known_flow({2});
    known_flow.data_ptr<float>()[0] = 3.0f;
    known_flow.data_ptr<float>()[1] = 4.0f;
    float magnitude = flow.flow_magnitude(known_flow);
    
    // Magnitude should be sqrt(3^2 + 4^2) = 5
    ASSERT(std::abs(magnitude - 5.0f) < 1e-5f, "Magnitude should be 5 for flow [3, 4]");
    
    std::cout << "✓ LatentOpticalFlow magnitude computation passed" << std::endl;
}

void test_optical_flow_saccade_detection() {
    std::cout << "Testing LatentOpticalFlow saccade detection..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Create small flow (not saccade)
    jin::Tensor small_flow({2});
    small_flow.fill(0.1f);
    ASSERT(!flow.is_saccade(small_flow, 0.5f), "Small flow should not be detected as saccade");
    
    // Create large flow (saccade)
    jin::Tensor large_flow({2});
    large_flow.fill(1.0f);
    ASSERT(flow.is_saccade(large_flow, 0.5f), "Large flow should be detected as saccade");
    
    // Test with empty flow
    ASSERT(!flow.is_saccade(jin::Tensor(), 0.5f), "Empty flow should not be saccade");
    
    std::cout << "✓ LatentOpticalFlow saccade detection passed" << std::endl;
}

void test_optical_flow_motion_classification() {
    std::cout << "Testing LatentOpticalFlow motion classification..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Test static motion
    jin::Tensor static_flow({2});
    static_flow.fill(0.05f);
    auto static_motion = flow.classify_motion(static_flow);
    ASSERT(static_motion == jin::temporal::LatentOpticalFlow::MotionType::STATIC, 
           "Small flow should be classified as STATIC");
    
    // Test smooth motion
    jin::Tensor smooth_flow({2});
    smooth_flow.fill(0.3f);
    auto smooth_motion = flow.classify_motion(smooth_flow);
    ASSERT(smooth_motion == jin::temporal::LatentOpticalFlow::MotionType::SMOOTH, 
           "Medium flow should be classified as SMOOTH");
    
    // Test saccade motion
    jin::Tensor saccade_flow({2});
    saccade_flow.fill(0.7f);
    auto saccade_motion = flow.classify_motion(saccade_flow);
    ASSERT(saccade_motion == jin::temporal::LatentOpticalFlow::MotionType::SACCADE, 
           "Large flow should be classified as SACCADE");
    
    // Test transition motion
    jin::Tensor transition_flow({2});
    transition_flow.fill(1.5f);
    auto transition_motion = flow.classify_motion(transition_flow);
    ASSERT(transition_motion == jin::temporal::LatentOpticalFlow::MotionType::TRANSITION, 
           "Very large flow should be classified as TRANSITION");
    
    std::cout << "✓ LatentOpticalFlow motion classification passed" << std::endl;
}

void test_optical_flow_smoothing() {
    std::cout << "Testing LatentOpticalFlow smoothing..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Reset flow
    flow.reset();
    
    // First flow (raw)
    jin::Tensor latent1({2});
    latent1.fill(1.0f);
    jin::Tensor latent2({2});
    latent2.fill(3.0f);  // Big jump
    jin::Tensor flow1 = flow.estimate_flow(latent1, latent2);
    
    // Second flow with smoothing
    jin::Tensor latent3({2});
    latent3.fill(3.5f);  // Another jump
    jin::Tensor flow2 = flow.estimate_flow(latent2, latent3);
    
    // Flow2 should be smoothed compared to raw difference
    float raw_diff = 0.5f;  // (3.5 - 3.0) = 0.5
    const float* flow2_data = flow2.data_ptr<float>();
    
    // With smoothing, flow2 should be influenced by previous flow1
    ASSERT(flow2_data[0] <= raw_diff * 1.1f, "Smoothing should reduce large changes");
    
    std::cout << "✓ LatentOpticalFlow smoothing passed" << std::endl;
}

void test_optical_flow_average_magnitude() {
    std::cout << "Testing LatentOpticalFlow average magnitude..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Create sequence of flows
    std::vector<jin::Tensor> flows;
    
    jin::Tensor flow1({2});
    flow1.data_ptr<float>()[0] = 1.0f;
    flow1.data_ptr<float>()[1] = 2.0f;
    flows.push_back(flow1);
    
    jin::Tensor flow2({2});
    flow2.data_ptr<float>()[0] = 3.0f;
    flow2.data_ptr<float>()[1] = 4.0f;
    flows.push_back(flow2);
    
    float avg_magnitude = flow.average_flow_magnitude(flows);
    
    // Magnitudes: sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236
    //             sqrt(3^2 + 4^2) = 5
    // Average: (2.236 + 5) / 2 = 3.618
    float expected_avg = (std::sqrt(5.0f) + 5.0f) / 2.0f;
    ASSERT(std::abs(avg_magnitude - expected_avg) < 1e-2f, 
           "Average magnitude should match expected value");
    
    std::cout << "✓ LatentOpticalFlow average magnitude passed" << std::endl;
}

void test_optical_flow_reset() {
    std::cout << "Testing LatentOpticalFlow reset..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Compute some flow
    jin::Tensor latent1({2});
    latent1.fill(1.0f);
    jin::Tensor latent2({2});
    latent2.fill(2.0f);
    flow.estimate_flow(latent1, latent2);
    
    // Check that last flow is stored
    const auto& last_flow = flow.get_last_flow();
    ASSERT(last_flow.size() > 0, "Last flow should be stored");
    
    // Reset
    flow.reset();
    
    // Last flow should be empty after reset
    const auto& reset_last_flow = flow.get_last_flow();
    ASSERT(reset_last_flow.size() == 0, "Last flow should be empty after reset");
    
    std::cout << "✓ LatentOpticalFlow reset passed" << std::endl;
}

void test_optical_flow_edge_cases() {
    std::cout << "Testing LatentOpticalFlow edge cases..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Test with mismatched tensor sizes
    jin::Tensor latent_small({2});
    latent_small.fill(1.0f);
    jin::Tensor latent_large({4});
    latent_large.fill(2.0f);
    
    jin::Tensor flow_result = flow.estimate_flow(latent_small, latent_large);
    ASSERT(flow_result.size() == 0, "Should return empty tensor for mismatched sizes");
    
    // Test with empty tensors
    jin::Tensor empty_flow = flow.estimate_flow(jin::Tensor(), jin::Tensor());
    ASSERT(empty_flow.size() == 0, "Should handle empty tensors gracefully");
    
    // Test magnitude with empty flow
    float empty_magnitude = flow.flow_magnitude(jin::Tensor());
    ASSERT(std::abs(empty_magnitude) < 1e-5f, "Empty flow should have zero magnitude");
    
    // Test average magnitude with empty vector
    float empty_avg_magnitude = flow.average_flow_magnitude({});
    ASSERT(std::abs(empty_avg_magnitude) < 1e-5f, "Empty flow vector should have zero average");
    
    std::cout << "✓ LatentOpticalFlow edge cases passed" << std::endl;
}

void test_optical_flow_configuration() {
    std::cout << "Testing LatentOpticalFlow configuration..." << std::endl;
    
    jin::temporal::LatentOpticalFlow flow;
    
    // Test flow smoothing setting
    flow.set_flow_smoothing(0.5f);
    ASSERT(flow.get_flow_smoothing() == 0.5f, "Flow smoothing should be updated");
    
    // Test saccade threshold setting
    flow.set_saccade_threshold(0.8f);
    ASSERT(flow.get_saccade_threshold() == 0.8f, "Saccade threshold should be updated");
    
    // Test clamping
    flow.set_flow_smoothing(-0.5f);
    ASSERT(flow.get_flow_smoothing() == 0.0f, "Negative smoothing should be clamped to 0");
    
    flow.set_flow_smoothing(1.5f);
    ASSERT(flow.get_flow_smoothing() == 1.0f, "Smoothing > 1 should be clamped to 1");
    
    flow.set_saccade_threshold(-1.0f);
    ASSERT(flow.get_saccade_threshold() >= 0.0f, "Negative threshold should be clamped");
    
    std::cout << "✓ LatentOpticalFlow configuration passed" << std::endl;
}

void run_optical_flow_tests() {
    std::cout << "\n=== Running LatentOpticalFlow Tests ===" << std::endl;
    
    test_optical_flow_basic();
    test_optical_flow_frame_state();
    test_optical_flow_magnitude();
    test_optical_flow_saccade_detection();
    test_optical_flow_motion_classification();
    test_optical_flow_smoothing();
    test_optical_flow_average_magnitude();
    test_optical_flow_reset();
    test_optical_flow_edge_cases();
    test_optical_flow_configuration();
    
    std::cout << "✓ All LatentOpticalFlow tests passed!" << std::endl;
}