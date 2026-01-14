#include "../src/temporal/motion_encoder.h"
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

void test_motion_encoder_basic() {
    std::cout << "Testing MotionEncoder basic functionality..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(8);  // motion_dim = 8
    
    // Create frame features sequence
    std::vector<jin::Tensor> frame_features;
    for (int i = 0; i < 3; ++i) {
        jin::Tensor feature({4});
        feature.fill(static_cast<float>(i + 1));
        frame_features.push_back(feature);
    }
    
    // Encode motion
    jin::Tensor motion_code = encoder.encode(frame_features);
    
    ASSERT(motion_code.size() > 0, "Motion code should have data");
    ASSERT(motion_code.size() == 8, "Motion code should have specified dimension");
    
    std::cout << "✓ MotionEncoder basic functionality passed" << std::endl;
}

void test_motion_encoder_frame_states() {
    std::cout << "Testing MotionEncoder with FrameStates..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(6);
    
    // Create FrameState sequence
    std::vector<jin::temporal::FrameState> frame_states;
    
    for (int i = 0; i < 4; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        
        // Add character latent
        jin::Tensor char_latent({3});
        char_latent.fill(static_cast<float>(i));
        state.character_latent = char_latent;
        
        // Add scene latent
        jin::Tensor scene_latent({3});
        scene_latent.fill(static_cast<float>(i * 2));
        state.scene_latent = scene_latent;
        
        frame_states.push_back(state);
    }
    
    // Encode FrameStates
    jin::Tensor motion_code = encoder.encode_frame_states(frame_states);
    
    ASSERT(motion_code.size() == 6, "Encoded motion code should have correct dimension");
    ASSERT(motion_code.size() > 0, "Motion code should have data");
    
    std::cout << "✓ MotionEncoder FrameState processing passed" << std::endl;
}

void test_motion_encoder_single_frame() {
    std::cout << "Testing MotionEncoder with single frame..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(5);
    
    // Single frame
    std::vector<jin::Tensor> single_frame = {jin::Tensor({3})};
    single_frame[0].fill(2.0f);
    
    jin::Tensor motion_code = encoder.encode(single_frame);
    
    ASSERT(motion_code.size() == 5, "Single frame should still produce correct dimension");
    
    std::cout << "✓ MotionEncoder single frame passed" << std::endl;
}

void test_motion_encoder_empty_sequence() {
    std::cout << "Testing MotionEncoder with empty sequence..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(4);
    
    // Empty sequence
    std::vector<jin::Tensor> empty_frames;
    
    jin::Tensor motion_code = encoder.encode(empty_frames);
    
    ASSERT(motion_code.size() == 4, "Empty sequence should produce zero-filled motion code");
    
    // Check that it's all zeros
    const float* data = motion_code.data_ptr<float>();
    for (size_t i = 0; i < motion_code.size(); ++i) {
        ASSERT(std::abs(data[i]) < 1e-5f, "Empty sequence should produce zero motion code");
    }
    
    std::cout << "✓ MotionEncoder empty sequence passed" << std::endl;
}

void test_motion_encoder_decode() {
    std::cout << "Testing MotionEncoder decode..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(6);
    
    // Create motion code
    jin::Tensor motion_code({6});
    motion_code.fill(1.5f);
    
    // Decode to frames
    std::vector<jin::Tensor> decoded_frames = encoder.decode(motion_code, 3);
    
    ASSERT(decoded_frames.size() == 3, "Should decode to specified number of frames");
    
    for (const auto& frame : decoded_frames) {
        ASSERT(frame.size() == 6, "Each decoded frame should have same dimension as motion code");
    }
    
    std::cout << "✓ MotionEncoder decode passed" << std::endl;
}

void test_motion_encoder_motion_pattern_analysis() {
    std::cout << "Testing MotionEncoder pattern analysis..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(8);
    
    // Create motion code with different characteristics
    jin::Tensor motion_code({8});
    float values[] = {1.0f, -1.0f, 2.0f, -2.0f, 1.5f, -1.5f, 0.5f, -0.5f};
    for (size_t i = 0; i < 8; ++i) {
        motion_code.data_ptr<float>()[i] = values[i];
    }
    
    auto pattern = encoder.analyze_motion_pattern(motion_code);
    
    // Check that pattern analysis produces reasonable values
    ASSERT(pattern.action_intensity >= 0.0f && pattern.action_intensity <= 1.0f, 
           "Action intensity should be in [0, 1]");
    ASSERT(pattern.smoothness >= 0.0f && pattern.smoothness <= 1.0f, 
           "Smoothness should be in [0, 1]");
    ASSERT(pattern.walk_cycle_score >= 0.0f && pattern.walk_cycle_score <= 1.0f, 
           "Walk cycle score should be in [0, 1]");
    ASSERT(pattern.direction_change >= 0.0f, "Direction change should be non-negative");
    ASSERT(pattern.transition_probability >= 0.0f && pattern.transition_probability <= 1.0f, 
           "Transition probability should be in [0, 1]");
    
    std::cout << "✓ MotionEncoder pattern analysis passed" << std::endl;
}

void test_motion_encoder_similarity() {
    std::cout << "Testing MotionEncoder similarity computation..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(4);
    
    // Create similar motion codes
    jin::Tensor code1({4});
    code1.fill(1.0f);
    
    jin::Tensor code2({4});
    code2.fill(1.1f);  // Slightly different
    
    jin::Tensor code3({4});
    code3.fill(5.0f);  // Very different
    
    float similarity_12 = encoder.motion_similarity(code1, code2);
    float similarity_13 = encoder.motion_similarity(code1, code3);
    
    ASSERT(similarity_12 > similarity_13, "Similar codes should have higher similarity");
    ASSERT(similarity_12 > 0.9f, "Very similar codes should have high similarity");
    ASSERT(similarity_13 < 0.5f, "Very different codes should have low similarity");
    
    // Test with empty tensors
    float empty_similarity = encoder.motion_similarity(jin::Tensor(), code1);
    ASSERT(std::abs(empty_similarity) < 1e-5f, "Empty tensor similarity should be zero");
    
    std::cout << "✓ MotionEncoder similarity computation passed" << std::endl;
}

void test_motion_encoder_interpolation() {
    std::cout << "Testing MotionEncoder interpolation..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(3);
    
    // Create two motion codes
    jin::Tensor code1({3});
    code1.fill(0.0f);
    
    jin::Tensor code2({3});
    code2.fill(2.0f);
    
    // Test interpolation at different t values
    jin::Tensor interp_0 = encoder.interpolate_motion(code1, code2, 0.0f);
    jin::Tensor interp_0_5 = encoder.interpolate_motion(code1, code2, 0.5f);
    jin::Tensor interp_1 = encoder.interpolate_motion(code1, code2, 1.0f);
    
    // Check interpolation values
    const float* data_0 = interp_0.data_ptr<float>();
    const float* data_0_5 = interp_0_5.data_ptr<float>();
    const float* data_1 = interp_1.data_ptr<float>();
    
    for (size_t i = 0; i < 3; ++i) {
        ASSERT(std::abs(data_0[i] - 0.0f) < 1e-5f, "t=0 should give first code");
        ASSERT(std::abs(data_0_5[i] - 1.0f) < 1e-5f, "t=0.5 should give average");
        ASSERT(std::abs(data_1[i] - 2.0f) < 1e-5f, "t=1 should give second code");
    }
    
    // Test clamping
    jin::Tensor interp_neg = encoder.interpolate_motion(code1, code2, -0.5f);
    jin::Tensor interp_over = encoder.interpolate_motion(code1, code2, 1.5f);
    
    const float* data_neg = interp_neg.data_ptr<float>();
    const float* data_over = interp_over.data_ptr<float>();
    
    // Should be clamped to [0, 1] range
    ASSERT(data_neg[0] == data_0[0], "Negative t should be clamped to 0");
    ASSERT(data_over[0] == data_1[0], "t > 1 should be clamped to 1");
    
    std::cout << "✓ MotionEncoder interpolation passed" << std::endl;
}

void test_motion_encoder_dimension_variations() {
    std::cout << "Testing MotionEncoder with different dimensions..." << std::endl;
    
    // Test with different motion dimensions
    std::vector<int> dimensions = {1, 4, 16, 64};
    
    for (int dim : dimensions) {
        jin::temporal::MotionEncoder encoder(dim);
        
        // Create input frames
        std::vector<jin::Tensor> frames;
        for (int i = 0; i < 2; ++i) {
            jin::Tensor frame({8});
            frame.fill(static_cast<float>(i));
            frames.push_back(frame);
        }
        
        jin::Tensor motion_code = encoder.encode(frames);
        ASSERT(motion_code.size() == dim, "Output dimension should match specified motion_dim");
    }
    
    std::cout << "✓ MotionEncoder dimension variations passed" << std::endl;
}

void test_motion_encoder_edge_cases() {
    std::cout << "Testing MotionEncoder edge cases..." << std::endl;
    
    jin::temporal::MotionEncoder encoder(2);
    
    // Test with mismatched frame dimensions
    std::vector<jin::Tensor> frames;
    frames.push_back(jin::Tensor({3}));  // 3D
    frames.push_back(jin::Tensor({5}));  // 5D (different)
    
    // Should handle gracefully (not crash)
    jin::Tensor motion_code = encoder.encode(frames);
    ASSERT(motion_code.size() == 2, "Should still produce output with correct dimension");
    
    // Test decode with zero frames
    jin::Tensor any_code({2});
    any_code.fill(1.0f);
    auto empty_decoded = encoder.decode(any_code, 0);
    ASSERT(empty_decoded.empty(), "Zero frames should produce empty result");
    
    std::cout << "✓ MotionEncoder edge cases passed" << std::endl;
}

void run_motion_encoder_tests() {
    std::cout << "\n=== Running MotionEncoder Tests ===" << std::endl;
    
    test_motion_encoder_basic();
    test_motion_encoder_frame_states();
    test_motion_encoder_single_frame();
    test_motion_encoder_empty_sequence();
    test_motion_encoder_decode();
    test_motion_encoder_motion_pattern_analysis();
    test_motion_encoder_similarity();
    test_motion_encoder_interpolation();
    test_motion_encoder_dimension_variations();
    test_motion_encoder_edge_cases();
    
    std::cout << "✓ All MotionEncoder tests passed!" << std::endl;
}