#include "../src/temporal/frame_state.h"
#include "../src/core/tensor.h"
#include "../src/core/math_ops.h"
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

void test_frame_state_basic() {
    std::cout << "Testing FrameState basic functionality..." << std::endl;
    
    // Create FrameState
    jin::temporal::FrameState state;
    state.frame_index = 0;
    state.timestamp = 0.0f;
    state.drift_score = 0.0f;
    state.motion_smoothness = 1.0f;
    state.scene_stability = 1.0f;
    
    // Create tensors
    jin::Tensor character_latent({4});
    character_latent.fill(0.5f);
    state.character_latent = character_latent;
    
    jin::Tensor scene_latent({4});
    scene_latent.fill(0.3f);
    state.scene_latent = scene_latent;
    
    // Test serialization
    jin::Tensor state_tensor = state.to_tensor();
    ASSERT(state_tensor.size() > 0, "Serialization should produce non-empty tensor");
    
    // Test deserialization
    auto recovered_state = jin::temporal::FrameState::from_tensor(state_tensor, 4, 0, 0);
    ASSERT(recovered_state.frame_index == state.frame_index, "Frame index should match");
    ASSERT(std::abs(recovered_state.drift_score - state.drift_score) < 1e-5f, "Drift score should match");
    ASSERT(recovered_state.character_latent.size() > 0, "Character latent should be recovered");
    
    std::cout << "✓ FrameState basic functionality passed" << std::endl;
}

void test_frame_state_hash_and_compare() {
    std::cout << "Testing FrameState hash and comparison..." << std::endl;
    
    jin::temporal::FrameState state1, state2;
    
    // Identical states
    state1.frame_index = 5;
    state1.drift_score = 0.2f;
    state2.frame_index = 5;
    state2.drift_score = 0.2f;
    
    ASSERT(state1 == state2, "Identical states should be equal");
    ASSERT(!(state1 != state2), "Identical states should not be unequal");
    
    // Different states
    state2.drift_score = 0.3f;
    ASSERT(!(state1 == state2), "Different states should not be equal");
    ASSERT(state1 != state2, "Different states should be unequal");
    
    // Hash function
    size_t hash1 = state1.hash();
    size_t hash2 = state2.hash();
    ASSERT(hash1 != hash2, "Different states should have different hashes");
    
    std::cout << "✓ FrameState hash and comparison passed" << std::endl;
}

void test_frame_state_distance() {
    std::cout << "Testing FrameState distance calculation..." << std::endl;
    
    jin::temporal::FrameState state1, state2;
    
    // Create different character latents
    jin::Tensor char_latent1({3});
    char_latent1.fill(1.0f);
    state1.character_latent = char_latent1;
    
    jin::Tensor char_latent2({3});
    char_latent2.fill(2.0f);
    state2.character_latent = char_latent2;
    
    // Distance should be positive
    float distance = state1.distance_to(state2);
    ASSERT(distance > 0.0f, "Distance should be positive");
    
    // Distance to self should be zero
    float self_distance = state1.distance_to(state1);
    ASSERT(std::abs(self_distance) < 1e-5f, "Distance to self should be zero");
    
    std::cout << "✓ FrameState distance calculation passed" << std::endl;
}

void test_frame_state_edge_cases() {
    std::cout << "Testing FrameState edge cases..." << std::endl;
    
    // Empty tensor serialization
    jin::temporal::FrameState empty_state;
    jin::Tensor empty_tensor = empty_state.to_tensor();
    ASSERT(empty_tensor.size() == 0, "Empty state should serialize to empty tensor");
    
    // Deserialization with zero dimensions
    auto recovered = jin::temporal::FrameState::from_tensor(empty_tensor, 0, 0, 0);
    ASSERT(recovered.frame_index == 0, "Should handle zero dimensions gracefully");
    
    // Hash of empty state
    size_t empty_hash = empty_state.hash();
    ASSERT(empty_hash >= 0, "Empty state should have valid hash");
    
    std::cout << "✓ FrameState edge cases passed" << std::endl;
}

void run_frame_state_tests() {
    std::cout << "\n=== Running FrameState Tests ===" << std::endl;
    
    test_frame_state_basic();
    test_frame_state_hash_and_compare();
    test_frame_state_distance();
    test_frame_state_edge_cases();
    
    std::cout << "✓ All FrameState tests passed!" << std::endl;
}