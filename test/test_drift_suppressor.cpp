#include "../src/temporal/drift_suppressor.h"
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

void test_drift_suppressor_initialization() {
    std::cout << "Testing DriftSuppressor initialization..." << std::endl;
    
    jin::temporal::DriftSuppressor suppressor;
    ASSERT(!suppressor.is_initialized(), "Suppressor should not be initialized initially");
    
    // Initialize with reference latent
    jin::Tensor reference({4});
    reference.fill(0.5f);
    suppressor.initialize(reference);
    
    ASSERT(suppressor.is_initialized(), "Suppressor should be initialized after initialization");
    
    // Test default configuration
    ASSERT(suppressor.get_drift_threshold() > 0, "Drift threshold should be positive");
    ASSERT(suppressor.get_correction_strength() > 0, "Correction strength should be positive");
    
    std::cout << "✓ DriftSuppressor initialization passed" << std::endl;
}

void test_drift_suppressor_detection() {
    std::cout << "Testing DriftSuppressor drift detection..." << std::endl;
    
    jin::temporal::DriftSuppressor suppressor;
    
    // Initialize with reference
    jin::Tensor reference({3});
    reference.fill(1.0f);
    suppressor.initialize(reference);
    
    // Test no drift case
    jin::Tensor same_latent({3});
    same_latent.fill(1.0f);  // Same as reference
    auto correction1 = suppressor.detect_and_correct(same_latent, same_latent);
    
    ASSERT(!correction1.detected, "Should not detect drift when latents are the same");
    ASSERT(std::abs(correction1.drift_score) < 1e-5f, "Drift score should be zero for same latents");
    
    // Test drift case
    jin::Tensor different_latent({3});
    different_latent.fill(2.0f);  // Far from reference
    auto correction2 = suppressor.detect_and_correct(same_latent, different_latent);
    
    ASSERT(correction2.detected, "Should detect drift when latents are different");
    ASSERT(correction2.drift_score > 0.0f, "Drift score should be positive");
    ASSERT(correction2.correction_vector.size() > 0, "Should have correction vector");
    
    std::cout << "✓ DriftSuppressor drift detection passed" << std::endl;
}

void test_drift_suppressor_frame_state() {
    std::cout << "Testing DriftSuppressor with FrameState..." << std::endl;
    
    jin::temporal::DriftSuppressor suppressor;
    
    // Initialize with reference
    jin::Tensor reference({4});
    reference.fill(1.0f);
    suppressor.initialize(reference);
    
    // Create FrameStates
    jin::temporal::FrameState prev_state, curr_state;
    
    prev_state.character_latent = reference.clone();
    curr_state.character_latent = reference.clone();
    
    // No drift case
    auto correction1 = suppressor.detect_and_correct_frame_state(prev_state, curr_state);
    ASSERT(!correction1.detected, "Should not detect drift with identical latents");
    
    // Drift case
    jin::Tensor different_latent({4});
    different_latent.fill(2.0f);
    curr_state.character_latent = different_latent;
    
    auto correction2 = suppressor.detect_and_correct_frame_state(prev_state, curr_state);
    ASSERT(correction2.detected, "Should detect drift with different latents");
    ASSERT(correction2.drift_score > 0.0f, "Should have positive drift score");
    
    std::cout << "✓ DriftSuppressor FrameState processing passed" << std::endl;
}

void test_drift_suppressor_correction_application() {
    std::cout << "Testing DriftSuppressor correction application..." << std::endl;
    
    jin::temporal::DriftSuppressor suppressor;
    
    // Initialize
    jin::Tensor reference({2});
    reference.fill(1.0f);
    suppressor.initialize(reference);
    
    // Create a drift case
    jin::Tensor current_latent({2});
    current_latent.fill(2.0f);
    
    auto correction = suppressor.detect_and_correct(reference, current_latent);
    
    if (correction.detected) {
        // Apply correction
        jin::Tensor corrected = suppressor.apply_correction(current_latent, correction);
        
        ASSERT(corrected.size() == current_latent.size(), "Corrected tensor should have same size");
        
        // Correction should move toward reference
        float original_distance = (current_latent - reference).sqrt().sum().item();
        float corrected_distance = (corrected - reference).sqrt().sum().item();
        ASSERT(corrected_distance < original_distance, "Correction should reduce drift");
    }
    
    std::cout << "✓ DriftSuppressor correction application passed" << std::endl;
}

void test_drift_suppressor_smoothing() {
    std::cout << "Testing DriftSuppressor smoothing..." << std::endl;
    
    jin::temporal::DriftSuppressor suppressor;
    
    // Initialize
    jin::Tensor reference({3});
    reference.fill(0.0f);
    suppressor.initialize(reference);
    
    // Create sequence with gradual drift
    std::vector<jin::Tensor> latents;
    for (int i = 0; i < 5; ++i) {
        jin::Tensor latent({3});
        latent.fill(static_cast<float>(i) * 0.5f);  // Gradual drift
        latents.push_back(latent);
    }
    
    std::vector<jin::temporal::DriftSuppressor::DriftCorrection> corrections;
    for (size_t i = 1; i < latents.size(); ++i) {
        auto correction = suppressor.detect_and_correct(latents[i-1], latents[i]);
        corrections.push_back(correction);
    }
    
    // Check that smoothing is applied (corrections should be gradual)
    for (size_t i = 1; i < corrections.size(); ++i) {
        if (corrections[i].correction_vector.size() > 0 && 
            corrections[i-1].correction_vector.size() > 0) {
            // Smoothing should prevent abrupt changes in correction magnitude
            float current_mag = corrections[i].correction_vector.sqrt().sum().item();
            float prev_mag = corrections[i-1].correction_vector.sqrt().sum().item();
            float change_ratio = std::abs(current_mag - prev_mag) / std::max(prev_mag, 1e-5f);
            ASSERT(change_ratio < 2.0f, "Smoothing should prevent abrupt correction changes");
        }
    }
    
    std::cout << "✓ DriftSuppressor smoothing passed" << std::endl;
}

void test_drift_suppressor_configuration() {
    std::cout << "Testing DriftSuppressor configuration..." << std::endl;
    
    jin::temporal::DriftSuppressor suppressor;
    
    // Test threshold setting
    float new_threshold = 0.5f;
    suppressor.set_drift_threshold(new_threshold);
    ASSERT(suppressor.get_drift_threshold() == new_threshold, "Threshold should be updated");
    
    // Test correction strength
    float new_strength = 0.2f;
    suppressor.set_correction_strength(new_strength);
    ASSERT(suppressor.get_correction_strength() == new_strength, "Correction strength should be updated");
    
    // Test invalid values (should be clamped)
    suppressor.set_drift_threshold(-1.0f);
    ASSERT(suppressor.get_drift_threshold() >= 0.0f, "Negative threshold should be clamped");
    
    suppressor.set_correction_strength(1.5f);
    ASSERT(suppressor.get_correction_strength() <= 1.0f, "Strength > 1 should be clamped");
    
    std::cout << "✓ DriftSuppressor configuration passed" << std::endl;
}

void test_drift_suppressor_edge_cases() {
    std::cout << "Testing DriftSuppressor edge cases..." << std::endl;
    
    jin::temporal::DriftSuppressor suppressor;
    
    // Test without initialization
    jin::Tensor any_latent({2});
    any_latent.fill(1.0f);
    auto correction = suppressor.detect_and_correct(any_latent, any_latent);
    ASSERT(!correction.detected, "Should not detect drift without initialization");
    
    // Test with empty tensors
    suppressor.initialize(jin::Tensor({0}));
    auto empty_correction = suppressor.detect_and_correct(jin::Tensor(), jin::Tensor());
    ASSERT(!empty_correction.detected, "Should handle empty tensors gracefully");
    
    // Test get_current_drift without initialization
    float drift = suppressor.get_current_drift(any_latent);
    ASSERT(std::abs(drift) < 1e-5f, "Drift should be zero without initialization");
    
    std::cout << "✓ DriftSuppressor edge cases passed" << std::endl;
}

void run_drift_suppressor_tests() {
    std::cout << "\n=== Running DriftSuppressor Tests ===" << std::endl;
    
    test_drift_suppressor_initialization();
    test_drift_suppressor_detection();
    test_drift_suppressor_frame_state();
    test_drift_suppressor_correction_application();
    test_drift_suppressor_smoothing();
    test_drift_suppressor_configuration();
    test_drift_suppressor_edge_cases();
    
    std::cout << "✓ All DriftSuppressor tests passed!" << std::endl;
}