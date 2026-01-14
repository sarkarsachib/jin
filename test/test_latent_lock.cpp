#include "../src/temporal/latent_lock.h"
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

void test_latent_lock_initialization() {
    std::cout << "Testing LatentLock initialization..." << std::endl;
    
    jin::temporal::LatentLock lock;
    ASSERT(!lock.is_initialized(), "Lock should not be initialized initially");
    
    // Initialize with anchor latent
    jin::Tensor anchor({4});
    anchor.fill(0.5f);
    lock.initialize(anchor);
    
    ASSERT(lock.is_initialized(), "Lock should be initialized after initialization");
    
    // Check that anchor latent is stored
    ASSERT(lock.get_anchor_latent().size() > 0, "Anchor latent should be stored");
    
    std::cout << "✓ LatentLock initialization passed" << std::endl;
}

void test_latent_lock_enforcement() {
    std::cout << "Testing LatentLock enforcement..." << std::endl;
    
    jin::temporal::LatentLock lock;
    
    // Initialize with known anchor
    jin::Tensor anchor({4});
    anchor.fill(1.0f);
    lock.initialize(anchor);
    
    // Create noisy latent
    jin::Tensor noisy_latent({4});
    noisy_latent.fill(0.5f);
    
    // Apply lock with default strength (0.7)
    jin::Tensor locked_latent = lock.enforce_lock(noisy_latent);
    
    ASSERT(locked_latent.size() > 0, "Locked latent should have data");
    
    // With lock strength 0.7, result should be:
    // result = anchor * 0.7 + noisy * 0.3 = 1.0 * 0.7 + 0.5 * 0.3 = 0.7 + 0.15 = 0.85
    const float* locked_data = locked_latent.data_ptr<float>();
    for (size_t i = 0; i < locked_latent.size(); ++i) {
        ASSERT(std::abs(locked_data[i] - 0.85f) < 1e-5f, 
               "Locked value should be 0.85f with default lock strength");
    }
    
    std::cout << "✓ LatentLock enforcement passed" << std::endl;
}

void test_latent_lock_drift_computation() {
    std::cout << "Testing LatentLock drift computation..." << std::endl;
    
    jin::temporal::LatentLock lock;
    
    // Initialize with anchor
    jin::Tensor anchor({3});
    anchor.fill(1.0f);
    lock.initialize(anchor);
    
    // Test zero drift (same as anchor)
    float zero_drift = lock.compute_drift(anchor);
    ASSERT(std::abs(zero_drift) < 1e-5f, "Drift from anchor should be zero");
    
    // Test positive drift
    jin::Tensor different_latent({3});
    different_latent.fill(2.0f);
    float drift = lock.compute_drift(different_latent);
    
    // Drift should be sqrt(3 * 1^2) = sqrt(3) ≈ 1.732
    float expected_drift = std::sqrt(3.0f);
    ASSERT(std::abs(drift - expected_drift) < 1e-5f, 
           "Drift should match expected L2 distance");
    
    std::cout << "✓ LatentLock drift computation passed" << std::endl;
}

void test_latent_lock_drift_components() {
    std::cout << "Testing LatentLock drift components..." << std::endl;
    
    jin::temporal::LatentLock lock;
    
    // Initialize with anchor
    jin::Tensor anchor({4});
    anchor.fill(1.0f);
    lock.initialize(anchor);
    
    // Create FrameState with character latent
    jin::temporal::FrameState state;
    state.frame_index = 5;
    jin::Tensor character_latent({4});
    character_latent.fill(1.5f);  // Different from anchor
    state.character_latent = character_latent;
    
    auto drift_components = lock.compute_drift_components(state);
    
    // Should have positive drift
    ASSERT(drift_components.character_drift > 0.0f, "Character drift should be positive");
    ASSERT(drift_components.total_drift > 0.0f, "Total drift should be positive");
    
    std::cout << "✓ LatentLock drift components passed" << std::endl;
}

void test_latent_lock_strength() {
    std::cout << "Testing LatentLock strength settings..." << std::endl;
    
    jin::temporal::LatentLock lock;
    
    // Initialize
    jin::Tensor anchor({2});
    anchor.fill(1.0f);
    lock.initialize(anchor);
    
    jin::Tensor noisy({2});
    noisy.fill(0.0f);
    
    // Test different lock strengths
    lock.set_lock_strength(0.0f);  // No lock
    jin::Tensor unlocked = lock.enforce_lock(noisy);
    ASSERT(unlocked.data_ptr<float>()[0] == 0.0f, "With zero lock, should return noisy as-is");
    
    lock.set_lock_strength(1.0f);  // Maximum lock
    jin::Tensor fully_locked = lock.enforce_lock(noisy);
    ASSERT(fully_locked.data_ptr<float>()[0] == 1.0f, "With full lock, should return anchor");
    
    // Test clamping
    lock.set_lock_strength(-0.5f);
    ASSERT(lock.get_lock_strength() == 0.0f, "Negative strength should be clamped to 0");
    
    lock.set_lock_strength(1.5f);
    ASSERT(lock.get_lock_strength() == 1.0f, "Strength > 1 should be clamped to 1");
    
    std::cout << "✓ LatentLock strength settings passed" << std::endl;
}

void test_latent_lock_uninitialized() {
    std::cout << "Testing LatentLock uninitialized behavior..." << std::endl;
    
    jin::temporal::LatentLock lock;
    
    // Should not be initialized
    ASSERT(!lock.is_initialized(), "Lock should not be initialized");
    
    // Apply lock to noisy tensor
    jin::Tensor noisy({3});
    noisy.fill(0.5f);
    jin::Tensor result = lock.enforce_lock(noisy);
    
    // Should return input unchanged when not initialized
    ASSERT(result.size() == noisy.size(), "Result should have same size as input");
    
    // Drift should be zero when not initialized
    float drift = lock.compute_drift(noisy);
    ASSERT(std::abs(drift) < 1e-5f, "Drift should be zero when not initialized");
    
    std::cout << "✓ LatentLock uninitialized behavior passed" << std::endl;
}

void run_latent_lock_tests() {
    std::cout << "\n=== Running LatentLock Tests ===" << std::endl;
    
    test_latent_lock_initialization();
    test_latent_lock_enforcement();
    test_latent_lock_drift_computation();
    test_latent_lock_drift_components();
    test_latent_lock_strength();
    test_latent_lock_uninitialized();
    
    std::cout << "✓ All LatentLock tests passed!" << std::endl;
}