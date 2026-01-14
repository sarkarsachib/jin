#include "../src/temporal/frame_state.h"
#include "../src/temporal/latent_lock.h"
#include "../src/temporal/drift_suppressor.h"
#include "../src/temporal/coherence_graph.h"
#include "../src/temporal/optical_flow.h"
#include "../src/temporal/temporal_attention.h"
#include "../src/temporal/motion_encoder.h"
#include "../src/temporal/scene_memory.h"
#include "../src/temporal/consistency_loss.h"
#include "../src/core/tensor.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>

// Simple assertion macro for testing
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while(0)

void test_long_sequence_1000_frames() {
    std::cout << "Testing 1000-frame sequence stability..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize temporal components
    jin::temporal::LatentLock latent_lock;
    jin::temporal::DriftSuppressor drift_suppressor;
    jin::temporal::CoherenceGraph coherence_graph;
    jin::temporal::LatentOpticalFlow optical_flow;
    jin::temporal::TemporalAttention temporal_attention(8, 16);
    jin::temporal::MotionEncoder motion_encoder(64);
    jin::temporal::SceneMemory scene_memory;
    
    // Create reference character latent from frame 0
    jin::Tensor reference_latent({8});
    reference_latent.fill(0.5f);
    
    // Initialize systems
    latent_lock.initialize(reference_latent);
    drift_suppressor.initialize(reference_latent);
    optical_flow.reset();
    
    std::vector<jin::temporal::FrameState> frame_states;
    std::vector<jin::Tensor> motion_codes;
    
    // Generate 1000 frames
    const int num_frames = 1000;
    for (int i = 0; i < num_frames; ++i) {
        // Create frame state
        jin::temporal::FrameState frame_state;
        frame_state.frame_index = i;
        frame_state.timestamp = i * 0.033f;  // ~30 FPS
        
        // Character latent with controlled drift
        jin::Tensor character_latent({8});
        float drift_amount = static_cast<float>(i) * 0.001f;  // Very slow drift
        for (size_t j = 0; j < 8; ++j) {
            character_latent.data_ptr<float>()[j] = 0.5f + drift_amount * (j % 2 == 0 ? 1.0f : -1.0f);
        }
        frame_state.character_latent = character_latent;
        
        // Scene latent (mostly stable)
        jin::Tensor scene_latent({8});
        for (size_t j = 0; j < 8; ++j) {
            scene_latent.data_ptr<float>()[j] = 0.3f + 0.01f * std::sin(i * 0.01f);
        }
        frame_state.scene_latent = scene_latent;
        
        // Motion code
        jin::Tensor motion_code({64});
        for (size_t j = 0; j < 64; ++j) {
            motion_code.data_ptr<float>()[j] = 0.1f * std::sin(i * 0.1f + j);
        }
        frame_state.motion_code = motion_code;
        
        // Apply latent lock
        if (i > 0) {
            character_latent = latent_lock.enforce_lock(character_latent);
        }
        
        // Drift suppression
        if (i > 0) {
            auto prev_state = frame_states.back();
            auto correction = drift_suppressor.detect_and_correct_frame_state(prev_state, frame_state);
            if (correction.detected) {
                character_latent = drift_suppressor.apply_correction(character_latent, correction);
                frame_state.character_latent = character_latent;
            }
        }
        
        // Compute drift score
        frame_state.drift_score = latent_lock.compute_drift(character_latent);
        
        // Optical flow
        if (i > 0) {
            auto prev_state = frame_states.back();
            jin::Tensor flow = optical_flow.estimate_flow_frame_state(prev_state, frame_state);
            frame_state.motion_smoothness = 1.0f / (1.0f + optical_flow.flow_magnitude(flow));
        } else {
            frame_state.motion_smoothness = 1.0f;
        }
        
        frame_state.scene_stability = 0.9f;  // High stability for this test
        
        frame_states.push_back(frame_state);
        motion_codes.push_back(motion_code);
        
        // Add to coherence graph
        coherence_graph.add_frame(i, frame_state);
        
        // Update scene memory
        scene_memory.update(scene_latent, jin::Tensor(), jin::Tensor(), i);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Test 1: Drift should be under control
    float final_drift = frame_states.back().drift_score;
    ASSERT(final_drift <= 0.5f, "Final drift should be under 0.5 for controlled sequence");
    
    // Test 2: Frame 1000 should be similar to frame 0
    float sequence_drift = frame_states[0].distance_to(frame_states.back());
    ASSERT(sequence_drift <= 0.2f, "Frame 1000 should be within 0.2 of frame 0");
    
    // Test 3: Coherence should be maintained
    float coherence_0_100 = coherence_graph.coherence(0, 100);
    float coherence_0_500 = coherence_graph.coherence(0, 500);
    float coherence_0_999 = coherence_graph.coherence(0, 999);
    
    ASSERT(coherence_0_100 > 0.8f, "Coherence between frames 0-100 should be high");
    ASSERT(coherence_0_500 > 0.7f, "Coherence between frames 0-500 should be reasonable");
    ASSERT(coherence_0_999 > 0.5f, "Coherence between frames 0-999 should be acceptable");
    
    // Test 4: Motion continuity
    auto pattern = motion_encoder.analyze_motion_pattern(motion_codes.back());
    ASSERT(pattern.smoothness > 0.3f, "Motion should have some smoothness");
    
    // Test 5: Performance check
    std::cout << "1000-frame sequence completed in " << duration.count() << "ms" << std::endl;
    ASSERT(duration.count() < 10000, "1000-frame sequence should complete within 10 seconds");
    
    std::cout << "✓ 1000-frame sequence test passed" << std::endl;
}

void test_drift_with_strong_lock() {
    std::cout << "Testing drift with strong latent lock..." << std::endl;
    
    jin::temporal::LatentLock latent_lock;
    
    // Initialize with reference
    jin::Tensor reference({4});
    reference.fill(1.0f);
    latent_lock.initialize(reference);
    
    // Set strong lock
    latent_lock.set_lock_strength(0.9f);
    
    // Generate sequence with increasing noise
    std::vector<float> drifts;
    for (int i = 0; i < 100; ++i) {
        // Add increasing noise
        jin::Tensor noisy_latent({4});
        float noise_level = static_cast<float>(i) * 0.01f;
        for (size_t j = 0; j < 4; ++j) {
            noisy_latent.data_ptr<float>()[j] = 1.0f + noise_level * (j % 2 == 0 ? 1.0f : -1.0f);
        }
        
        // Apply strong lock
        jin::Tensor locked_latent = latent_lock.enforce_lock(noisy_latent);
        
        // Compute drift
        float drift = latent_lock.compute_drift(locked_latent);
        drifts.push_back(drift);
    }
    
    // With strong lock, drift should remain low even with increasing noise
    float max_drift = *std::max_element(drifts.begin(), drifts.end());
    ASSERT(max_drift < 0.3f, "Strong lock should keep drift under 0.3");
    
    // Drift should generally increase but stay controlled
    for (size_t i = 1; i < drifts.size(); ++i) {
        // Allow some increase but not too much
        float drift_increase = drifts[i] - drifts[i-1];
        ASSERT(drift_increase < 0.1f, "Drift increase should be gradual");
    }
    
    std::cout << "✓ Drift with strong lock test passed" << std::endl;
}

void test_scene_transitions() {
    std::cout << "Testing scene transition handling..." << std::endl;
    
    jin::temporal::SceneMemory scene_memory;
    jin::temporal::CoherenceGraph coherence_graph;
    
    // Simulate scene 1 (stable)
    for (int i = 0; i < 50; ++i) {
        jin::Tensor bg({4});
        for (size_t j = 0; j < 4; ++j) {
            bg.data_ptr<float>()[j] = 1.0f + 0.01f * std::sin(i);
        }
        
        scene_memory.update(bg, jin::Tensor(), jin::Tensor(), i);
        
        jin::temporal::FrameState state;
        state.frame_index = i;
        state.scene_latent = bg;
        coherence_graph.add_frame(i, state);
    }
    
    // Scene transition (big change)
    for (int i = 50; i < 60; ++i) {
        jin::Tensor bg({4});
        for (size_t j = 0; j < 4; ++j) {
            bg.data_ptr<float>()[j] = 5.0f + 0.1f * std::cos(i);  // Big change
        }
        
        scene_memory.update(bg, jin::Tensor(), jin::Tensor(), i);
        
        jin::temporal::FrameState state;
        state.frame_index = i;
        state.scene_latent = bg;
        coherence_graph.add_frame(i, state);
    }
    
    // Scene 2 (stable in new location)
    for (int i = 60; i < 100; ++i) {
        jin::Tensor bg({4});
        for (size_t j = 0; j < 4; ++j) {
            bg.data_ptr<float>()[j] = 5.0f + 0.01f * std::sin(i);
        }
        
        scene_memory.update(bg, jin::Tensor(), jin::Tensor(), i);
        
        jin::temporal::FrameState state;
        state.frame_index = i;
        state.scene_latent = bg;
        coherence_graph.add_frame(i, state);
    }
    
    // Test scene change detection
    auto changes = scene_memory.detect_scene_changes(0.7f);
    ASSERT(changes.size() > 0, "Should detect at least one scene change");
    
    // Test coherence across scene boundary
    float coherence_before = coherence_graph.coherence(30, 40);  // Same scene
    float coherence_after = coherence_graph.coherence(40, 70);  // Different scenes
    float coherence_same_scene = coherence_graph.coherence(70, 80);  // Same scene again
    
    ASSERT(coherence_before > coherence_after, "Coherence should drop at scene boundaries");
    ASSERT(coherence_same_scene > coherence_after, "Coherence should recover within same scene");
    
    std::cout << "✓ Scene transition test passed" << std::endl;
}

void test_temporal_attention_efficiency() {
    std::cout << "Testing temporal attention efficiency..." << std::endl;
    
    jin::temporal::TemporalAttention attention(64, 16);  // Keep only last 16 frames
    
    // Create long sequence of past frames
    std::vector<jin::Tensor> past_frames;
    for (int i = 0; i < 100; ++i) {
        jin::Tensor frame({64});
        frame.fill(static_cast<float>(i));
        past_frames.push_back(frame);
    }
    
    jin::Tensor current({64});
    current.fill(50.0f);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Forward pass
    jin::Tensor output = attention.forward(current, past_frames);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    ASSERT(output.size() == 64, "Output should have correct dimension");
    
    // Should only use last 16 frames
    auto weights = attention.get_attention_weights(current, past_frames);
    ASSERT(weights.size() == 16, "Should only use last 16 frames");
    
    // Should be fast
    std::cout << "Temporal attention took " << duration.count() << " microseconds" << std::endl;
    ASSERT(duration.count() < 1000, "Should complete within 1ms");
    
    std::cout << "✓ Temporal attention efficiency test passed" << std::endl;
}

void test_consistency_losses_numerical_stability() {
    std::cout << "Testing consistency losses numerical stability..." << std::endl;
    
    // Create long sequence
    std::vector<jin::temporal::FrameState> frames;
    std::vector<std::vector<float>> attention_weights;
    
    jin::Tensor reference({4});
    reference.fill(1.0f);
    
    for (int i = 0; i < 100; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        
        // Character latent with some drift
        jin::Tensor char_latent({4});
        for (size_t j = 0; j < 4; ++j) {
            char_latent.data_ptr<float>()[j] = 1.0f + 0.01f * std::sin(i + j);
        }
        state.character_latent = char_latent;
        
        // Motion code
        jin::Tensor motion_code({8});
        for (size_t j = 0; j < 8; ++j) {
            motion_code.data_ptr<float>()[j] = 0.1f * std::cos(i * 0.1f + j);
        }
        state.motion_code = motion_code;
        
        // Scene latent
        jin::Tensor scene_latent({4});
        for (size_t j = 0; j < 4; ++j) {
            scene_latent.data_ptr<float>()[j] = 0.5f + 0.001f * std::sin(i * 0.01f + j);
        }
        state.scene_latent = scene_latent;
        
        frames.push_back(state);
        
        // Attention weights
        std::vector<float> weights(16);
        for (size_t j = 0; j < 16; ++j) {
            weights[j] = 0.06f + 0.01f * std::sin(i + j);
        }
        attention_weights.push_back(weights);
    }
    
    // Compute losses
    auto losses = jin::temporal::compute_temporal_loss(frames, reference, attention_weights);
    
    // Check numerical stability
    ASSERT(!jin::temporal::has_nan_or_inf(losses.total), "Total loss should be numerically stable");
    
    // Normalize losses
    auto normalized = jin::temporal::normalize_losses(losses);
    ASSERT(!jin::temporal::has_nan_or_inf(normalized.total), "Normalized losses should be stable");
    
    // Check that all loss components are reasonable
    ASSERT(normalized.total.item() > 0.0f, "Total loss should be positive");
    ASSERT(normalized.total.item() < 100.0f, "Total loss should not be extremely large");
    
    std::cout << "✓ Consistency losses numerical stability test passed" << std::endl;
}

void test_performance_benchmark() {
    std::cout << "Running performance benchmark..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize components
    jin::temporal::LatentLock latent_lock;
    jin::temporal::DriftSuppressor drift_suppressor;
    jin::temporal::CoherenceGraph coherence_graph;
    jin::temporal::LatentOpticalFlow optical_flow;
    jin::temporal::TemporalAttention temporal_attention(32, 8);
    jin::temporal::MotionEncoder motion_encoder(32);
    
    // Initialize with reference
    jin::Tensor reference({32});
    reference.fill(0.5f);
    latent_lock.initialize(reference);
    drift_suppressor.initialize(reference);
    
    // Process 1000 frames
    for (int i = 0; i < 1000; ++i) {
        // Create frame data
        jin::Tensor char_latent({32});
        char_latent.fill(0.5f + 0.001f * std::sin(i));
        
        jin::Tensor scene_latent({32});
        scene_latent.fill(0.3f + 0.0005f * std::cos(i));
        
        jin::Tensor motion_code({32});
        motion_code.fill(0.1f * std::sin(i * 0.1f));
        
        // Apply temporal processing
        char_latent = latent_lock.enforce_lock(char_latent);
        
        // Create frame state
        jin::temporal::FrameState state;
        state.frame_index = i;
        state.character_latent = char_latent;
        state.scene_latent = scene_latent;
        state.motion_code = motion_code;
        state.drift_score = latent_lock.compute_drift(char_latent);
        state.motion_smoothness = 0.9f;
        state.scene_stability = 0.95f;
        
        // Drift suppression
        if (i > 0) {
            // Simulate previous state
            jin::temporal::FrameState prev_state;
            prev_state.character_latent = reference.clone();
            auto correction = drift_suppressor.detect_and_correct_frame_state(prev_state, state);
            if (correction.detected) {
                char_latent = drift_suppressor.apply_correction(char_latent, correction);
                state.character_latent = char_latent;
            }
        }
        
        // Coherence graph
        coherence_graph.add_frame(i, state);
        
        // Temporal attention (with dummy past frames)
        std::vector<jin::Tensor> past_frames;
        for (int j = std::max(0, i - 8); j < i; ++j) {
            jin::Tensor past({32});
            past.fill(static_cast<float>(j));
            past_frames.push_back(past);
        }
        
        if (!past_frames.empty()) {
            temporal_attention.forward(char_latent, past_frames);
        }
        
        // Motion encoding
        if (i >= 4) {
            std::vector<jin::Tensor> recent_motions;
            for (int j = i - 4; j <= i; ++j) {
                jin::Tensor motion({32});
                motion.fill(0.1f * std::sin(j * 0.1f));
                recent_motions.push_back(motion);
            }
            motion_encoder.encode(recent_motions);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Performance benchmark: " << duration.count() << "ms for 1000 frames" << std::endl;
    
    // Should process 1000 frames in under 100ms (very generous)
    ASSERT(duration.count() < 100, "Should process 1000 frames in under 100ms");
    
    std::cout << "✓ Performance benchmark passed" << std::endl;
}

void run_long_sequence_tests() {
    std::cout << "\n=== Running Long Sequence Tests ===" << std::endl;
    
    test_long_sequence_1000_frames();
    test_drift_with_strong_lock();
    test_scene_transitions();
    test_temporal_attention_efficiency();
    test_consistency_losses_numerical_stability();
    test_performance_benchmark();
    
    std::cout << "✓ All long sequence tests passed!" << std::endl;
}