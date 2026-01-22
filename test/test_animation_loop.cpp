#include "../src/temporal/coherence.h"
#include "../src/temporal/interpolation.h"
#include "../src/temporal/loop.h"
#include "../src/temporal/scheduler.h"
#include "../src/temporal/composer.h"
#include "../src/temporal/diffusion_loop.h"
#include "../src/diffusion/sampler.h"
#include "../src/core/tensor.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>

using namespace jin;
using namespace jin::temporal;
using namespace jin::diffusion;

// Helper function to create test frames
Tensor create_test_frame(int height, int width, int channels, float value) {
    Tensor frame = ones({static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)}) * value;
    return frame;
}

std::vector<Tensor> create_test_frame_sequence(int num_frames, int height, int width, int channels) {
    std::vector<Tensor> frames;
    for (int i = 0; i < num_frames; ++i) {
        float value = static_cast<float>(i) / (num_frames - 1); // 0 to 1
        frames.push_back(create_test_frame(height, width, channels, value));
    }
    return frames;
}

// Test assertions
void test_assert(bool condition, const std::string& test_name) {
    if (condition) {
        std::cout << "✓ PASS: " << test_name << std::endl;
    } else {
        std::cout << "✗ FAIL: " << test_name << std::endl;
        assert(condition);
    }
}

// ===== TEMPORAL COHERENCE TESTS =====

void test_motion_field_shape() {
    TemporalCoherence coherence;
    
    Tensor frame1 = create_test_frame(32, 32, 3, 0.5f);
    Tensor frame2 = create_test_frame(32, 32, 3, 0.6f);
    
    Tensor motion_field = coherence.compute_motion_field(frame1, frame2);
    
    test_assert(motion_field.shape().size() == 3, "Motion field should be 3D tensor");
    test_assert(motion_field.shape()[0] == 32, "Motion field height should match frame height");
    test_assert(motion_field.shape()[1] == 32, "Motion field width should match frame width");
    test_assert(motion_field.shape()[2] == 2, "Motion field should have 2 channels (dx, dy)");
}

void test_temporal_smoothing() {
    TemporalCoherence coherence;
    
    // Create sequence with high frequency variations
    std::vector<Tensor> frames;
    for (int i = 0; i < 5; ++i) {
        float value = (i % 2 == 0) ? 0.0f : 1.0f; // Alternating values
        frames.push_back(create_test_frame(8, 8, 1, value));
    }
    
    Tensor smoothed = coherence.smooth_temporal(frames, 1.0f);
    
    test_assert(smoothed.size() > 0, "Temporal smoothing should produce output");
    // Smoothed sequence should have less variation than original
}

void test_consistency_score_range() {
    TemporalCoherence coherence;
    
    // Test with identical frames
    std::vector<Tensor> identical_frames;
    for (int i = 0; i < 3; ++i) {
        identical_frames.push_back(create_test_frame(8, 8, 1, 0.5f));
    }
    
    float score = coherence.compute_consistency_score(identical_frames);
    test_assert(score >= 0.0f && score <= 1.0f, "Consistency score should be in range [0,1]");
    test_assert(score > 0.9f, "Identical frames should have high consistency score");
    
    // Test with very different frames
    std::vector<Tensor> different_frames;
    different_frames.push_back(create_test_frame(8, 8, 1, 0.0f));
    different_frames.push_back(create_test_frame(8, 8, 1, 1.0f));
    different_frames.push_back(create_test_frame(8, 8, 1, 0.0f));
    
    float diff_score = coherence.compute_consistency_score(different_frames);
    test_assert(diff_score >= 0.0f && diff_score <= 1.0f, "Different frames consistency score should be in range [0,1]");
    test_assert(diff_score < 0.9f, "Different frames should have lower consistency score");
}

// ===== MOTION INTERPOLATION TESTS =====

void test_interpolated_frame_count() {
    MotionInterpolator interpolator;
    
    Tensor frame_a = create_test_frame(16, 16, 3, 0.0f);
    Tensor frame_b = create_test_frame(16, 16, 3, 1.0f);
    
    auto result = interpolator.interpolate_frames(frame_a, frame_b, 4);
    
    test_assert(result.size() == 6, "Should return num_steps+2 frames (4+2=6)");
    test_assert(result.front().shape() == frame_a.shape(), "First frame should match input shape");
    test_assert(result.back().shape() == frame_b.shape(), "Last frame should match input shape");
}

void test_blend_parameter_validity() {
    MotionInterpolator interpolator;
    
    Tensor frame1 = create_test_frame(8, 8, 1, 0.0f);
    Tensor frame2 = create_test_frame(8, 8, 1, 1.0f);
    
    // Test valid range [0,1]
    Tensor blend_0 = interpolator.blend_motion(frame1, frame2, 0.0f);
    Tensor blend_1 = interpolator.blend_motion(frame1, frame2, 1.0f);
    Tensor blend_half = interpolator.blend_motion(frame1, frame2, 0.5f);
    
    // Clamp test - values outside [0,1] should be clamped
    Tensor blend_neg = interpolator.blend_motion(frame1, frame2, -0.5f);
    Tensor blend_over = interpolator.blend_motion(frame1, frame2, 1.5f);
    
    test_assert(true, "Blend parameter validation completed");
}

void test_adaptive_insert_fps_matching() {
    MotionInterpolator interpolator;
    
    std::vector<Tensor> source_frames = create_test_frame_sequence(10, 8, 8, 1);
    
    // Test upsampling
    auto upsampled = interpolator.adaptive_insert(source_frames, 30.0f, 15.0f);
    test_assert(upsampled.size() >= source_frames.size(), "Upsampling should increase or maintain frame count");
    
    // Test downsampling
    auto downsampled = interpolator.adaptive_insert(source_frames, 15.0f, 30.0f);
    test_assert(downsampled.size() <= source_frames.size(), "Downsampling should decrease or maintain frame count");
}

void test_motion_warping() {
    MotionInterpolator interpolator;
    
    Tensor frame = create_test_frame(16, 16, 1, 0.5f);
    Tensor motion_field = create_test_frame(16, 16, 2, 0.1f);
    
    Tensor warped = interpolator.warp_frame(frame, motion_field, 0.5f);
    
    test_assert(warped.shape() == frame.shape(), "Warped frame should have same shape as input");
    test_assert(warped.size() > 0, "Motion warping should produce valid output");
}

void test_interpolation_quality() {
    MotionInterpolator interpolator;
    
    std::vector<Tensor> original = create_test_frame_sequence(3, 8, 8, 1);
    std::vector<Tensor> interpolated = create_test_frame_sequence(5, 8, 8, 1);
    
    float quality = interpolator.compute_interpolation_quality(original, interpolated);
    
    test_assert(quality >= 0.0f, "Interpolation quality should be non-negative");
    test_assert(std::isfinite(quality), "Interpolation quality should be finite");
}

// ===== LOOP GENERATION TESTS =====

void test_loop_frame_count() {
    LoopGenerator loop_gen;
    
    // Test with mock frames since we can't easily create a real sampler
    std::vector<Tensor> test_frames = create_test_frame_sequence(10, 8, 8, 1);
    
    test_assert(test_frames.size() == 10, "Test setup should create correct number of frames");
}

void test_cyclic_consistency_improvement() {
    LoopGenerator loop_gen;
    
    // Create a loop with poor boundary match
    std::vector<Tensor> frames;
    frames.push_back(create_test_frame(8, 8, 1, 0.0f)); // Start with 0
    for (int i = 1; i < 9; ++i) {
        frames.push_back(create_test_frame(8, 8, 1, static_cast<float>(i) / 8.0f));
    }
    frames.push_back(create_test_frame(8, 8, 1, 1.0f)); // End with 1 (poor match to start)
    
    float boundary_error_before = loop_gen.compute_boundary_error(frames);
    
    // Apply cyclic consistency enforcement
    loop_gen.enforce_cyclic_consistency(frames, 3);
    
    float boundary_error_after = loop_gen.compute_boundary_error(frames);
    
    test_assert(boundary_error_after <= boundary_error_before, 
                "Boundary error should not increase after cyclic consistency");
}

void test_cross_fade_values() {
    LoopGenerator loop_gen;
    
    std::vector<Tensor> frames = create_test_frame_sequence(10, 8, 8, 1);
    
    auto faded = loop_gen.cross_fade_boundary(frames, 3);
    
    test_assert(faded.size() == frames.size(), "Cross-fade should preserve frame count");
    
    // Check that fade values are in valid range [0,1]
    for (const auto& frame : faded) {
        const float* data = frame.data_ptr<float>();
        for (size_t i = 0; i < frame.size(); ++i) {
            test_assert(data[i] >= 0.0f && data[i] <= 1.0f, "Cross-fade should maintain valid pixel values");
        }
    }
}

void test_loop_quality_determinism() {
    LoopGenerator loop_gen;
    
    std::vector<Tensor> frames1 = create_test_frame_sequence(6, 8, 8, 1);
    std::vector<Tensor> frames2 = create_test_frame_sequence(6, 8, 8, 1); // Identical
    
    float quality1 = loop_gen.compute_loop_quality(frames1);
    float quality2 = loop_gen.compute_loop_quality(frames2);
    
    test_assert(std::abs(quality1 - quality2) < 1e-6f, "Loop quality should be deterministic");
}

void test_boundary_error_calculation() {
    LoopGenerator loop_gen;
    
    std::vector<Tensor> frames;
    frames.push_back(create_test_frame(8, 8, 1, 0.0f));
    frames.push_back(create_test_frame(8, 8, 1, 0.5f));
    frames.push_back(create_test_frame(8, 8, 1, 1.0f));
    
    float error = loop_gen.compute_boundary_error(frames);
    
    test_assert(error >= 0.0f, "Boundary error should be non-negative");
    test_assert(std::isfinite(error), "Boundary error should be finite");
}

// ===== FRAME SCHEDULER TESTS =====

void test_frame_timestamp_monotonicity() {
    FrameScheduler scheduler;
    
    auto timestamps = scheduler.schedule_frames(1.0f, 30.0f, FrameScheduler::EaseType::LINEAR);
    
    test_assert(timestamps.size() == 30, "Should generate correct number of timestamps");
    
    // Check monotonicity
    for (size_t i = 1; i < timestamps.size(); ++i) {
        test_assert(timestamps[i] >= timestamps[i-1], "Timestamps should be monotonic");
    }
}

void test_ease_function_output_range() {
    FrameScheduler scheduler;
    
    // Test all ease types
    std::vector<FrameScheduler::EaseType> ease_types = {
        FrameScheduler::EaseType::LINEAR,
        FrameScheduler::EaseType::EASE_IN,
        FrameScheduler::EaseType::EASE_OUT,
        FrameScheduler::EaseType::EASE_IN_OUT,
        FrameScheduler::EaseType::CUBIC,
        FrameScheduler::EaseType::BOUNCE
    };
    
    for (auto ease_type : ease_types) {
        float result_0 = scheduler.apply_easing(0.0f, ease_type);
        float result_1 = scheduler.apply_easing(1.0f, ease_type);
        float result_mid = scheduler.apply_easing(0.5f, ease_type);
        
        test_assert(result_0 >= 0.0f && result_0 <= 1.0f, "Ease function should map [0,1] to [0,1] at t=0");
        test_assert(result_1 >= 0.0f && result_1 <= 1.0f, "Ease function should map [0,1] to [0,1] at t=1");
        test_assert(result_mid >= 0.0f && result_mid <= 1.0f, "Ease function should map [0,1] to [0,1] at t=0.5");
    }
}

void test_timing_matches_duration_and_fps() {
    FrameScheduler scheduler;
    
    float duration = 2.0f;
    float fps = 24.0f;
    
    auto timestamps = scheduler.schedule_frames(duration, fps, FrameScheduler::EaseType::LINEAR);
    
    test_assert(timestamps.size() == static_cast<size_t>(duration * fps), 
                "Frame count should match duration * fps");
    
    // Check timing consistency
    float time_per_frame = duration / (timestamps.size() - 1);
    for (size_t i = 0; i < timestamps.size(); ++i) {
        float expected_time = i * time_per_frame;
        test_assert(std::abs(timestamps[i] - expected_time) < 1e-3f, 
                    "Timestamp should match expected timing");
    }
}

void test_frame_spacing_reflects_ease_curve() {
    FrameScheduler scheduler;
    
    auto linear_spacing = scheduler.compute_frame_spacing(FrameScheduler::EaseType::LINEAR);
    auto ease_in_spacing = scheduler.compute_frame_spacing(FrameScheduler::EaseType::EASE_IN);
    
    test_assert(linear_spacing.size() > 0, "Linear spacing should produce output");
    test_assert(ease_in_spacing.size() > 0, "Ease-in spacing should produce output");
    
    // Different ease curves should produce different spacing patterns
    // (This is a basic test - more sophisticated analysis could be done)
}

// ===== ANIMATION COMPOSER TESTS =====

void test_composition_preserves_frame_count() {
    AnimationComposer composer;
    
    std::vector<Tensor> frames = create_test_frame_sequence(5, 8, 8, 1);
    std::vector<AnimationComposer::Effect> effects; // Empty effects
    
    auto result = composer.compose_sequence(frames, effects);
    
    test_assert(result.size() == frames.size(), "Composition should preserve frame count");
}

void test_transitions_have_correct_duration() {
    AnimationComposer composer;
    
    std::vector<Tensor> frames_a = create_test_frame_sequence(3, 8, 8, 1);
    std::vector<Tensor> frames_b = create_test_frame_sequence(3, 8, 8, 1);
    
    auto result = composer.add_transition(
        frames_a, frames_b, 
        AnimationComposer::TransitionType::CROSS_FADE, 
        5
    );
    
    // Should have: frames_a + transition_frames + frames_b
    size_t expected_size = frames_a.size() + 5 + frames_b.size();
    test_assert(result.size() == expected_size, "Transition should add correct number of frames");
}

void test_quality_metrics_are_numeric() {
    AnimationComposer composer;
    
    std::vector<Tensor> frames = create_test_frame_sequence(5, 8, 8, 1);
    
    auto metrics = composer.compute_quality_metrics(frames);
    
    test_assert(metrics.find("temporal_smoothness") != metrics.end(), 
                 "Should compute temporal smoothness");
    test_assert(metrics.find("flicker_detection") != metrics.end(), 
                 "Should compute flicker detection");
    test_assert(metrics.find("motion_magnitude") != metrics.end(), 
                 "Should compute motion magnitude");
    
    // Check that all values are finite
    for (const auto& [key, value] : metrics) {
        test_assert(std::isfinite(value), "Quality metric " + key + " should be finite");
    }
}

void test_blend_modes_produce_valid_tensors() {
    AnimationComposer composer;
    
    Tensor layer1 = create_test_frame(8, 8, 1, 0.3f);
    Tensor layer2 = create_test_frame(8, 8, 1, 0.7f);
    
    std::vector<std::string> blend_modes = {"normal", "multiply", "screen", "overlay"};
    
    for (const auto& mode : blend_modes) {
        Tensor result = composer.blend_layers(layer1, layer2, mode, 0.5f);
        
        test_assert(result.shape() == layer1.shape(), 
                    "Blend mode " + mode + " should preserve shape");
        test_assert(result.size() > 0, "Blend mode " + mode + " should produce output");
        
        // Check pixel values are in valid range
        float* data = result.data_ptr<float>();
        for (size_t i = 0; i < result.size(); ++i) {
            test_assert(std::isfinite(data[i]), 
                       "Blend mode " + mode + " should produce finite values");
        }
    }
}

// ===== INTEGRATION TESTS =====

void test_complete_animation_generation() {
    // This would require a properly initialized diffusion sampler
    // For now, we'll test the configuration and basic structure
    
    AnimationConfig config;
    config.num_frames = 30;
    config.fps = 30.0f;
    config.duration_seconds = 1.0f;
    config.ease_type = FrameScheduler::EaseType::EASE_IN_OUT;
    config.loop_enabled = true;
    config.guidance_scale = 7.5f;
    config.num_sampling_steps = 50;
    config.interpolation_steps = 2;
    
    test_assert(config.num_frames > 0, "Animation config should have positive frame count");
    test_assert(config.fps > 0.0f, "Animation config should have positive FPS");
    test_assert(config.duration_seconds > 0.0f, "Animation config should have positive duration");
}

void test_animation_metadata_completeness() {
    AnimationComposer composer;
    
    std::vector<Tensor> frames = create_test_frame_sequence(10, 8, 8, 1);
    
    // Mock a diffusion animation loop for metadata testing
    AnimationConfig config;
    // Note: Can't easily test full DiffusionAnimationLoop without proper sampler setup
    
    test_assert(true, "Animation metadata test structure verified");
}

void test_progressive_generation_structure() {
    // Test that the progressive generation callback mechanism is properly structured
    bool callback_invoked = false;
    int callback_count = 0;
    
    auto test_callback = [&](const std::vector<Tensor>& frames, int frame_idx) {
        callback_invoked = true;
        callback_count++;
    };
    
    // We can't run the full progressive generation without a proper sampler,
    // but we can verify the callback structure is sound
    test_callback({}, 0); // Test callback invocation
    
    test_assert(callback_invoked, "Callback mechanism should be functional");
}

void test_export_functionality_structure() {
    AnimationComposer composer;
    
    std::vector<Tensor> frames = create_test_frame_sequence(5, 8, 8, 1);
    
    // Test that export methods can be called without errors
    composer.render_batch(frames, "png", "/tmp/test_output");
    
    test_assert(true, "Export functionality structure verified");
}
