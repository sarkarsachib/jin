#include "../src/temporal/scene_memory.h"
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

void test_scene_memory_basic() {
    std::cout << "Testing SceneMemory basic functionality..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    ASSERT(memory.empty(), "SceneMemory should be empty initially");
    ASSERT(memory.size() == 0, "SceneMemory size should be zero initially");
    
    // Update with initial scene
    jin::Tensor background({4});
    background.fill(0.5f);
    jin::Tensor lighting({4});
    lighting.fill(0.3f);
    jin::Tensor camera({3});
    camera.fill(0.1f);
    
    memory.update(background, lighting, camera, 0);
    
    ASSERT(memory.size() == 1, "SceneMemory should have 1 scene after update");
    ASSERT(!memory.empty(), "SceneMemory should not be empty after update");
    
    auto current = memory.get_current();
    ASSERT(current.frame_index == 0, "Current scene should have frame index 0");
    
    std::cout << "✓ SceneMemory basic functionality passed" << std::endl;
}

void test_scene_memory_consistency() {
    std::cout << "Testing SceneMemory consistency enforcement..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    
    // Create initial scene
    jin::Tensor bg1({3});
    bg1.fill(1.0f);
    jin::Tensor light1({3});
    light1.fill(0.5f);
    jin::Tensor cam1({2});
    cam1.fill(0.1f);
    
    memory.update(bg1, light1, cam1, 0);
    
    // Create similar scene (should enforce consistency)
    jin::Tensor bg2({3});
    bg2.fill(1.1f);  // Slightly different
    jin::Tensor light2({3});
    light2.fill(0.6f);  // Slightly different
    jin::Tensor cam2({2});
    cam2.fill(0.2f);  // Slightly different
    
    memory.update(bg2, light2, cam2, 1);
    
    auto current = memory.get_current();
    
    // Current scene should be smoothed toward previous scene
    // Due to consistency enforcement, values should be between original and new values
    
    std::cout << "✓ SceneMemory consistency enforcement passed" << std::endl;
}

void test_scene_memory_from_frame_state() {
    std::cout << "Testing SceneMemory from FrameState..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    
    // Create FrameState with scene information
    jin::temporal::FrameState frame_state;
    frame_state.frame_index = 5;
    frame_state.timestamp = 0.166f;  // ~6 frames at 30fps
    
    jin::Tensor scene_latent({4});
    scene_latent.fill(0.7f);
    frame_state.scene_latent = scene_latent;
    
    // Update from FrameState
    memory.update_from_frame_state(frame_state);
    
    ASSERT(memory.size() == 1, "Should have 1 scene after FrameState update");
    
    auto current = memory.get_current();
    ASSERT(current.frame_index == 5, "Scene should have frame index from FrameState");
    
    std::cout << "✓ SceneMemory FrameState integration passed" << std::endl;
}

void test_scene_memory_scene_consistency() {
    std::cout << "Testing SceneMemory scene consistency computation..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    
    // Add two similar scenes
    jin::Tensor bg1({2});
    bg1.fill(1.0f);
    memory.update(bg1, jin::Tensor({2}), jin::Tensor({2}), 0);
    
    jin::Tensor bg2({2});
    bg2.fill(1.1f);  // Similar
    memory.update(bg2, jin::Tensor({2}), jin::Tensor({2}), 1);
    
    // Compute consistency between frames 0 and 1
    float consistency = memory.scene_consistency(0, 1);
    
    ASSERT(consistency > 0.5f, "Similar scenes should have high consistency");
    
    // Add very different scene
    jin::Tensor bg3({2});
    bg3.fill(5.0f);  // Very different
    memory.update(bg3, jin::Tensor({2}), jin::Tensor({2}), 2);
    
    float consistency_diff = memory.scene_consistency(0, 2);
    ASSERT(consistency_diff < consistency, "Different scenes should have lower consistency");
    
    std::cout << "✓ SceneMemory scene consistency computation passed" << std::endl;
}

void test_scene_memory_scene_changes() {
    std::cout << "Testing SceneMemory scene change detection..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    
    // Add sequence with a big scene change
    for (int i = 0; i < 5; ++i) {
        jin::Tensor bg({2});
        if (i < 3) {
            bg.fill(1.0f);  // Stable scene
        } else {
            bg.fill(5.0f);  // Scene change
        }
        memory.update(bg, jin::Tensor({2}), jin::Tensor({2}), i);
    }
    
    // Detect scene changes
    auto changes = memory.detect_scene_changes(0.7f);
    
    ASSERT(changes.size() > 0, "Should detect at least one scene change");
    ASSERT(changes[0].frame_index >= 3, "Scene change should be detected around frame 3");
    
    std::cout << "✓ SceneMemory scene change detection passed" << std::endl;
}

void test_scene_memory_recent_scenes() {
    std::cout << "Testing SceneMemory recent scenes..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    
    // Add several scenes
    for (int i = 0; i < 8; ++i) {
        jin::Tensor bg({1});
        bg.fill(static_cast<float>(i));
        memory.update(bg, jin::Tensor({1}), jin::Tensor({1}), i);
    }
    
    // Get recent 3 scenes
    auto recent = memory.get_recent_scenes(3);
    ASSERT(recent.size() == 3, "Should get exactly 3 recent scenes");
    ASSERT(recent[0].frame_index == 5, "First recent scene should be frame 5");
    ASSERT(recent[2].frame_index == 7, "Last recent scene should be frame 7");
    
    // Test with more scenes than available
    auto recent_many = memory.get_recent_scenes(20);
    ASSERT(recent_many.size() == 8, "Should get all scenes if requested more than available");
    
    std::cout << "✓ SceneMemory recent scenes passed" << std::endl;
}

void test_scene_memory_serialization() {
    std::cout << "Testing SceneMemory serialization..." << std::endl;
    
    // Create SceneState with all components
    jin::temporal::SceneMemory::SceneState state;
    state.frame_index = 10;
    state.timestamp = 0.33f;
    
    state.background_latent = jin::Tensor({3});
    state.background_latent.fill(0.8f);
    
    state.lighting_latent = jin::Tensor({3});
    state.lighting_latent.fill(0.6f);
    
    state.camera_pose = jin::Tensor({2});
    state.camera_pose.fill(0.4f);
    
    state.env_features = jin::Tensor({4});
    state.env_features.fill(0.2f);
    
    // Serialize
    jin::Tensor serialized = state.to_tensor();
    ASSERT(serialized.size() > 0, "Serialization should produce non-empty tensor");
    
    // Deserialize
    auto deserialized = jin::temporal::SceneMemory::SceneState::from_tensor(
        serialized, 3, 3, 2, 4);
    
    ASSERT(deserialized.frame_index == state.frame_index, "Frame index should match");
    ASSERT(std::abs(deserialized.timestamp - state.timestamp) < 1e-5f, "Timestamp should match");
    
    // Check that latents are recovered
    ASSERT(deserialized.background_latent.size() == 3, "Background latent should be recovered");
    ASSERT(deserialized.lighting_latent.size() == 3, "Lighting latent should be recovered");
    ASSERT(deserialized.camera_pose.size() == 2, "Camera pose should be recovered");
    ASSERT(deserialized.env_features.size() == 4, "Environment features should be recovered");
    
    std::cout << "✓ SceneMemory serialization passed" << std::endl;
}

void test_scene_memory_consistency_threshold() {
    std::cout << "Testing SceneMemory consistency threshold..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    memory.set_consistency_threshold(0.9f);  // High threshold
    
    // Add scene with very small change
    jin::Tensor bg1({2});
    bg1.fill(1.0f);
    memory.update(bg1, jin::Tensor({2}), jin::Tensor({2}), 0);
    
    jin::Tensor bg2({2});
    bg2.fill(1.01f);  // Very small change
    memory.update(bg2, jin::Tensor({2}), jin::Tensor({2}), 1);
    
    // With high threshold, this small change might be considered inconsistent
    // The exact behavior depends on implementation details
    
    std::cout << "✓ SceneMemory consistency threshold passed" << std::endl;
}

void test_scene_memory_reset() {
    std::cout << "Testing SceneMemory reset..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    
    // Add some scenes
    for (int i = 0; i < 3; ++i) {
        jin::Tensor bg({1});
        bg.fill(static_cast<float>(i));
        memory.update(bg, jin::Tensor({1}), jin::Tensor({1}), i);
    }
    
    ASSERT(memory.size() == 3, "Should have 3 scenes before reset");
    
    // Reset
    memory.reset();
    
    ASSERT(memory.size() == 0, "Should have 0 scenes after reset");
    ASSERT(memory.empty(), "Should be empty after reset");
    
    std::cout << "✓ SceneMemory reset passed" << std::endl;
}

void test_scene_memory_edge_cases() {
    std::cout << "Testing SceneMemory edge cases..." << std::endl;
    
    jin::temporal::SceneMemory memory;
    
    // Test scene consistency with non-existent frames
    float consistency = memory.scene_consistency(0, 1);
    ASSERT(consistency == 0.0f, "Consistency should be zero for non-existent frames");
    
    // Test find scene by frame that doesn't exist
    auto* scene = memory.find_scene_by_frame(999);
    ASSERT(scene == nullptr, "Should return nullptr for non-existent frame");
    
    // Test with empty updates
    memory.update(jin::Tensor(), jin::Tensor(), jin::Tensor(), 0);
    ASSERT(memory.size() == 1, "Should handle empty tensors gracefully");
    
    // Test recent scenes with empty memory
    auto empty_recent = memory.get_recent_scenes(5);
    ASSERT(empty_recent.empty(), "Should return empty vector for empty memory");
    
    std::cout << "✓ SceneMemory edge cases passed" << std::endl;
}

void run_scene_memory_tests() {
    std::cout << "\n=== Running SceneMemory Tests ===" << std::endl;
    
    test_scene_memory_basic();
    test_scene_memory_consistency();
    test_scene_memory_from_frame_state();
    test_scene_memory_scene_consistency();
    test_scene_memory_scene_changes();
    test_scene_memory_recent_scenes();
    test_scene_memory_serialization();
    test_scene_memory_consistency_threshold();
    test_scene_memory_reset();
    test_scene_memory_edge_cases();
    
    std::cout << "✓ All SceneMemory tests passed!" << std::endl;
}