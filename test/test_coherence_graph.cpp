#include "../src/temporal/coherence_graph.h"
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

void test_coherence_graph_basic() {
    std::cout << "Testing CoherenceGraph basic functionality..." << std::endl;
    
    jin::temporal::CoherenceGraph graph;
    ASSERT(graph.empty(), "Graph should be empty initially");
    ASSERT(graph.size() == 0, "Graph size should be zero initially");
    
    // Add first frame
    jin::temporal::FrameState state1;
    state1.frame_index = 0;
    state1.drift_score = 0.1f;
    jin::Tensor char_latent1({2});
    char_latent1.fill(1.0f);
    state1.character_latent = char_latent1;
    
    graph.add_frame(0, state1);
    ASSERT(graph.size() == 1, "Graph should have 1 frame after adding first frame");
    
    // Add second frame
    jin::temporal::FrameState state2;
    state2.frame_index = 1;
    state2.drift_score = 0.2f;
    jin::Tensor char_latent2({2});
    char_latent2.fill(1.1f);  // Slightly different
    state2.character_latent = char_latent2;
    
    graph.add_frame(1, state2);
    ASSERT(graph.size() == 2, "Graph should have 2 frames after adding second frame");
    
    // Find nodes
    auto* node1 = graph.find_node(0);
    auto* node2 = graph.find_node(1);
    
    ASSERT(node1 != nullptr, "Should find frame 0");
    ASSERT(node2 != nullptr, "Should find frame 1");
    ASSERT(node1->frame_idx == 0, "Node 1 should have frame index 0");
    ASSERT(node2->frame_idx == 1, "Node 2 should have frame index 1");
    
    std::cout << "✓ CoherenceGraph basic functionality passed" << std::endl;
}

void test_coherence_graph_coherence_computation() {
    std::cout << "Testing CoherenceGraph coherence computation..." << std::endl;
    
    jin::temporal::CoherenceGraph graph;
    
    // Add similar frames
    jin::temporal::FrameState state1, state2;
    state1.frame_index = 0;
    state2.frame_index = 1;
    
    jin::Tensor char_latent({3});
    char_latent.fill(1.0f);
    state1.character_latent = char_latent;
    state2.character_latent = char_latent;  // Identical
    
    graph.add_frame(0, state1);
    graph.add_frame(1, state2);
    
    // Compute coherence between identical frames
    float coherence = graph.coherence(0, 1);
    ASSERT(coherence > 0.9f, "Coherence between identical frames should be high");
    
    // Add different frame
    jin::temporal::FrameState state3;
    state3.frame_index = 2;
    jin::Tensor different_latent({3});
    different_latent.fill(2.0f);  // Very different
    state3.character_latent = different_latent;
    
    graph.add_frame(2, state3);
    
    float coherence_different = graph.coherence(0, 2);
    ASSERT(coherence_different < coherence, "Coherence with different frames should be lower");
    
    std::cout << "✓ CoherenceGraph coherence computation passed" << std::endl;
}

void test_coherence_graph_discontinuities() {
    std::cout << "Testing CoherenceGraph discontinuity detection..." << std::endl;
    
    jin::temporal::CoherenceGraph graph;
    
    // Add frames with varying coherence
    for (int i = 0; i < 5; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        
        // Create latents with a big jump at frame 3
        jin::Tensor char_latent({2});
        if (i < 3) {
            char_latent.fill(1.0f);  // Similar frames
        } else {
            char_latent.fill(3.0f);  // Big jump
        }
        state.character_latent = char_latent;
        
        graph.add_frame(i, state);
    }
    
    // Find discontinuities with low threshold
    auto discontinuities = graph.find_discontinuities(0.5f);
    
    // Should detect discontinuity at frame 3 (where the jump happens)
    ASSERT(discontinuities.size() > 0, "Should detect at least one discontinuity");
    ASSERT(discontinuities[0] >= 1, "Discontinuity should be at frame >= 1");
    
    std::cout << "✓ CoherenceGraph discontinuity detection passed" << std::endl;
}

void test_coherence_graph_longest_subsequence() {
    std::cout << "Testing CoherenceGraph longest coherent subsequence..." << std::endl;
    
    jin::temporal::CoherenceGraph graph;
    
    // Add frames with some coherence pattern
    for (int i = 0; i < 6; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        
        jin::Tensor char_latent({2});
        if (i < 3) {
            char_latent.fill(1.0f);  // First 3 frames similar
        } else if (i < 5) {
            char_latent.fill(1.1f);  // Next 2 frames similar but different
        } else {
            char_latent.fill(5.0f);  // Last frame very different
        }
        state.character_latent = char_latent;
        
        graph.add_frame(i, state);
    }
    
    // Find longest coherent subsequence with moderate threshold
    auto subsequence = graph.longest_coherent_subsequence(0.8f);
    
    // Should find a subsequence of at least 3 consecutive frames
    ASSERT(subsequence.size() >= 3, "Should find subsequence of at least 3 frames");
    
    // Frames should be consecutive
    for (size_t i = 1; i < subsequence.size(); ++i) {
        ASSERT(subsequence[i] == subsequence[i-1] + 1, "Subsequence should be consecutive");
    }
    
    std::cout << "✓ CoherenceGraph longest coherent subsequence passed" << std::endl;
}

void test_coherence_graph_recent_frames() {
    std::cout << "Testing CoherenceGraph recent frames..." << std::endl;
    
    jin::temporal::CoherenceGraph graph;
    
    // Add several frames
    for (int i = 0; i < 10; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        jin::Tensor char_latent({1});
        char_latent.fill(static_cast<float>(i));
        state.character_latent = char_latent;
        
        graph.add_frame(i, state);
    }
    
    // Get recent 3 frames
    auto recent = graph.get_recent_frames(3);
    ASSERT(recent.size() == 3, "Should get exactly 3 recent frames");
    ASSERT(recent[0].frame_idx == 7, "First recent frame should be 7");
    ASSERT(recent[2].frame_idx == 9, "Last recent frame should be 9");
    
    // Test with more frames than available
    auto recent_many = graph.get_recent_frames(20);
    ASSERT(recent_many.size() == 10, "Should get all frames if requested more than available");
    
    std::cout << "✓ CoherenceGraph recent frames passed" << std::endl;
}

void test_coherence_graph_memory_limit() {
    std::cout << "Testing CoherenceGraph memory limit..." << std::endl;
    
    jin::temporal::CoherenceGraph graph;
    graph.set_max_frames(5);  // Limit to 5 frames
    
    // Add more than limit
    for (int i = 0; i < 10; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        jin::Tensor char_latent({1});
        char_latent.fill(static_cast<float>(i));
        state.character_latent = char_latent;
        
        graph.add_frame(i, state);
    }
    
    // Should only keep last 5 frames
    ASSERT(graph.size() == 5, "Graph should be limited to max frames");
    
    // Oldest frames should be gone
    ASSERT(graph.find_node(0) == nullptr, "Frame 0 should be removed");
    ASSERT(graph.find_node(4) == nullptr, "Frame 4 should be removed");
    
    // Recent frames should remain
    ASSERT(graph.find_node(5) != nullptr, "Frame 5 should remain");
    ASSERT(graph.find_node(9) != nullptr, "Frame 9 should remain");
    
    std::cout << "✓ CoherenceGraph memory limit passed" << std::endl;
}

void test_coherence_graph_clear() {
    std::cout << "Testing CoherenceGraph clear..." << std::endl;
    
    jin::temporal::CoherenceGraph graph;
    
    // Add some frames
    for (int i = 0; i < 3; ++i) {
        jin::temporal::FrameState state;
        state.frame_index = i;
        jin::Tensor char_latent({1});
        char_latent.fill(static_cast<float>(i));
        state.character_latent = char_latent;
        
        graph.add_frame(i, state);
    }
    
    ASSERT(graph.size() == 3, "Graph should have 3 frames");
    
    // Clear
    graph.clear();
    
    ASSERT(graph.empty(), "Graph should be empty after clear");
    ASSERT(graph.size() == 0, "Graph size should be zero after clear");
    
    std::cout << "✓ CoherenceGraph clear passed" << std::endl;
}

void test_coherence_graph_edge_cases() {
    std::cout << "Testing CoherenceGraph edge cases..." << std::endl;
    
    jin::temporal::CoherenceGraph graph;
    
    // Test coherence with non-existent frames
    float coherence = graph.coherence(0, 1);
    ASSERT(coherence == 0.0f, "Coherence should be zero for non-existent frames");
    
    // Test find_node on empty graph
    auto* node = graph.find_node(0);
    ASSERT(node == nullptr, "find_node should return nullptr for empty graph");
    
    // Test recent frames on empty graph
    auto recent = graph.get_recent_frames(5);
    ASSERT(recent.empty(), "Recent frames should be empty for empty graph");
    
    // Test discontinuities on empty graph
    auto discontinuities = graph.find_discontinuities(0.5f);
    ASSERT(discontinuities.empty(), "Discontinuities should be empty for empty graph");
    
    std::cout << "✓ CoherenceGraph edge cases passed" << std::endl;
}

void run_coherence_graph_tests() {
    std::cout << "\n=== Running CoherenceGraph Tests ===" << std::endl;
    
    test_coherence_graph_basic();
    test_coherence_graph_coherence_computation();
    test_coherence_graph_discontinuities();
    test_coherence_graph_longest_subsequence();
    test_coherence_graph_recent_frames();
    test_coherence_graph_memory_limit();
    test_coherence_graph_clear();
    test_coherence_graph_edge_cases();
    
    std::cout << "✓ All CoherenceGraph tests passed!" << std::endl;
}