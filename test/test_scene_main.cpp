#include "test_scene_graph.cpp"
#include "test_timeline.cpp"
#include "test_layer_stack.cpp"
#include "test_sprite.cpp"
#include "test_particle_system.cpp"
#include "test_text_renderer.cpp"
#include "test_background.cpp"
#include "test_compositor.cpp"
#include "test_renderer.cpp"
#include "test_full_scene.cpp"
#include "test_assertions.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=== Scene Composition Engine Test Suite ===\n\n";
    
    auto& runner = jin::test::TestRunner::instance();
    
    // Register all scene tests
    test_scene_graph::register_tests();
    test_timeline::register_tests();
    test_layer_stack::register_tests();
    test_sprite::register_tests();
    test_particle_system::register_tests();
    test_text_renderer::register_tests();
    test_background::register_tests();
    test_compositor::register_tests();
    test_renderer::register_tests();
    test_full_scene::register_tests();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int failed = runner.run_all();
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== Test Results ===\n";
    std::cout << "Total tests: " << runner.test_count() << "\n";
    std::cout << "Failed: " << failed << "\n";
    std::cout << "Execution time: " << duration.count() << "ms\n";
    
    if (failed == 0) {
        std::cout << "\n✓ All tests passed!\n";
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed!\n";
        return 1;
    }
}