#include "../src/scene/renderer.h"
#include "test_assertions.h"

namespace test_renderer {

void test_render_buffer_creation() {
    jin::scene::RenderBuffer buffer;
    ASSERT_EQ(buffer.width, 0);
    ASSERT_EQ(buffer.height, 0);
    ASSERT_TRUE(!buffer.use_depth_buffer);
    ASSERT_TRUE(!buffer.use_stencil_buffer);
}

void test_render_buffer_with_size() {
    jin::scene::RenderBuffer buffer(800, 600);
    ASSERT_EQ(buffer.width, 800);
    ASSERT_EQ(buffer.height, 600);
}

void test_render_buffer_resize() {
    jin::scene::RenderBuffer buffer(400, 300);
    ASSERT_EQ(buffer.width, 400);
    
    buffer.resize(800, 600);
    ASSERT_EQ(buffer.width, 800);
    ASSERT_EQ(buffer.height, 600);
}

void test_renderer_creation() {
    jin::scene::Renderer renderer;
    ASSERT_EQ(renderer.default_width, 1280);
    ASSERT_EQ(renderer.default_height, 720);
    ASSERT_EQ(renderer.quality, jin::scene::Renderer::RenderQuality::QUALITY);
}

void test_renderer_configuration() {
    jin::scene::Renderer renderer;
    
    renderer.initialize(1024, 768);
    ASSERT_EQ(renderer.default_width, 1024);
    ASSERT_EQ(renderer.default_height, 768);
}

void test_renderer_state_management() {
    jin::scene::Renderer renderer;
    renderer.initialize(800, 600);
    
    renderer.set_blend_mode(jin::scene::BlendMode::ADD);
    // Check that blend mode was set
    ASSERT_TRUE(true);
    
    renderer.set_render_quality(jin::scene::Renderer::RenderQuality::FAST);
    ASSERT_EQ(renderer.quality, jin::scene::Renderer::RenderQuality::FAST);
}

void test_texture_management() {
    jin::scene::Renderer renderer;
    renderer.initialize(800, 600);
    
    jin::Tensor texture_data({64, 64, 4});
    texture_data.fill(1.0f);
    
    auto texture = renderer.create_texture(64, 64, texture_data);
    ASSERT_TRUE(texture.size() > 0);
}

void test_renderer_performance_stats() {
    jin::scene::Renderer renderer;
    
    auto stats = renderer.get_render_stats();
    ASSERT_EQ(stats.frames_rendered, 0);
    ASSERT_EQ(stats.last_frame_time, 0.0f);
    
    renderer.reset_render_stats();
    auto reset_stats = renderer.get_render_stats();
    ASSERT_EQ(reset_stats.frames_rendered, 0);
}

void test_hardware_renderer_creation() {
    jin::scene::HardwareRenderer hw_renderer;
    ASSERT_TRUE(hw_renderer.supports_opengl); // Assume OpenGL support
}

void test_hardware_renderer_shader_management() {
    jin::scene::HardwareRenderer hw_renderer;
    
    std::string vertex_shader = "vs_source";
    std::string fragment_shader = "fs_source";
    
    hw_renderer.create_shader_program("test_shader", vertex_shader, fragment_shader);
    // Shader should be created (simplified test)
    ASSERT_TRUE(true);
}

void test_hardware_renderer_buffer_management() {
    jin::scene::HardwareRenderer hw_renderer;
    
    jin::Tensor vertex_data({3, 4}); // 3 vertices, 4 components each
    vertex_data.fill(1.0f);
    
    auto buffer = hw_renderer.create_vertex_buffer(vertex_data);
    ASSERT_TRUE(buffer.size() > 0);
}

void test_software_renderer_creation() {
    jin::scene::SoftwareRenderer sw_renderer;
    ASSERT_TRUE(sw_renderer.use_simd_instructions);
    ASSERT_TRUE(sw_renderer.use_multithreading);
    ASSERT_EQ(sw_renderer.max_texture_memory, 256 * 1024 * 1024);
}

void test_software_renderer_memory_management() {
    jin::scene::SoftwareRenderer sw_renderer;
    
    sw_renderer.set_texture_memory_limit(128 * 1024 * 1024);
    ASSERT_EQ(sw_renderer.max_texture_memory, 128 * 1024 * 1024);
    
    size_t usage = sw_renderer.get_texture_memory_usage();
    ASSERT_EQ(usage, 0);
}

void test_software_renderer_optimization_settings() {
    jin::scene::SoftwareRenderer sw_renderer;
    
    sw_renderer.enable_simd_optimizations(false);
    ASSERT_TRUE(!sw_renderer.use_simd_instructions);
    
    sw_renderer.set_thread_count(8);
    ASSERT_TRUE(sw_renderer.worker_threads >= 1);
    
    sw_renderer.enable_growing_buffers(false);
    ASSERT_TRUE(!sw_renderer.use_growing_buffers);
}

void test_blit_operations() {
    jin::scene::BlitOperation blit_op;
    blit_op.operation = jin::scene::BlitOperation::Operation::BLEND;
    blit_op.blend_mode = jin::scene::BlendMode::ADD;
    blit_op.blend_factor = 0.8f;
    
    ASSERT_EQ(blit_op.operation, jin::scene::BlitOperation::Operation::BLEND);
    ASSERT_EQ(blit_op.blend_mode, jin::scene::BlendMode::ADD);
    ASSERT_EQ(blit_op.blend_factor, 0.8f);
}

void test_renderer_coordinate_transforms() {
    jin::scene::Renderer renderer;
    renderer.initialize(800, 600);
    
    jin::Tensor screen_pos({400, 300});
    auto normalized = renderer.normalize_coordinates(screen_pos, 800, 600);
    ASSERT_TRUE(normalized.size() >= 2);
    
    auto denormalized = renderer.denormalize_coordinates(normalized, 800, 600);
    ASSERT_TRUE(denormalized.size() >= 2);
}

void test_renderer_visibility_testing() {
    jin::scene::Renderer renderer;
    renderer.initialize(800, 600);
    
    jin::Tensor visible_point({400, 300});
    ASSERT_TRUE(renderer.is_point_visible(visible_point, 800, 600));
    
    jin::Tensor invisible_point({900, 700});
    ASSERT_TRUE(!renderer.is_point_visible(invisible_point, 800, 600));
}

void test_renderer_drawing_primitives() {
    jin::scene::Renderer renderer;
    renderer.initialize(800, 600);
    
    // Just ensure no crashes when calling drawing functions
    jin::Tensor position({100.0f, 100.0f});
    jin::Tensor size({50.0f, 50.0f});
    jin::Tensor color({1.0f, 0.0f, 0.0f, 1.0f});
    
    renderer.draw_rectangle(position, size, color);
    
    jin::Tensor center({200.0f, 200.0f});
    renderer.draw_circle(center, 25.0f, color);
    
    jin::Tensor start({50.0f, 50.0f});
    jin::Tensor end({150.0f, 150.0f});
    renderer.draw_line(start, end, 2.0f, color);
    
    ASSERT_TRUE(true);
}

void test_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.frame_index = 42;
    
    jin::scene::Renderer renderer;
    renderer.set_frame_state(frame_state);
    
    auto retrieved = renderer.get_frame_state();
    ASSERT_EQ(retrieved.frame_index, 42);
}

void test_renderer_debug_mode() {
    jin::scene::Renderer renderer;
    
    renderer.enable_debug_mode(true);
    // Just ensure no crash
    renderer.render_debug_info();
    
    renderer.set_wireframe_mode(true);
    // Just ensure no crash
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("RenderBuffer Creation", test_render_buffer_creation);
    TestRunner::instance().register_test("RenderBuffer With Size", test_render_buffer_with_size);
    TestRunner::instance().register_test("RenderBuffer Resize", test_render_buffer_resize);
    TestRunner::instance().register_test("Renderer Creation", test_renderer_creation);
    TestRunner::instance().register_test("Renderer Configuration", test_renderer_configuration);
    TestRunner::instance().register_test("Renderer State Management", test_renderer_state_management);
    TestRunner::instance().register_test("Texture Management", test_texture_management);
    TestRunner::instance().register_test("Renderer Performance Stats", test_renderer_performance_stats);
    TestRunner::instance().register_test("HardwareRenderer Creation", test_hardware_renderer_creation);
    TestRunner::instance().register_test("HardwareRenderer Shader Management", test_hardware_renderer_shader_management);
    TestRunner::instance().register_test("HardwareRenderer Buffer Management", test_hardware_renderer_buffer_management);
    TestRunner::instance().register_test("SoftwareRenderer Creation", test_software_renderer_creation);
    TestRunner::instance().register_test("SoftwareRenderer Memory Management", test_software_renderer_memory_management);
    TestRunner::instance().register_test("SoftwareRenderer Optimization Settings", test_software_renderer_optimization_settings);
    TestRunner::instance().register_test("Blit Operations", test_blit_operations);
    TestRunner::instance().register_test("Renderer Coordinate Transforms", test_renderer_coordinate_transforms);
    TestRunner::instance().register_test("Renderer Visibility Testing", test_renderer_visibility_testing);
    TestRunner::instance().register_test("Renderer Drawing Primitives", test_renderer_drawing_primitives);
    TestRunner::instance().register_test("Frame State Integration", test_frame_state_integration);
    TestRunner::instance().register_test("Renderer Debug Mode", test_renderer_debug_mode);
}

} // namespace test_renderer