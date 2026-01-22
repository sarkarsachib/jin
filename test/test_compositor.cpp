#include "../src/scene/compositor.h"
#include "test_assertions.h"

namespace test_compositor {

void test_compositor_creation() {
    jin::scene::Compositor compositor;
    ASSERT_EQ(compositor.output_width, 1280);
    ASSERT_EQ(compositor.output_height, 720);
    ASSERT_EQ(compositor.frame_rate, 30.0f);
    ASSERT_EQ(compositor.quality, jin::scene::Compositor::QualityLevel::HIGH);
}

void test_compositor_configuration() {
    jin::scene::Compositor compositor;
    
    compositor.set_output_size(1920, 1080);
    ASSERT_EQ(compositor.output_width, 1920);
    ASSERT_EQ(compositor.output_height, 1080);
    
    compositor.set_frame_rate(60.0f);
    ASSERT_EQ(compositor.frame_rate, 60.0f);
    
    compositor.set_quality(jin::scene::Compositor::QualityLevel::ULTRA);
    ASSERT_EQ(compositor.quality, jin::scene::Compositor::QualityLevel::ULTRA);
}

void test_render_pass_creation() {
    jin::scene::RenderPass pass;
    pass.name = "test_pass";
    pass.pass_type = jin::scene::RenderPass::PassType::SPRITES;
    pass.enabled = true;
    pass.opacity = 0.8f;
    
    ASSERT_EQ(pass.name, "test_pass");
    ASSERT_EQ(pass.pass_type, jin::scene::RenderPass::PassType::SPRITES);
    ASSERT_TRUE(pass.enabled);
    ASSERT_EQ(pass.opacity, 0.8f);
}

void test_post_processing_settings() {
    jin::scene::Compositor compositor;
    
    // Test bloom settings
    compositor.post_processing.bloom_enabled = true;
    compositor.post_processing.bloom_intensity = 0.7f;
    ASSERT_TRUE(compositor.post_processing.bloom_enabled);
    ASSERT_EQ(compositor.post_processing.bloom_intensity, 0.7f);
    
    // Test color grading settings
    compositor.post_processing.color_grading_enabled = true;
    compositor.post_processing.exposure = jin::Tensor({1.2f, 1.1f, 0.9f});
    ASSERT_TRUE(compositor.post_processing.color_grading_enabled);
}

void test_compositor_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.emotion_embedding = jin::Tensor({0.8f, 0.7f, 0.9f});
    
    jin::scene::Compositor compositor;
    compositor.set_frame_state(frame_state);
    
    auto retrieved = compositor.get_frame_state();
    ASSERT_EQ(retrieved.emotion_embedding.size(), 3);
}

void test_performance_monitoring() {
    jin::scene::Compositor compositor;
    
    auto metrics = compositor.get_performance_metrics();
    ASSERT_EQ(metrics.frames_rendered, 0);
    ASSERT_EQ(metrics.total_render_time, 0.0f);
    
    compositor.reset_performance_metrics();
    auto reset_metrics = compositor.get_performance_metrics();
    ASSERT_EQ(reset_metrics.frames_rendered, 0);
}

void test_render_pass_management() {
    jin::scene::Compositor compositor;
    
    jin::scene::RenderPass custom_pass;
    custom_pass.name = "custom_pass";
    custom_pass.pass_type = jin::scene::RenderPass::PassType::EFFECTS;
    
    compositor.add_render_pass(custom_pass);
    ASSERT_EQ(compositor.render_passes.size(), 6); // Default passes + custom
    
    compositor.remove_render_pass("custom_pass");
    ASSERT_EQ(compositor.render_passes.size(), 5); // Back to default
    
    compositor.enable_pass("background", false);
    // Would need to check if pass is disabled
}

void test_compositor_settings() {
    jin::scene::Compositor compositor;
    
    auto settings = compositor.get_post_processing_settings();
    ASSERT_EQ(settings.bloom_enabled, true);
    
    jin::scene::Compositor::PostProcessingSettings new_settings;
    new_settings.bloom_enabled = false;
    new_settings.gamma = 2.4f;
    
    compositor.set_post_processing_settings(new_settings);
    auto updated_settings = compositor.get_post_processing_settings();
    ASSERT_EQ(updated_settings.bloom_enabled, false);
    ASSERT_EQ(updated_settings.gamma, 2.4f);
}

void test_real_time_compositor_creation() {
    jin::scene::RealTimeCompositor real_time_compositor;
    ASSERT_EQ(real_time_compositor.target_frame_time, 16.67f);
    ASSERT_TRUE(real_time_compositor.use_frame_prediction);
    ASSERT_TRUE(real_time_compositor.use_adaptive_quality);
}

void test_real_time_compositor_settings() {
    jin::scene::RealTimeCompositor real_time_compositor;
    
    real_time_compositor.set_target_frame_time(33.33f); // 30 FPS
    ASSERT_EQ(real_time_compositor.target_frame_time, 33.33f);
    
    real_time_compositor.enable_adaptive_quality(false);
    ASSERT_TRUE(!real_time_compositor.use_adaptive_quality);
    
    real_time_compositor.enable_frame_prediction(false);
    ASSERT_TRUE(!real_time_compositor.use_frame_prediction);
}

void test_offline_compositor_creation() {
    jin::scene::OfflineCompositor offline_compositor;
    ASSERT_EQ(offline_compositor.samples_per_pixel, 16);
    ASSERT_EQ(offline_compositor.max_bounces, 4);
    ASSERT_TRUE(!offline_compositor.use_ray_tracing);
}

void test_offline_compositor_settings() {
    jin::scene::OfflineCompositor offline_compositor;
    
    offline_compositor.set_samples_per_pixel(64);
    ASSERT_EQ(offline_compositor.samples_per_pixel, 64);
    
    offline_compositor.set_max_light_bounces(8);
    ASSERT_EQ(offline_compositor.max_bounces, 8);
    
    offline_compositor.enable_ray_tracing_effects(true);
    ASSERT_TRUE(offline_compositor.use_ray_tracing);
}

void test_quality_levels() {
    jin::scene::Compositor compositor;
    
    compositor.set_quality(jin::scene::Compositor::QualityLevel::LOW);
    ASSERT_EQ(compositor.quality, jin::scene::Compositor::QualityLevel::LOW);
    
    compositor.set_quality(jin::scene::Compositor::QualityLevel::MEDIUM);
    ASSERT_EQ(compositor.quality, jin::scene::Compositor::QualityLevel::MEDIUM);
    
    compositor.set_quality(jin::scene::Compositor::QualityLevel::HIGH);
    ASSERT_EQ(compositor.quality, jin::scene::Compositor::QualityLevel::HIGH);
    
    compositor.set_quality(jin::scene::Compositor::QualityLevel::ULTRA);
    ASSERT_EQ(compositor.quality, jin::scene::Compositor::QualityLevel::ULTRA);
}

void test_debug_and_profiling() {
    jin::scene::Compositor compositor;
    
    compositor.enable_profiling(true);
    // Would need to check internal profiling state
    
    compositor.set_debug_output(true);
    // Would need to check internal debug state
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("Compositor Creation", test_compositor_creation);
    TestRunner::instance().register_test("Compositor Configuration", test_compositor_configuration);
    TestRunner::instance().register_test("RenderPass Creation", test_render_pass_creation);
    TestRunner::instance().register_test("PostProcessing Settings", test_post_processing_settings);
    TestRunner::instance().register_test("Frame State Integration", test_compositor_frame_state_integration);
    TestRunner::instance().register_test("Performance Monitoring", test_performance_monitoring);
    TestRunner::instance().register_test("RenderPass Management", test_render_pass_management);
    TestRunner::instance().register_test("Compositor Settings", test_compositor_settings);
    TestRunner::instance().register_test("RealTimeCompositor Creation", test_real_time_compositor_creation);
    TestRunner::instance().register_test("RealTimeCompositor Settings", test_real_time_compositor_settings);
    TestRunner::instance().register_test("OfflineCompositor Creation", test_offline_compositor_creation);
    TestRunner::instance().register_test("OfflineCompositor Settings", test_offline_compositor_settings);
    TestRunner::instance().register_test("Quality Levels", test_quality_levels);
    TestRunner::instance().register_test("Debug and Profiling", test_debug_and_profiling);
}

} // namespace test_compositor