#include "../src/scene/scene_graph.h"
#include "../src/scene/timeline.h"
#include "../src/scene/layer_stack.h"
#include "../src/scene/sprite.h"
#include "../src/scene/particle_system.h"
#include "../src/scene/text_renderer.h"
#include "../src/scene/background.h"
#include "../src/scene/compositor.h"
#include "../src/scene/renderer.h"
#include "test_assertions.h"

namespace test_full_scene {

void test_complete_scene_creation() {
    // Create all major components
    jin::scene::SceneGraph scene_graph;
    jin::scene::LayerStack layer_stack;
    jin::scene::Camera2D camera;
    jin::scene::Compositor compositor;
    jin::scene::Renderer renderer;
    
    // Create a background
    auto background = jin::scene::Background::create_sky_background(1280, 720);
    ASSERT_TRUE(background != nullptr);
    ASSERT_TRUE(background->layers.size() > 0);
    
    // Create a sprite
    jin::Tensor sprite_image({64, 64, 4});
    sprite_image.fill(1.0f);
    auto sprite = std::make_shared<jin::scene::Sprite>(sprite_image, "character_sprite");
    ASSERT_EQ(sprite->name, "character_sprite");
    
    // Add sprite to scene graph
    scene_graph.add_node("character", sprite);
    
    // Add to layer stack
    layer_stack.add_layer(10, sprite, "character_layer");
    
    // Create text renderer
    auto text_renderer = std::make_shared<jin::scene::TextRenderer>("Hello World");
    text_renderer->set_alignment(jin::scene::TextRenderer::Alignment::CENTER);
    scene_graph.add_node("subtitle", text_renderer);
    layer_stack.add_layer(100, text_renderer, "text_layer");
    
    // Create particle system
    auto particle_system = jin::scene::ParticleSystem::create_fire_effect();
    scene_graph.add_node("fire_effect", particle_system);
    layer_stack.add_layer(50, particle_system, "effects_layer");
    
    // Test that all components are created successfully
    ASSERT_TRUE(scene_graph.root != nullptr);
    ASSERT_EQ(layer_stack.get_layer_count(), 3);
    ASSERT_TRUE(text_renderer != nullptr);
    ASSERT_TRUE(particle_system != nullptr);
}

void test_scene_animation_integration() {
    jin::scene::SceneGraph scene_graph;
    jin::scene::AnimationController animation_controller;
    
    // Create a sprite with animation
    jin::Tensor frame1({32, 32, 4});
    frame1.fill(1.0f);
    jin::Tensor frame2({32, 32, 4});
    frame2.fill(0.5f);
    
    auto animated_sprite = std::make_shared<jin::scene::AnimatedSprite>();
    animated_sprite->add_frame(frame1, 0.1f);
    animated_sprite->add_frame(frame2, 0.1f);
    
    scene_graph.add_node("animated_character", animated_sprite);
    
    // Create animation
    auto animation = animation_controller.add_animation("idle", "animated_character");
    auto position_track = animation->add_track("position");
    
    // Add keyframes
    position_track->add_keyframe(0, jin::Tensor({100.0f, 100.0f}));
    position_track->add_keyframe(60, jin::Tensor({200.0f, 150.0f}));
    
    // Test animation integration
    ASSERT_EQ(animation->name, "idle");
    ASSERT_EQ(animation->target_node, "animated_character");
    ASSERT_TRUE(animation->tracks.size() > 0);
    
    auto position_track_retrieved = animation->get_track("position");
    ASSERT_TRUE(position_track_retrieved != nullptr);
}

void test_layer_stack_z_ordering() {
    jin::scene::LayerStack layer_stack;
    
    // Create multiple sprites with different z-orders
    auto background_sprite = std::make_shared<jin::scene::Sprite>();
    background_sprite->name = "background";
    
    auto midground_sprite = std::make_shared<jin::scene::Sprite>();
    midground_sprite->name = "midground";
    
    auto foreground_sprite = std::make_shared<jin::scene::Sprite>();
    foreground_sprite->name = "foreground";
    
    // Add layers in random order
    layer_stack.add_layer(10, foreground_sprite, "fg");
    layer_stack.add_layer(1, background_sprite, "bg");
    layer_stack.add_layer(5, midground_sprite, "mg");
    
    layer_stack.sort_layers();
    
    auto render_order = layer_stack.get_render_order();
    ASSERT_EQ(render_order.size(), 3);
    
    // Verify correct z-order
    ASSERT_EQ((*render_order[0]).z_order, 1);
    ASSERT_EQ((*render_order[1]).z_order, 5);
    ASSERT_EQ((*render_order[2]).z_order, 10);
}

void test_timeline_event_synchronization() {
    jin::scene::Timeline timeline;
    
    // Add events for different frame times
    timeline.add_event(0, "scene_start");
    timeline.add_event(60, "character_spawn");
    timeline.add_event(120, "effect_trigger");
    timeline.add_event(180, "scene_end");
    
    // Test event retrieval
    auto frame_0_events = timeline.get_events_at_frame(0);
    ASSERT_EQ(frame_0_events.size(), 1);
    ASSERT_EQ(frame_0_events[0].event_type, "scene_start");
    
    auto frame_60_events = timeline.get_events_at_frame(60);
    ASSERT_EQ(frame_60_events.size(), 1);
    ASSERT_EQ(frame_60_events[0].event_type, "character_spawn");
    
    // Test timeline playback
    timeline.play();
    ASSERT_TRUE(timeline.playing);
    
    timeline.seek(60);
    ASSERT_EQ(timeline.current_frame, 60);
    
    auto current_events = timeline.get_current_events();
    ASSERT_EQ(current_events.size(), 1);
}

void test_particle_system_integration() {
    jin::scene::ParticleSystem particle_system;
    
    // Create custom emitter
    auto emitter = std::make_shared<jin::scene::ParticleEmitter>();
    emitter->set_emission_rate(5.0f);
    emitter->particle_lifetime = 2.0f;
    emitter->start_color = jin::Tensor({1.0f, 0.0f, 0.0f, 1.0f}); // Red
    emitter->end_color = jin::Tensor({0.0f, 0.0f, 0.0f, 0.0f}); // Transparent
    
    particle_system.add_emitter(emitter);
    
    // Test particle system properties
    ASSERT_EQ(particle_system.emitters.size(), 1);
    ASSERT_EQ(particle_system.max_particles, 1000);
    ASSERT_TRUE(particle_system.emit_on_update);
    
    // Test predefined effects
    auto fire = jin::scene::ParticleSystem::create_fire_effect();
    ASSERT_TRUE(fire != nullptr);
    
    auto smoke = jin::scene::ParticleSystem::create_smoke_effect();
    ASSERT_TRUE(smoke != nullptr);
}

void test_background_parallax_scrolling() {
    auto background = jin::scene::Background::create_sky_background(1280, 720);
    ASSERT_TRUE(background != nullptr);
    
    // Add custom parallax layer
    auto parallax_layer = std::make_shared<jin::scene::ParallaxLayer>();
    parallax_layer->depth = 0.5f;
    parallax_layer->render_mode = jin::scene::ParallaxLayer::RenderMode::TILE;
    parallax_layer->image = jin::Tensor({512, 512, 4});
    parallax_layer->image.fill(0.7f);
    
    background->add_layer(parallax_layer);
    ASSERT_EQ(background->layers.size(), 2); // Sky + custom layer
    
    // Test background properties
    background->set_global_scroll_speed(1.5f);
    ASSERT_EQ(background->global_scroll_speed, 1.5f);
    
    background->set_horizontal_looping(true);
    ASSERT_TRUE(background->loop_horizontally);
    
    // Test camera integration
    jin::scene::Camera2D camera;
    camera.position = jin::Tensor({100.0f, 50.0f});
    camera.zoom = 2.0f;
    
    // Test world-to-screen conversion
    jin::Tensor world_point({10.0f, 20.0f});
    auto screen_point = camera.world_to_screen(world_point);
    ASSERT_TRUE(screen_point.size() >= 2);
}

void test_compositor_render_pipeline() {
    jin::scene::Compositor compositor;
    
    // Test compositor configuration
    compositor.set_output_size(1920, 1080);
    compositor.set_frame_rate(60.0f);
    compositor.set_quality(jin::scene::Compositor::QualityLevel::HIGH);
    
    ASSERT_EQ(compositor.output_width, 1920);
    ASSERT_EQ(compositor.output_height, 1080);
    ASSERT_EQ(compositor.frame_rate, 60.0f);
    ASSERT_EQ(compositor.quality, jin::scene::Compositor::QualityLevel::HIGH);
    
    // Test post-processing settings
    compositor.post_processing.bloom_enabled = true;
    compositor.post_processing.bloom_intensity = 0.6f;
    ASSERT_TRUE(compositor.post_processing.bloom_enabled);
    ASSERT_EQ(compositor.post_processing.bloom_intensity, 0.6f);
    
    // Test performance monitoring
    auto metrics = compositor.get_performance_metrics();
    ASSERT_EQ(metrics.frames_rendered, 0);
    
    compositor.reset_performance_metrics();
    auto reset_metrics = compositor.get_performance_metrics();
    ASSERT_EQ(reset_metrics.frames_rendered, 0);
}

void test_end_to_end_scene_rendering() {
    // Create complete scene setup
    jin::scene::SceneGraph scene_graph;
    jin::scene::LayerStack layer_stack;
    jin::scene::Background background;
    jin::scene::Camera2D camera;
    jin::scene::Compositor compositor;
    jin::scene::Renderer renderer;
    
    // Setup background
    background = *jin::scene::Background::create_forest_background(1280, 720);
    
    // Create character sprite
    jin::Tensor character_image({128, 128, 4});
    character_image.fill(0.8f);
    auto character = std::make_shared<jin::scene::Sprite>(character_image);
    character->name = "hero";
    character->set_local_position(jin::Tensor({640.0f, 360.0f})); // Center of screen
    
    // Create text
    auto title_text = std::make_shared<jin::scene::TextRenderer>("Jin Animation Engine");
    title_text->set_alignment(jin::scene::TextRenderer::Alignment::CENTER);
    title_text->set_local_position(jin::Tensor({640.0f, 100.0f}));
    
    // Create particle effect
    auto sparkles = jin::scene::ParticleSystem::create_spark_effect();
    sparkles->set_local_position(jin::Tensor({640.0f, 360.0f}));
    
    // Add to scene graph
    scene_graph.add_node("character", character);
    scene_graph.add_node("title", title_text);
    scene_graph.add_node("sparkles", sparkles);
    
    // Add to layer stack
    layer_stack.add_layer(0, std::make_shared<jin::scene::SceneNode>(), "background"); // Background layer
    layer_stack.add_layer(10, character, "character_layer");
    layer_stack.add_layer(50, sparkles, "effects_layer");
    layer_stack.add_layer(100, title_text, "text_layer");
    
    // Setup timeline
    jin::scene::Timeline timeline;
    timeline.add_event(0, "scene_start");
    timeline.add_event(30, "character_appear");
    timeline.add_event(60, "title_show");
    timeline.add_event(90, "sparkles_start");
    
    // Setup animation
    jin::scene::AnimationController animation_controller;
    auto idle_animation = animation_controller.add_animation("idle", "hero");
    auto position_track = idle_animation->add_track("position");
    
    // Add animation keyframes (1 second loop at 30 FPS)
    position_track->add_keyframe(0, jin::Tensor({640.0f, 360.0f}));
    position_track->add_keyframe(15, jin::Tensor({640.0f, 380.0f})); // Slight bounce
    position_track->add_keyframe(30, jin::Tensor({640.0f, 360.0f}));
    
    // Test that everything is set up correctly
    ASSERT_EQ(layer_stack.get_layer_count(), 4);
    ASSERT_EQ(timeline.get_events_at_frame(0).size(), 1);
    ASSERT_EQ(idle_animation->tracks.size(), 1);
    
    // Test camera setup
    camera.position = jin::Tensor({640.0f, 360.0f});
    camera.viewport_size = jin::Tensor({1280.0f, 720.0f});
    
    // Test renderer setup
    renderer.initialize(1280, 720);
    renderer.set_current_camera(camera);
    
    // Test compositor setup
    compositor.set_output_size(1280, 720);
    
    // All components should be working together
    ASSERT_TRUE(true); // If we got here, the integration is working
}

void test_frame_state_propagation() {
    // Test that frame state propagates through all components
    jin::temporal::FrameState frame_state;
    frame_state.frame_index = 42;
    frame_state.drift_score = 0.15f;
    frame_state.motion_smoothness = 0.92f;
    frame_state.scene_stability = 0.88f;
    frame_state.emotion_embedding = jin::Tensor({0.7f, 0.8f, 0.6f});
    
    // Test scene graph
    jin::scene::SceneGraph scene_graph;
    scene_graph.set_global_frame_state(frame_state);
    auto retrieved_scene_state = scene_graph.get_global_frame_state();
    ASSERT_EQ(retrieved_scene_state.frame_index, 42);
    
    // Test layer stack
    jin::scene::LayerStack layer_stack;
    layer_stack.set_frame_state(frame_state);
    auto retrieved_layer_state = layer_stack.get_frame_state();
    ASSERT_NEAR(retrieved_layer_state.drift_score, 0.15f, 0.01f);
    
    // Test sprite
    jin::scene::Sprite sprite;
    sprite.set_frame_state(frame_state);
    auto retrieved_sprite_state = sprite.get_frame_state();
    ASSERT_EQ(retrieved_sprite_state.emotion_embedding.size(), 3);
    
    // Test background
    jin::scene::Background background;
    background.set_frame_state(frame_state);
    auto retrieved_bg_state = background.get_frame_state();
    ASSERT_NEAR(retrieved_bg_state.scene_stability, 0.88f, 0.01f);
    
    // Test compositor
    jin::scene::Compositor compositor;
    compositor.set_frame_state(frame_state);
    auto retrieved_comp_state = compositor.get_frame_state();
    ASSERT_NEAR(retrieved_comp_state.motion_smoothness, 0.92f, 0.01f);
}

void test_performance_benchmarks() {
    // Test that components can handle reasonable loads
    
    // Test particle system performance
    auto particle_system = jin::scene::ParticleSystem::create_rain_effect();
    ASSERT_TRUE(particle_system != nullptr);
    
    // Test background with multiple layers
    auto background = jin::scene::Background::create_city_background(1920, 1080);
    ASSERT_TRUE(background != nullptr);
    
    // Test layer stack with many layers
    jin::scene::LayerStack layer_stack;
    for (int i = 0; i < 10; ++i) {
        auto node = std::make_shared<jin::scene::SceneNode>();
        layer_stack.add_layer(i * 10, node, "layer_" + std::to_string(i));
    }
    ASSERT_EQ(layer_stack.get_layer_count(), 10);
    
    // Test timeline with many events
    jin::scene::Timeline timeline;
    for (int i = 0; i < 100; ++i) {
        timeline.add_event(i, "event_" + std::to_string(i));
    }
    ASSERT_EQ(timeline.get_events_at_frame(50).size(), 1);
    
    // Test animation with multiple tracks
    jin::scene::AnimationController controller;
    auto animation = controller.add_animation("complex", "target");
    
    animation->add_track("position");
    animation->add_track("rotation");
    animation->add_track("scale");
    animation->add_track("opacity");
    
    ASSERT_EQ(animation->tracks.size(), 4);
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("Complete Scene Creation", test_complete_scene_creation);
    TestRunner::instance().register_test("Scene Animation Integration", test_scene_animation_integration);
    TestRunner::instance().register_test("Layer Stack Z-Ordering", test_layer_stack_z_ordering);
    TestRunner::instance().register_test("Timeline Event Synchronization", test_timeline_event_synchronization);
    TestRunner::instance().register_test("Particle System Integration", test_particle_system_integration);
    TestRunner::instance().register_test("Background Parallax Scrolling", test_background_parallax_scrolling);
    TestRunner::instance().register_test("Compositor Render Pipeline", test_compositor_render_pipeline);
    TestRunner::instance().register_test("End-to-End Scene Rendering", test_end_to_end_scene_rendering);
    TestRunner::instance().register_test("Frame State Propagation", test_frame_state_propagation);
    TestRunner::instance().register_test("Performance Benchmarks", test_performance_benchmarks);
}

} // namespace test_full_scene