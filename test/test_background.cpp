#include "../src/scene/background.h"
#include "test_assertions.h"

namespace test_background {

void test_parallax_layer_creation() {
    auto layer = std::make_shared<jin::scene::ParallaxLayer>();
    
    ASSERT_EQ(layer->depth, 1.0f);
    ASSERT_TRUE(layer->visible);
    ASSERT_EQ(layer->opacity, 1.0f);
    ASSERT_EQ(layer->render_mode, jin::scene::ParallaxLayer::RenderMode::TILE);
}

void test_background_creation() {
    jin::scene::Background background;
    ASSERT_EQ(background.name, "background");
    ASSERT_EQ(background.width, 1920);
    ASSERT_EQ(background.height, 1080);
    ASSERT_TRUE(background.follow_camera);
}

void test_background_layer_management() {
    jin::scene::Background background;
    auto layer = std::make_shared<jin::scene::ParallaxLayer>();
    layer->name = "test_layer";
    
    background.add_layer(layer);
    ASSERT_EQ(background.layers.size(), 1);
    
    auto retrieved = background.get_layer("test_layer");
    ASSERT_EQ(retrieved, layer);
    
    background.remove_layer("test_layer");
    ASSERT_EQ(background.layers.size(), 0);
}

void test_background_properties() {
    jin::scene::Background background;
    
    jin::Tensor color({0.5f, 0.5f, 0.8f, 1.0f});
    background.set_background_color(color);
    ASSERT_EQ(background.background_color.size(), 4);
    
    background.set_viewport_size(800, 600);
    ASSERT_EQ(background.viewport_size.data_ptr<float>()[0], 800.0f);
    
    background.set_dimensions(1024, 768);
    ASSERT_EQ(background.width, 1024);
    ASSERT_EQ(background.height, 768);
}

void test_camera_integration() {
    jin::scene::Background background;
    
    background.set_follow_camera(false);
    ASSERT_TRUE(!background.follow_camera);
    
    background.set_parallax_enabled(false);
    ASSERT_TRUE(!background.parallax_enabled);
    
    background.set_global_scroll_speed(2.0f);
    ASSERT_EQ(background.global_scroll_speed, 2.0f);
}

void test_looping_settings() {
    jin::scene::Background background;
    
    background.set_horizontal_looping(false);
    ASSERT_TRUE(!background.loop_horizontally);
    
    background.set_vertical_looping(false);
    ASSERT_TRUE(!background.loop_vertically);
}

void test_animated_background_creation() {
    jin::scene::AnimatedBackground animated_bg;
    ASSERT_EQ(animated_bg.name, "animated_background");
    ASSERT_EQ(animated_bg.animation_type, jin::scene::AnimatedBackground::AnimationType::DAY_NIGHT_CYCLE);
}

void test_animated_background_animation_control() {
    jin::scene::AnimatedBackground animated_bg;
    
    animated_bg.set_animation_type(jin::scene::AnimatedBackground::AnimationType::SEASONAL_CHANGE);
    ASSERT_EQ(animated_bg.animation_type, jin::scene::AnimatedBackground::AnimationType::SEASONAL_CHANGE);
    
    animated_bg.set_animation_speed(2.0f);
    ASSERT_EQ(animated_bg.animation_speed, 2.0f);
    
    animated_bg.set_animation_duration(60.0f);
    ASSERT_EQ(animated_bg.animation_duration, 60.0f);
    
    animated_bg.set_loop_animation(false);
    ASSERT_TRUE(!animated_bg.loop_animation);
}

void test_animated_background_time_control() {
    jin::scene::AnimatedBackground animated_bg;
    
    animated_bg.set_current_time(0.5f);
    ASSERT_EQ(animated_bg.current_time, 0.5f);
    
    animated_bg.advance_time(0.1f);
    ASSERT_NEAR(animated_bg.current_time, 0.6f, 0.01f);
}

void test_interactive_background_creation() {
    jin::scene::InteractiveBackground interactive_bg;
    ASSERT_EQ(interactive_bg.name, "interactive_background");
    ASSERT_TRUE(!interactive_bg.mouse_reactive);
    ASSERT_TRUE(!interactive_bg.audio_reactive);
    ASSERT_TRUE(!interactive_bg.physics_enabled);
}

void test_interactive_background_interaction() {
    jin::scene::InteractiveBackground interactive_bg;
    
    interactive_bg.set_mouse_reactive(true);
    ASSERT_TRUE(interactive_bg.mouse_reactive);
    
    interactive_bg.set_audio_reactive(true);
    ASSERT_TRUE(interactive_bg.audio_reactive);
    
    interactive_bg.set_physics_enabled(true);
    ASSERT_TRUE(interactive_bg.physics_enabled);
    
    interactive_bg.set_mouse_position(jin::Tensor({100.0f, 200.0f}));
    ASSERT_EQ(interactive_bg.mouse_position.data_ptr<float>()[0], 100.0f);
}

void test_interactive_background_mouse_settings() {
    jin::scene::InteractiveBackground interactive_bg;
    
    interactive_bg.set_mouse_influence(50.0f, 0.8f);
    ASSERT_EQ(interactive_bg.mouse_influence_radius, 50.0f);
    ASSERT_EQ(interactive_bg.mouse_influence_strength, 0.8f);
}

void test_interactive_background_audio_settings() {
    jin::scene::InteractiveBackground interactive_bg;
    
    interactive_bg.set_audio_level(0.7f);
    ASSERT_EQ(interactive_bg.audio_level, 0.7f);
    
    interactive_bg.set_audio_influence_strength(1.5f);
    ASSERT_EQ(interactive_bg.audio_influence_strength, 1.5f);
}

void test_interactive_background_physics_settings() {
    jin::scene::InteractiveBackground interactive_bg;
    
    interactive_bg.set_gravity_center(jin::Tensor({0.0f, 0.0f}));
    interactive_bg.set_gravity_strength(2.0f);
    interactive_bg.set_friction(0.9f);
    
    ASSERT_EQ(interactive_bg.gravity_strength, 2.0f);
    ASSERT_EQ(interactive_bg.friction, 0.9f);
}

void test_predefined_backgrounds() {
    auto sky_bg = jin::scene::Background::create_sky_background(800, 600);
    auto forest_bg = jin::scene::Background::create_forest_background(800, 600);
    auto city_bg = jin::scene::Background::create_city_background(800, 600);
    auto space_bg = jin::scene::Background::create_space_background(800, 600);
    auto underwater_bg = jin::scene::Background::create_underwater_background(800, 600);
    
    ASSERT_TRUE(sky_bg != nullptr);
    ASSERT_TRUE(forest_bg != nullptr);
    ASSERT_TRUE(city_bg != nullptr);
    ASSERT_TRUE(space_bg != nullptr);
    ASSERT_TRUE(underwater_bg != nullptr);
    
    // Check that they have layers
    ASSERT_TRUE(sky_bg->layers.size() > 0);
    ASSERT_TRUE(forest_bg->layers.size() > 0);
}

void test_layer_utilities() {
    jin::scene::Background background;
    auto layer1 = std::make_shared<jin::scene::ParallaxLayer>();
    auto layer2 = std::make_shared<jin::scene::ParallaxLayer>();
    
    layer1->name = "layer1";
    layer2->name = "layer2";
    
    background.add_layer(layer1);
    background.add_layer(layer2);
    
    background.set_all_layers_visible(false);
    ASSERT_TRUE(!layer1->visible);
    ASSERT_TRUE(!layer2->visible);
    
    background.set_layer_opacity("layer1", 0.5f);
    ASSERT_NEAR(layer1->opacity, 0.5f, 0.01f);
    
    background.set_layer_depth("layer2", 0.3f);
    ASSERT_NEAR(layer2->depth, 0.3f, 0.01f);
    
    background.set_layer_render_mode("layer1", jin::scene::ParallaxLayer::RenderMode::STRETCH);
    ASSERT_EQ(layer1->render_mode, jin::scene::ParallaxLayer::RenderMode::STRETCH);
}

void test_performance_optimization() {
    jin::scene::Background background;
    
    background.optimize_for_performance();
    ASSERT_TRUE(background.pre_render_layers);
    ASSERT_TRUE(background.use_layer_culling);
    
    background.pre_render_all_layers();
    // Just ensure no crash
}

void test_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.scene_stability = 0.85f;
    
    jin::scene::Background background;
    background.set_frame_state(frame_state);
    
    auto retrieved = background.get_frame_state();
    ASSERT_NEAR(retrieved.scene_stability, 0.85f, 0.01f);
}

void test_animation_setup_methods() {
    jin::scene::AnimatedBackground animated_bg;
    
    animated_bg.setup_day_night_cycle();
    ASSERT_EQ(animated_bg.animation_type, jin::scene::AnimatedBackground::AnimationType::DAY_NIGHT_CYCLE);
    
    animated_bg.setup_seasonal_change();
    ASSERT_EQ(animated_bg.animation_type, jin::scene::AnimatedBackground::AnimationType::SEASONAL_CHANGE);
    
    animated_bg.setup_weather_change();
    ASSERT_EQ(animated_bg.animation_type, jin::scene::AnimatedBackground::AnimationType::WEATHER_CHANGE);
    
    animated_bg.setup_time_lapse();
    ASSERT_EQ(animated_bg.animation_type, jin::scene::AnimatedBackground::AnimationType::TIME_LAPSE);
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("ParallaxLayer Creation", test_parallax_layer_creation);
    TestRunner::instance().register_test("Background Creation", test_background_creation);
    TestRunner::instance().register_test("Background Layer Management", test_background_layer_management);
    TestRunner::instance().register_test("Background Properties", test_background_properties);
    TestRunner::instance().register_test("Camera Integration", test_camera_integration);
    TestRunner::instance().register_test("Looping Settings", test_looping_settings);
    TestRunner::instance().register_test("AnimatedBackground Creation", test_animated_background_creation);
    TestRunner::instance().register_test("AnimatedBackground Animation Control", test_animated_background_animation_control);
    TestRunner::instance().register_test("AnimatedBackground Time Control", test_animated_background_time_control);
    TestRunner::instance().register_test("InteractiveBackground Creation", test_interactive_background_creation);
    TestRunner::instance().register_test("InteractiveBackground Interaction", test_interactive_background_interaction);
    TestRunner::instance().register_test("InteractiveBackground Mouse Settings", test_interactive_background_mouse_settings);
    TestRunner::instance().register_test("InteractiveBackground Audio Settings", test_interactive_background_audio_settings);
    TestRunner::instance().register_test("InteractiveBackground Physics Settings", test_interactive_background_physics_settings);
    TestRunner::instance().register_test("Predefined Backgrounds", test_predefined_backgrounds);
    TestRunner::instance().register_test("Layer Utilities", test_layer_utilities);
    TestRunner::instance().register_test("Performance Optimization", test_performance_optimization);
    TestRunner::instance().register_test("Frame State Integration", test_frame_state_integration);
    TestRunner::instance().register_test("Animation Setup Methods", test_animation_setup_methods);
}

} // namespace test_background