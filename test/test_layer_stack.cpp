#include "../src/scene/layer_stack.h"
#include "test_assertions.h"

namespace test_layer_stack {

void test_layer_creation() {
    auto node = std::make_shared<jin::scene::SceneNode>();
    jin::scene::Layer layer;
    layer.z_order = 5;
    layer.node = node;
    layer.visible = true;
    
    ASSERT_EQ(layer.z_order, 5);
    ASSERT_TRUE(layer.visible);
    ASSERT_EQ(layer.node, node);
}

void test_layer_stack_creation() {
    jin::scene::LayerStack stack;
    ASSERT_EQ(stack.get_layer_count(), 0);
}

void test_layer_stack_add_remove() {
    jin::scene::LayerStack stack;
    auto node = std::make_shared<jin::scene::SceneNode>();
    
    stack.add_layer(10, node, "test_layer");
    ASSERT_EQ(stack.get_layer_count(), 1);
    
    auto found = stack.get_layer_node("test_layer");
    ASSERT_TRUE(found != nullptr);
    
    stack.remove_layer("test_layer");
    ASSERT_EQ(stack.get_layer_count(), 0);
}

void test_layer_stack_sorting() {
    jin::scene::LayerStack stack;
    auto node1 = std::make_shared<jin::scene::SceneNode>();
    auto node2 = std::make_shared<jin::scene::SceneNode>();
    auto node3 = std::make_shared<jin::scene::SceneNode>();
    
    stack.add_layer(30, node3, "layer3");
    stack.add_layer(10, node1, "layer1");
    stack.add_layer(20, node2, "layer2");
    
    stack.sort_layers();
    
    auto render_order = stack.get_render_order();
    ASSERT_EQ(render_order.size(), 3);
    
    // Check that layers are sorted by z-order
    ASSERT_EQ((*render_order[0]).z_order, 10);
    ASSERT_EQ((*render_order[1]).z_order, 20);
    ASSERT_EQ((*render_order[2]).z_order, 30);
}

void test_layer_stack_visibility() {
    jin::scene::LayerStack stack;
    auto node = std::make_shared<jin::scene::SceneNode>();
    
    stack.add_layer(5, node, "visible_layer");
    stack.set_layer_visibility("visible_layer", false);
    
    auto visible_nodes = stack.get_visible_nodes();
    ASSERT_EQ(visible_nodes.size(), 0);
}

void test_layer_stack_opacity() {
    jin::scene::LayerStack stack;
    auto node = std::make_shared<jin::scene::SceneNode>();
    
    stack.add_layer(5, node, "opacity_layer");
    stack.set_layer_opacity("opacity_layer", 0.5f);
    
    auto layer = stack.get_layer("opacity_layer");
    ASSERT_TRUE(layer != nullptr);
    ASSERT_NEAR(layer->opacity, 0.5f, 0.01f);
}

void test_layer_range_queries() {
    jin::scene::LayerStack stack;
    auto node1 = std::make_shared<jin::scene::SceneNode>();
    auto node2 = std::make_shared<jin::scene::SceneNode>();
    auto node3 = std::make_shared<jin::scene::SceneNode>();
    
    stack.add_layer(10, node1, "layer1");
    stack.add_layer(20, node2, "layer2");
    stack.add_layer(30, node3, "layer3");
    
    auto layers_above = stack.get_layers_above(20);
    ASSERT_EQ(layers_above.size(), 1);
    ASSERT_EQ((*layers_above[0]).z_order, 30);
    
    auto layers_below = stack.get_layers_below(20);
    ASSERT_EQ(layers_below.size(), 1);
    ASSERT_EQ((*layers_below[0]).z_order, 10);
    
    auto layers_in_range = stack.get_layers_in_range(10, 25);
    ASSERT_EQ(layers_in_range.size(), 2);
}

void test_parallax_camera() {
    jin::scene::ParallaxCamera2D camera;
    camera.set_parallax_factor(5, 0.5f);
    
    ASSERT_NEAR(camera.get_parallax_factor(5), 0.5f, 0.01f);
    ASSERT_EQ(camera.get_parallax_factor(10), 1.0f); // Default factor
}

void test_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.scene_stability = 0.8f;
    
    jin::scene::LayerStack stack;
    stack.set_frame_state(frame_state);
    
    auto retrieved = stack.get_frame_state();
    ASSERT_NEAR(retrieved.scene_stability, 0.8f, 0.01f);
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("Layer Creation", test_layer_creation);
    TestRunner::instance().register_test("LayerStack Creation", test_layer_stack_creation);
    TestRunner::instance().register_test("LayerStack Add/Remove", test_layer_stack_add_remove);
    TestRunner::instance().register_test("LayerStack Sorting", test_layer_stack_sorting);
    TestRunner::instance().register_test("LayerStack Visibility", test_layer_stack_visibility);
    TestRunner::instance().register_test("LayerStack Opacity", test_layer_stack_opacity);
    TestRunner::instance().register_test("Layer Range Queries", test_layer_range_queries);
    TestRunner::instance().register_test("Parallax Camera", test_parallax_camera);
    TestRunner::instance().register_test("Frame State Integration", test_frame_state_integration);
}

} // namespace test_layer_stack