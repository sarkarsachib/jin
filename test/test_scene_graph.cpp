#include "../src/scene/scene_graph.h"
#include "test_assertions.h"

namespace test_scene_graph {

void test_scene_node_creation() {
    auto node = std::make_shared<jin::scene::SceneNode>();
    ASSERT_TRUE(node != nullptr);
    ASSERT_EQ(node->name, "");
    ASSERT_TRUE(node->visible);
    ASSERT_EQ(node->z_order, 0);
}

void test_scene_node_transform() {
    auto node = std::make_shared<jin::scene::SceneNode>();
    node->name = "test_node";
    
    // Test position setting
    jin::Tensor position({10.0f, 20.0f});
    node->set_local_position(position);
    
    auto transform = node->get_world_transform();
    ASSERT_TRUE(transform.position.size() >= 2);
}

void test_scene_node_hierarchy() {
    auto parent = std::make_shared<jin::scene::SceneNode>();
    auto child = std::make_shared<jin::scene::SceneNode>();
    
    parent->name = "parent";
    child->name = "child";
    
    parent->add_child(child);
    
    auto found_child = parent->find_child("child");
    ASSERT_TRUE(found_child != nullptr);
    ASSERT_EQ(found_child->name, "child");
}

void test_scene_graph_creation() {
    jin::scene::SceneGraph graph;
    ASSERT_TRUE(graph.root != nullptr);
    ASSERT_EQ(graph.root->name, "root");
}

void test_scene_graph_node_management() {
    jin::scene::SceneGraph graph;
    
    auto node = std::make_shared<jin::scene::SceneNode>();
    graph.add_node("test_node", node);
    
    auto found = graph.find_node("test_node");
    ASSERT_TRUE(found != nullptr);
    ASSERT_EQ(found->name, "test_node");
}

void test_camera2d_creation() {
    jin::scene::Camera2D camera;
    ASSERT_TRUE(camera.position.size() >= 2);
    ASSERT_EQ(camera.zoom, 1.0f);
    ASSERT_EQ(camera.rotation, 0.0f);
}

void test_camera2d_transform() {
    jin::scene::Camera2D camera;
    camera.position = jin::Tensor({100.0f, 200.0f});
    camera.zoom = 2.0f;
    
    jin::Tensor world_point({10.0f, 20.0f});
    auto screen_point = camera.world_to_screen(world_point);
    
    ASSERT_TRUE(screen_point.size() >= 2);
}

void test_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.frame_index = 42;
    frame_state.drift_score = 0.1f;
    
    auto node = std::make_shared<jin::scene::SceneNode>();
    node->set_frame_state(frame_state);
    
    auto retrieved = node->get_frame_state();
    ASSERT_EQ(retrieved.frame_index, 42);
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("SceneNode Creation", test_scene_node_creation);
    TestRunner::instance().register_test("SceneNode Transform", test_scene_node_transform);
    TestRunner::instance().register_test("SceneNode Hierarchy", test_scene_node_hierarchy);
    TestRunner::instance().register_test("SceneGraph Creation", test_scene_graph_creation);
    TestRunner::instance().register_test("SceneGraph Node Management", test_scene_graph_node_management);
    TestRunner::instance().register_test("Camera2D Creation", test_camera2d_creation);
    TestRunner::instance().register_test("Camera2D Transform", test_camera2d_transform);
    TestRunner::instance().register_test("Frame State Integration", test_frame_state_integration);
}

} // namespace test_scene_graph