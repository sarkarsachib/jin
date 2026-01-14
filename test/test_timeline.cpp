#include "../src/scene/timeline.h"
#include "test_assertions.h"

namespace test_timeline {

void test_timeline_creation() {
    jin::scene::Timeline timeline;
    ASSERT_EQ(timeline.current_frame, 0);
    ASSERT_EQ(timeline.frame_rate, 24.0f);
    ASSERT_TRUE(timeline.playing == false);
}

void test_timeline_event_management() {
    jin::scene::Timeline timeline;
    
    // Add an event
    std::map<std::string, std::string> params = {{"param1", "value1"}};
    timeline.add_event(10, "test_event", params);
    
    auto events = timeline.get_events_at_frame(10);
    ASSERT_EQ(events.size(), 1);
    ASSERT_EQ(events[0].event_type, "test_event");
}

void test_timeline_playback_control() {
    jin::scene::Timeline timeline;
    
    timeline.play();
    ASSERT_TRUE(timeline.playing);
    
    timeline.pause();
    ASSERT_TRUE(!timeline.playing);
    
    timeline.seek(50);
    ASSERT_EQ(timeline.current_frame, 50);
}

void test_animation_track_creation() {
    jin::scene::AnimationTrack track("position");
    ASSERT_EQ(track.property_name, "position");
    ASSERT_TRUE(track.keyframes.empty());
}

void test_animation_track_keyframes() {
    jin::scene::AnimationTrack track("position");
    
    jin::Tensor value({10.0f, 20.0f});
    track.add_keyframe(0, value);
    track.add_keyframe(10, jin::Tensor({30.0f, 40.0f}));
    
    ASSERT_EQ(track.keyframes.size(), 2);
}

void test_animation_track_interpolation() {
    jin::scene::AnimationTrack track("position");
    
    track.add_keyframe(0, jin::Tensor({0.0f, 0.0f}));
    track.add_keyframe(10, jin::Tensor({100.0f, 200.0f}));
    
    auto value = track.sample_value(5);
    ASSERT_TRUE(value.size() >= 2);
    
    // Should be halfway between keyframes
    ASSERT_NEAR(value.data_ptr<float>()[0], 50.0f, 1.0f);
}

void test_animation_creation() {
    jin::scene::Animation animation("test_anim", "target_node");
    ASSERT_EQ(animation.name, "test_anim");
    ASSERT_EQ(animation.target_node, "target_node");
    ASSERT_TRUE(animation.tracks.empty());
}

void test_animation_track_management() {
    jin::scene::Animation animation("test_anim", "target_node");
    
    auto track = animation.add_track("position");
    ASSERT_TRUE(track != nullptr);
    ASSERT_EQ(track->property_name, "position");
    
    auto retrieved = animation.get_track("position");
    ASSERT_TRUE(retrieved != nullptr);
}

void test_animation_controller() {
    jin::scene::AnimationController controller;
    ASSERT_TRUE(controller.animations.empty());
}

void test_animation_controller_integration() {
    jin::scene::AnimationController controller;
    
    auto animation = controller.add_animation("test_anim", "target_node");
    ASSERT_EQ(animation->name, "test_anim");
    ASSERT_EQ(controller.animations.size(), 1);
}

void test_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.frame_index = 100;
    frame_state.motion_smoothness = 0.9f;
    
    jin::scene::Timeline timeline;
    timeline.set_frame_state(frame_state);
    
    auto retrieved = timeline.get_frame_state();
    ASSERT_EQ(retrieved.frame_index, 100);
}

void test_serialization() {
    jin::scene::Timeline timeline;
    timeline.add_event(5, "test_event");
    timeline.add_event(10, "another_event");
    
    auto serialized = timeline.serialize();
    ASSERT_TRUE(!serialized.empty());
    
    jin::scene::Timeline new_timeline;
    new_timeline.deserialize(serialized);
    
    auto events = new_timeline.get_events_at_frame(5);
    ASSERT_EQ(events.size(), 1);
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("Timeline Creation", test_timeline_creation);
    TestRunner::instance().register_test("Timeline Event Management", test_timeline_event_management);
    TestRunner::instance().register_test("Timeline Playback Control", test_timeline_playback_control);
    TestRunner::instance().register_test("Animation Track Creation", test_animation_track_creation);
    TestRunner::instance().register_test("Animation Track Keyframes", test_animation_track_keyframes);
    TestRunner::instance().register_test("Animation Track Interpolation", test_animation_track_interpolation);
    TestRunner::instance().register_test("Animation Creation", test_animation_creation);
    TestRunner::instance().register_test("Animation Track Management", test_animation_track_management);
    TestRunner::instance().register_test("Animation Controller", test_animation_controller);
    TestRunner::instance().register_test("Animation Controller Integration", test_animation_controller_integration);
    TestRunner::instance().register_test("Frame State Integration", test_frame_state_integration);
    TestRunner::instance().register_test("Timeline Serialization", test_serialization);
}

} // namespace test_timeline