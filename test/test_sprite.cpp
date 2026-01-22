#include "../src/scene/sprite.h"
#include "test_assertions.h"

namespace test_sprite {

void test_sprite_creation() {
    jin::scene::Sprite sprite;
    ASSERT_EQ(sprite.name, "sprite");
    ASSERT_EQ(sprite.blend_mode, jin::scene::BlendMode::NORMAL);
    ASSERT_EQ(sprite.opacity, 1.0f);
    ASSERT_TRUE(sprite.use_alpha);
}

void test_sprite_with_image() {
    jin::Tensor image({100, 100, 4});
    image.fill(1.0f);
    
    jin::scene::Sprite sprite(image);
    ASSERT_EQ(sprite.get_image().size(), image.size());
}

void test_sprite_image_management() {
    jin::scene::Sprite sprite;
    
    jin::Tensor image({50, 50, 4});
    image.fill(0.5f);
    
    sprite.set_image(image);
    ASSERT_EQ(sprite.get_image().size(), image.size());
}

void test_sprite_transform_properties() {
    jin::scene::Sprite sprite;
    
    sprite.set_blend_mode(jin::scene::BlendMode::ADD);
    ASSERT_EQ(sprite.get_blend_mode(), jin::scene::BlendMode::ADD);
    
    sprite.set_opacity(0.7f);
    ASSERT_NEAR(sprite.get_opacity(), 0.7f, 0.01f);
    
    sprite.set_use_alpha(false);
    ASSERT_TRUE(!sprite.get_use_alpha());
}

void test_animated_sprite_creation() {
    jin::scene::AnimatedSprite animated_sprite;
    ASSERT_EQ(animated_sprite.name, "animated_sprite");
    ASSERT_EQ(animated_sprite.current_frame, 0);
    ASSERT_EQ(animated_sprite.frame_timer, 0.0f);
    ASSERT_TRUE(animated_sprite.is_playing == false);
}

void test_animated_sprite_frame_management() {
    jin::scene::AnimatedSprite animated_sprite;
    
    jin::Tensor frame1({32, 32, 4});
    frame1.fill(1.0f);
    
    jin::Tensor frame2({32, 32, 4});
    frame2.fill(0.5f);
    
    animated_sprite.add_frame(frame1, 0.1f, "frame1");
    animated_sprite.add_frame(frame2, 0.1f, "frame2");
    
    ASSERT_EQ(animated_sprite.get_frame_count(), 2);
}

void test_animated_sprite_playback() {
    jin::scene::AnimatedSprite animated_sprite;
    
    jin::Tensor frame1({32, 32, 4});
    frame1.fill(1.0f);
    
    animated_sprite.add_frame(frame1, 0.1f);
    animated_sprite.play();
    
    ASSERT_TRUE(animated_sprite.is_playing);
    
    animated_sprite.pause();
    ASSERT_TRUE(!animated_sprite.is_playing);
}

void test_sprite_sheet_creation() {
    jin::scene::SpriteSheet sprite_sheet;
    ASSERT_EQ(sprite_sheet.name, "sprite_sheet");
    ASSERT_EQ(sprite_sheet.sheet_columns, 1);
    ASSERT_EQ(sprite_sheet.sheet_rows, 1);
}

void test_sprite_sheet_configuration() {
    jin::scene::SpriteSheet sprite_sheet;
    sprite_sheet.configure_sheet(4, 4, 32, 32);
    
    ASSERT_EQ(sprite_sheet.sheet_columns, 4);
    ASSERT_EQ(sprite_sheet.sheet_rows, 4);
    ASSERT_EQ(sprite_sheet.region_width, 32);
    ASSERT_EQ(sprite_sheet.region_height, 32);
}

void test_sprite_sheet_region_management() {
    jin::scene::SpriteSheet sprite_sheet;
    sprite_sheet.configure_sheet(2, 2, 50, 50);
    
    // Add regions
    sprite_sheet.add_region(0, 0, 50, 50, "region1");
    sprite_sheet.add_region(50, 0, 50, 50, "region2");
    
    ASSERT_EQ(sprite_sheet.sprite_regions.size(), 2);
    
    auto region = sprite_sheet.get_region("region1");
    ASSERT_TRUE(region != nullptr);
    ASSERT_EQ(region->name, "region1");
}

void test_character_sprite_creation() {
    jin::scene::CharacterSprite character_sprite;
    ASSERT_EQ(character_sprite.name, "character_sprite");
    ASSERT_EQ(character_sprite.current_state, jin::scene::CharacterSprite::IDLE);
}

void test_character_sprite_emotion_management() {
    jin::scene::CharacterSprite character_sprite;
    
    character_sprite.set_emotion_state("happy", 0.8f);
    ASSERT_TRUE(character_sprite.emotion_intensities_.find("happy") != character_sprite.emotion_intensities_.end());
}

void test_character_sprite_consistency() {
    jin::scene::CharacterSprite character_sprite;
    
    jin::Tensor identity({64});
    identity.fill(0.5f);
    
    character_sprite.set_character_consistency_data(identity);
    
    auto retrieved = character_sprite.get_character_consistency_data();
    ASSERT_EQ(retrieved.size(), identity.size());
}

void test_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.emotion_embedding = jin::Tensor({0.8f, 0.6f, 0.9f});
    
    jin::scene::Sprite sprite;
    sprite.set_frame_state(frame_state);
    
    auto retrieved = sprite.get_frame_state();
    ASSERT_EQ(retrieved.emotion_embedding.size(), 3);
}

void test_audio_reactive_properties() {
    jin::scene::Sprite sprite;
    
    sprite.set_audio_reactive_mode(true);
    sprite.update_audio_properties(0.7f, 0.5f);
    
    ASSERT_NEAR(sprite.last_audio_level_, 0.7f, 0.01f);
}

void test_sprite_flip_operations() {
    jin::scene::Sprite sprite;
    
    // These operations would modify the image data
    sprite.flip_horizontal();
    sprite.flip_vertical();
    sprite.rotate_90_degrees();
    
    // No assertions needed - just ensure no crashes
    ASSERT_TRUE(true);
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("Sprite Creation", test_sprite_creation);
    TestRunner::instance().register_test("Sprite With Image", test_sprite_with_image);
    TestRunner::instance().register_test("Sprite Image Management", test_sprite_image_management);
    TestRunner::instance().register_test("Sprite Transform Properties", test_sprite_transform_properties);
    TestRunner::instance().register_test("Animated Sprite Creation", test_animated_sprite_creation);
    TestRunner::instance().register_test("Animated Sprite Frame Management", test_animated_sprite_frame_management);
    TestRunner::instance().register_test("Animated Sprite Playback", test_animated_sprite_playback);
    TestRunner::instance().register_test("SpriteSheet Creation", test_sprite_sheet_creation);
    TestRunner::instance().register_test("SpriteSheet Configuration", test_sprite_sheet_configuration);
    TestRunner::instance().register_test("SpriteSheet Region Management", test_sprite_sheet_region_management);
    TestRunner::instance().register_test("Character Sprite Creation", test_character_sprite_creation);
    TestRunner::instance().register_test("Character Sprite Emotion Management", test_character_sprite_emotion_management);
    TestRunner::instance().register_test("Character Sprite Consistency", test_character_sprite_consistency);
    TestRunner::instance().register_test("Frame State Integration", test_frame_state_integration);
    TestRunner::instance().register_test("Audio Reactive Properties", test_audio_reactive_properties);
    TestRunner::instance().register_test("Sprite Flip Operations", test_sprite_flip_operations);
}

} // namespace test_sprite