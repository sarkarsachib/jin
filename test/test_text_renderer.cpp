#include "../src/scene/text_renderer.h"
#include "test_assertions.h"

namespace test_text_renderer {

void test_text_renderer_creation() {
    jin::scene::TextRenderer text_renderer;
    ASSERT_EQ(text_renderer.name, "text_renderer");
    ASSERT_EQ(text_renderer.text, "");
    ASSERT_EQ(text_renderer.alignment, jin::scene::TextRenderer::Alignment::LEFT);
}

void test_text_renderer_with_text() {
    jin::scene::TextRenderer text_renderer("Hello World");
    ASSERT_EQ(text_renderer.text, "Hello World");
}

void test_text_content_management() {
    jin::scene::TextRenderer text_renderer;
    text_renderer.set_text("New Text");
    ASSERT_EQ(text_renderer.text, "New Text");
    ASSERT_EQ(text_renderer.get_text(), "New Text");
}

void test_text_appearance_properties() {
    jin::scene::TextRenderer text_renderer;
    
    text_renderer.set_alignment(jin::scene::TextRenderer::Alignment::CENTER);
    ASSERT_EQ(text_renderer.alignment, jin::scene::TextRenderer::Alignment::CENTER);
    
    jin::Tensor color({1.0f, 0.0f, 0.0f, 1.0f});
    text_renderer.set_color(color);
    ASSERT_EQ(text_renderer.color.size(), 4);
}

void test_text_layout_properties() {
    jin::scene::TextRenderer text_renderer;
    
    text_renderer.set_wrap_width(100.0f);
    text_renderer.set_line_spacing(1.5f);
    text_renderer.set_letter_spacing(2.0f);
    
    ASSERT_EQ(text_renderer.wrap_width, 100.0f);
    ASSERT_EQ(text_renderer.line_spacing, 1.5f);
    ASSERT_EQ(text_renderer.letter_spacing, 2.0f);
}

void test_text_effects() {
    jin::scene::TextRenderer text_renderer;
    
    text_renderer.set_text_effect(jin::scene::TextRenderer::TextEffect::OUTLINE);
    ASSERT_EQ(text_renderer.text_effect, jin::scene::TextRenderer::TextEffect::OUTLINE);
    
    jin::Tensor outline_color({0.0f, 0.0f, 0.0f, 1.0f});
    text_renderer.set_outline(outline_color, 2.0f);
    ASSERT_EQ(text_renderer.outline_width, 2.0f);
    
    text_renderer.set_shadow(outline_color, jin::Tensor({3.0f, 3.0f}));
    ASSERT_EQ(text_renderer.text_effect, jin::scene::TextRenderer::TextEffect::SHADOW);
    
    text_renderer.set_glow(1.5f);
    ASSERT_EQ(text_renderer.text_effect, jin::scene::TextRenderer::TextEffect::GLOW);
}

void test_typewriter_animation() {
    jin::scene::TextRenderer text_renderer;
    text_renderer.set_text("Hello World");
    text_renderer.set_typewriter_mode(true);
    text_renderer.set_typewriter_speed(10.0f);
    
    ASSERT_TRUE(text_renderer.typewriter_mode);
    ASSERT_EQ(text_renderer.typewriter_speed, 10.0f);
    
    text_renderer.reset_typewriter();
    ASSERT_EQ(text_renderer.typewriter_position, 0);
    
    text_renderer.skip_typewriter();
    ASSERT_EQ(text_renderer.typewriter_position, 11); // Length of "Hello World"
}

void test_subtitle_renderer_creation() {
    jin::scene::SubtitleRenderer subtitle_renderer;
    ASSERT_EQ(subtitle_renderer.name, "subtitle_renderer");
    ASSERT_EQ(subtitle_renderer.position, jin::scene::SubtitleRenderer::SubtitlePosition::BOTTOM_CENTER);
    ASSERT_EQ(subtitle_renderer.current_state, jin::scene::SubtitleRenderer::SubtitleState::HIDDEN);
}

void test_subtitle_content_management() {
    jin::scene::SubtitleRenderer subtitle_renderer;
    
    subtitle_renderer.show_subtitle("Test Subtitle", 3.0f);
    ASSERT_EQ(subtitle_renderer.text, "Test Subtitle");
    ASSERT_EQ(subtitle_renderer.display_duration, 3.0f);
    
    subtitle_renderer.hide_subtitle();
    ASSERT_EQ(subtitle_renderer.current_state, jin::scene::SubtitleRenderer::SubtitleState::FADING_OUT);
}

void test_subtitle_positioning() {
    jin::scene::SubtitleRenderer subtitle_renderer;
    
    subtitle_renderer.set_position(jin::scene::SubtitleRenderer::SubtitlePosition::TOP_CENTER);
    ASSERT_EQ(subtitle_renderer.position, jin::scene::SubtitleRenderer::SubtitlePosition::TOP_CENTER);
    
    subtitle_renderer.set_position_offset(jin::Tensor({0.0f, -20.0f}));
    ASSERT_EQ(subtitle_renderer.position_offset.data_ptr<float>()[1], -20.0f);
}

void test_caption_renderer_creation() {
    jin::scene::CaptionRenderer caption_renderer;
    ASSERT_EQ(caption_renderer.name, "caption_renderer");
    ASSERT_TRUE(caption_renderer.show_background);
}

void test_caption_renderer_styling() {
    jin::scene::CaptionRenderer caption_renderer;
    
    jin::scene::CaptionRenderer::CaptionStyle style;
    style.background_color = jin::Tensor({0.0f, 0.0f, 0.0f, 0.8f});
    style.padding = 15.0f;
    style.border_radius = 8.0f;
    
    caption_renderer.set_style(style);
    ASSERT_TRUE(caption_renderer.show_background);
}

void test_dynamic_text_creation() {
    jin::scene::DynamicText dynamic_text;
    ASSERT_EQ(dynamic_text.name, "dynamic_text");
    ASSERT_EQ(dynamic_text.source, jin::scene::DynamicText::DataSource::STATIC);
    ASSERT_EQ(dynamic_text.update_interval, 1.0f);
}

void test_dynamic_text_source_management() {
    jin::scene::DynamicText dynamic_text;
    
    dynamic_text.set_source(jin::scene::DynamicText::DataSource::FILE_WATCHER, "/path/to/file.txt");
    ASSERT_EQ(dynamic_text.source, jin::scene::DynamicText::DataSource::FILE_WATCHER);
    ASSERT_EQ(dynamic_text.source_path, "/path/to/file.txt");
    
    dynamic_text.set_update_interval(5.0f);
    ASSERT_EQ(dynamic_text.update_interval, 5.0f);
}

void test_text_metrics() {
    jin::scene::TextRenderer text_renderer;
    text_renderer.set_text("Hello World");
    
    auto metrics = text_renderer.measure_text();
    ASSERT_TRUE(metrics.width >= 0);
    ASSERT_TRUE(metrics.height >= 0);
    ASSERT_EQ(metrics.characters, 11);
    
    ASSERT_EQ(text_renderer.get_character_count(), 11);
}

void test_font_management() {
    jin::scene::TextRenderer text_renderer;
    auto font = std::make_shared<jin::scene::Font>();
    font->name = "Test Font";
    font->size = 24;
    
    text_renderer.set_font(font);
    ASSERT_TRUE(text_renderer.font != nullptr);
    
    text_renderer.set_font_size(32);
    ASSERT_EQ(font->size, 32);
}

void test_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.emotion_embedding = jin::Tensor({0.7f, 0.8f, 0.9f});
    
    jin::scene::TextRenderer text_renderer;
    text_renderer.set_frame_state(frame_state);
    
    auto retrieved = text_renderer.get_frame_state();
    ASSERT_EQ(retrieved.emotion_embedding.size(), 3);
}

void test_auto_size() {
    jin::scene::TextRenderer text_renderer;
    text_renderer.set_text("Auto Size Test");
    text_renderer.set_auto_size(true);
    
    ASSERT_TRUE(text_renderer.auto_size);
}

void test_cache_management() {
    jin::scene::TextRenderer text_renderer;
    text_renderer.set_text("Test");
    
    text_renderer.refresh_cache();
    ASSERT_TRUE(text_renderer.needs_update);
    
    text_renderer.clear_cache();
    // Just ensure no crash
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("TextRenderer Creation", test_text_renderer_creation);
    TestRunner::instance().register_test("TextRenderer With Text", test_text_renderer_with_text);
    TestRunner::instance().register_test("Text Content Management", test_text_content_management);
    TestRunner::instance().register_test("Text Appearance Properties", test_text_appearance_properties);
    TestRunner::instance().register_test("Text Layout Properties", test_text_layout_properties);
    TestRunner::instance().register_test("Text Effects", test_text_effects);
    TestRunner::instance().register_test("Typewriter Animation", test_typewriter_animation);
    TestRunner::instance().register_test("SubtitleRenderer Creation", test_subtitle_renderer_creation);
    TestRunner::instance().register_test("Subtitle Content Management", test_subtitle_content_management);
    TestRunner::instance().register_test("Subtitle Positioning", test_subtitle_positioning);
    TestRunner::instance().register_test("CaptionRenderer Creation", test_caption_renderer_creation);
    TestRunner::instance().register_test("CaptionRenderer Styling", test_caption_renderer_styling);
    TestRunner::instance().register_test("DynamicText Creation", test_dynamic_text_creation);
    TestRunner::instance().register_test("DynamicText Source Management", test_dynamic_text_source_management);
    TestRunner::instance().register_test("Text Metrics", test_text_metrics);
    TestRunner::instance().register_test("Font Management", test_font_management);
    TestRunner::instance().register_test("Frame State Integration", test_frame_state_integration);
    TestRunner::instance().register_test("Auto Size", test_auto_size);
    TestRunner::instance().register_test("Cache Management", test_cache_management);
}

} // namespace test_text_renderer