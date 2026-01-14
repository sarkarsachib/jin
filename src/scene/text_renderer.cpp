#include "text_renderer.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace jin {
namespace scene {

// TextRenderer implementation
TextRenderer::TextRenderer() {
    name = "text_renderer";
}

TextRenderer::TextRenderer(const std::string& initial_text) : TextRenderer() {
    text = initial_text;
}

TextRenderer::TextRenderer(const std::string& text, std::shared_ptr<Font> font) 
    : TextRenderer(text) {
    this->font = font;
}

void TextRenderer::set_text(const std::string& new_text) {
    if (text != new_text) {
        text = new_text;
        needs_update = true;
        invalidate_cache();
    }
}

void TextRenderer::set_font(std::shared_ptr<Font> new_font) {
    font = new_font;
    font_loaded_ = false;
    needs_update = true;
    invalidate_cache();
}

void TextRenderer::set_font_size(int size) {
    if (font) {
        font->size = size;
        needs_update = true;
        invalidate_cache();
    }
}

void TextRenderer::set_bold(bool bold) {
    if (font) {
        font->bold = bold;
        needs_update = true;
        invalidate_cache();
    }
}

void TextRenderer::set_italic(bool italic) {
    if (font) {
        font->italic = italic;
        needs_update = true;
        invalidate_cache();
    }
}

void TextRenderer::set_color(const Tensor& new_color) {
    color = new_color;
    needs_update = true;
}

void TextRenderer::set_alignment(Alignment new_alignment) {
    alignment = new_alignment;
    needs_update = true;
    invalidate_cache();
}

void TextRenderer::set_wrap_width(float width) {
    wrap_width = width;
    needs_update = true;
    invalidate_cache();
}

void TextRenderer::set_line_spacing(float spacing) {
    line_spacing = spacing;
    needs_update = true;
    invalidate_cache();
}

void TextRenderer::set_letter_spacing(float spacing) {
    letter_spacing = spacing;
    needs_update = true;
    invalidate_cache();
}

void TextRenderer::set_auto_size(bool auto_size_text) {
    auto_size = auto_size_text;
    needs_update = true;
    invalidate_cache();
}

void TextRenderer::set_text_effect(TextEffect effect) {
    text_effect = effect;
    needs_update = true;
}

void TextRenderer::set_outline(const Tensor& color, float width) {
    outline_color = color;
    outline_width = width;
    text_effect = TextEffect::OUTLINE;
    needs_update = true;
}

void TextRenderer::set_shadow(const Tensor& color, const Tensor& offset) {
    shadow_color = color;
    shadow_offset = offset;
    text_effect = TextEffect::SHADOW;
    needs_update = true;
}

void TextRenderer::set_glow(float intensity) {
    glow_intensity = intensity;
    text_effect = TextEffect::GLOW;
    needs_update = true;
}

void TextRenderer::set_typewriter_mode(bool enabled) {
    typewriter_mode = enabled;
    if (enabled) {
        reset_typewriter();
    }
    needs_update = true;
}

void TextRenderer::set_typewriter_speed(float speed) {
    typewriter_speed = std::max(1.0f, speed);
}

void TextRenderer::reset_typewriter() {
    typewriter_position = 0;
    typewriter_timer_ = 0.0f;
}

void TextRenderer::skip_typewriter() {
    typewriter_position = static_cast<int>(text.length());
}

void TextRenderer::play() {
    playing_ = true;
    if (typewriter_mode) {
        typewriter_timer_ = 0.0f;
    }
}

void TextRenderer::pause() {
    playing_ = false;
}

void TextRenderer::stop() {
    playing_ = false;
    reset_typewriter();
}

void TextRenderer::update(float delta_time) {
    SceneNode::update(delta_time);
    
    if (typewriter_mode && playing_) {
        typewriter_timer_ += delta_time;
        float chars_per_second = typewriter_speed;
        int new_position = static_cast<int>(text.length() * typewriter_timer_ * chars_per_second / text.length());
        new_position = std::min(new_position, static_cast<int>(text.length()));
        
        if (new_position > typewriter_position) {
            typewriter_position = new_position;
            needs_update = true;
        }
    }
}

Tensor TextRenderer::render(int width, int height, const Camera2D& camera) const {
    if (needs_update) {
        update_cache();
    }
    
    ensure_font_loaded();
    
    // Apply typewriter effect
    std::string render_text = text;
    if (typewriter_mode) {
        render_text = text.substr(0, typewriter_position);
    }
    
    // Render based on text effect
    switch (text_effect) {
        case TextEffect::NONE:
            return render_basic_text();
        case TextEffect::OUTLINE:
            return render_with_outline();
        case TextEffect::SHADOW:
            return render_with_shadow();
        case TextEffect::GLOW:
            return render_with_glow();
        case TextEffect::TYPEWRITER:
            return render_typewriter();
        default:
            return render_basic_text();
    }
}

void TextRenderer::set_frame_state(const temporal::FrameState& frame_state) {
    SceneNode::set_frame_state(frame_state);
    
    // Update text color based on emotion
    if (frame_state.emotion_embedding.size() >= 3) {
        float* emotion_data = frame_state.emotion_embedding.data_ptr<float>();
        Tensor emotion_color = Tensor({
            emotion_data[0], emotion_data[1], emotion_data[2], 1.0f
        });
        set_color(emotion_color);
    }
    
    needs_update = true;
}

TextRenderer::TextMetrics TextRenderer::measure_text() const {
    if (!metrics_valid_) {
        calculate_layout(const_cast<std::vector<std::string>&>(std::vector<std::string>()), 
                        const_cast<TextMetrics&>(cached_metrics_));
    }
    return cached_metrics_;
}

int TextRenderer::get_line_count() const {
    TextMetrics metrics = measure_text();
    return metrics.lines;
}

int TextRenderer::get_character_count() const {
    return static_cast<int>(text.length());
}

void TextRenderer::refresh_cache() {
    needs_update = true;
    invalidate_cache();
}

void TextRenderer::clear_cache() {
    cached_text_texture = Tensor();
    cached_metrics_ = TextMetrics();
    metrics_valid_ = false;
}

std::vector<std::string> TextRenderer::wrap_text(const std::string& text, float max_width) const {
    if (max_width <= 0.0f) {
        return {text};
    }
    
    std::vector<std::string> lines;
    std::istringstream text_stream(text);
    std::string word;
    std::string current_line;
    
    while (text_stream >> word) {
        std::string test_line = current_line.empty() ? word : current_line + " " + word;
        
        // Simple width estimation (would need actual font metrics in real implementation)
        float test_width = test_line.length() * (font ? font->size * 0.6f : 12.0f);
        
        if (test_width <= max_width || current_line.empty()) {
            current_line = test_line;
        } else {
            lines.push_back(current_line);
            current_line = word;
        }
    }
    
    if (!current_line.empty()) {
        lines.push_back(current_line);
    }
    
    return lines;
}

void TextRenderer::update_glyph_cache() {
    // This would load glyph data from font files
    // For now, just create simple placeholder glyphs
    for (char c : text) {
        if (glyph_cache_.find(c) == glyph_cache_.end()) {
            Glyph glyph;
            // Create a simple rectangular glyph (placeholder)
            glyph.texture = ones({font ? font->size : 24, (font ? font->size : 24) * 0.6f, 4});
            glyph.advance = (font ? font->size : 24) * 0.6f;
            glyph.width = static_cast<int>(glyph.advance);
            glyph.height = font ? font->size : 24;
            glyph_cache_[c] = glyph;
        }
    }
}

void TextRenderer::ensure_font_loaded() {
    if (!font_loaded_) {
        if (!font) {
            font = std::make_shared<Font>();
            font->name = "Default";
            font->size = 24;
        }
        update_glyph_cache();
        font_loaded_ = true;
    }
}

Glyph TextRenderer::get_glyph(char character) {
    ensure_font_loaded();
    
    auto it = glyph_cache_.find(character);
    if (it != glyph_cache_.end()) {
        return it->second;
    }
    
    // Create placeholder glyph if not found
    Glyph glyph;
    glyph.texture = ones({font->size, font->size * 0.6f, 4});
    glyph.advance = font->size * 0.6f;
    glyph.width = static_cast<int>(glyph.advance);
    glyph.height = font->size;
    
    return glyph;
}

Tensor TextRenderer::render_basic_text() const {
    if (text.empty()) {
        return zeros({font ? font->size : 24, 100, 4});
    }
    
    TextMetrics metrics = measure_text();
    Tensor text_texture = zeros({metrics.height, metrics.width, 4});
    
    // Simple text rendering (placeholder implementation)
    // In a real implementation, this would render actual glyphs
    
    return text_texture;
}

Tensor TextRenderer::render_with_outline() const {
    Tensor base_texture = render_basic_text();
    
    // Add outline effect (simplified)
    // In a real implementation, this would draw the outline
    
    return base_texture;
}

Tensor TextRenderer::render_with_shadow() const {
    Tensor base_texture = render_basic_text();
    
    // Add shadow effect (simplified)
    // In a real implementation, this would draw the shadow
    
    return base_texture;
}

Tensor TextRenderer::render_with_glow() const {
    Tensor base_texture = render_basic_text();
    
    // Add glow effect (simplified)
    // In a real implementation, this would apply a glow filter
    
    return base_texture;
}

Tensor TextRenderer::render_typewriter() const {
    // Render with typewriter cursor effect
    Tensor base_texture = render_basic_text();
    
    // Add cursor effect (simplified)
    // In a real implementation, this would draw a blinking cursor
    
    return base_texture;
}

void TextRenderer::calculate_layout(std::vector<std::string>& lines, TextMetrics& metrics) const {
    lines.clear();
    
    if (wrap_width > 0.0f) {
        lines = wrap_text(text, wrap_width);
    } else {
        lines = {text};
    }
    
    // Calculate metrics
    metrics.characters = static_cast<int>(text.length());
    metrics.lines = static_cast<int>(lines.size());
    metrics.width = 0;
    metrics.height = static_cast<int>(lines.size() * (font ? font->size * line_spacing : 24));
    
    for (const auto& line : lines) {
        float line_width = line.length() * (font ? font->size * 0.6f : 12.0f);
        metrics.width = std::max(metrics.width, static_cast<int>(line_width));
    }
}

void TextRenderer::apply_effects(Tensor& base_texture) const {
    // Apply text effects to the rendered texture
    // This would implement glow, outline, shadow, etc.
}

void TextRenderer::invalidate_cache() {
    metrics_valid_ = false;
}

void TextRenderer::update_cache() const {
    calculate_layout(const_cast<std::vector<std::string>&>(std::vector<std::string>()), 
                    const_cast<TextMetrics&>(cached_metrics_));
    metrics_valid_ = true;
}

// SubtitleRenderer implementation
SubtitleRenderer::SubtitleRenderer() : TextRenderer() {
    name = "subtitle_renderer";
}

void SubtitleRenderer::set_subtitle_text(const std::string& subtitle) {
    set_text(subtitle);
}

void SubtitleRenderer::show_subtitle(const std::string& text, float duration) {
    set_text(text);
    current_state = SubtitleState::FADING_IN;
    state_timer = 0.0f;
    
    if (duration > 0.0f) {
        display_duration = duration;
    }
}

void SubtitleRenderer::hide_subtitle() {
    current_state = SubtitleState::FADING_OUT;
    state_timer = 0.0f;
}

void SubtitleRenderer::update_subtitles(float delta_time) {
    update_state_machine(delta_time);
}

void SubtitleRenderer::set_position(SubtitlePosition new_position) {
    position = new_position;
}

void SubtitleRenderer::set_position_offset(const Tensor& offset) {
    position_offset = offset;
}

Tensor SubtitleRenderer::calculate_subtitle_position(int width, int height, const Camera2D& camera) const {
    Tensor position = Tensor({0.0f, 0.0f});
    
    // Convert position enum to actual position
    switch (this->position) {
        case SubtitlePosition::BOTTOM_CENTER:
            position = Tensor({width / 2.0f, 100.0f});
            break;
        case SubtitlePosition::TOP_CENTER:
            position = Tensor({width / 2.0f, height - 100.0f});
            break;
        case SubtitlePosition::BOTTOM_LEFT:
            position = Tensor({50.0f, 100.0f});
            break;
        case SubtitlePosition::BOTTOM_RIGHT:
            position = Tensor({width - 50.0f, 100.0f});
            break;
        case SubtitlePosition::CENTER:
            position = Tensor({width / 2.0f, height / 2.0f});
            break;
    }
    
    return position + position_offset;
}

Tensor SubtitleRenderer::render(int width, int height, const Camera2D& camera) const {
    Tensor base_texture = TextRenderer::render(width, height, camera);
    
    // Apply position
    Tensor subtitle_pos = calculate_subtitle_position(width, height, camera);
    
    // Apply fade effect
    float current_opacity = get_current_opacity();
    if (current_opacity < 1.0f) {
        // Apply opacity to texture (simplified)
        // In a real implementation, this would properly multiply the alpha channel
    }
    
    return base_texture;
}

void SubtitleRenderer::update_state_machine(float delta_time) {
    state_timer += delta_time;
    
    switch (current_state) {
        case SubtitleState::FADING_IN:
            if (state_timer >= fade_in_duration) {
                transition_to_state(SubtitleState::VISIBLE);
            }
            break;
        case SubtitleState::VISIBLE:
            if (state_timer >= display_duration) {
                transition_to_state(SubtitleState::FADING_OUT);
            }
            break;
        case SubtitleState::FADING_OUT:
            if (state_timer >= fade_out_duration) {
                transition_to_state(SubtitleState::HIDDEN);
            }
            break;
        case SubtitleState::HIDDEN:
            // Do nothing
            break;
    }
}

void SubtitleRenderer::transition_to_state(SubtitleState new_state) {
    current_state = new_state;
    state_timer = 0.0f;
}

float SubtitleRenderer::get_current_opacity() const {
    switch (current_state) {
        case SubtitleState::FADING_IN:
            return std::min(1.0f, state_timer / fade_in_duration);
        case SubtitleState::VISIBLE:
            return 1.0f;
        case SubtitleState::FADING_OUT:
            return std::max(0.0f, 1.0f - (state_timer / fade_out_duration));
        case SubtitleState::HIDDEN:
            return 0.0f;
        default:
            return 0.0f;
    }
}

// CaptionRenderer implementation
CaptionRenderer::CaptionRenderer() : TextRenderer() {
    name = "caption_renderer";
}

void CaptionRenderer::set_style(const CaptionStyle& new_style) {
    style = new_style;
    needs_update = true;
}

Tensor CaptionRenderer::render_with_background(int width, int height, const Camera2D& camera) {
    Tensor text_texture = TextRenderer::render(width, height, camera);
    TextMetrics metrics = measure_text();
    
    if (show_background) {
        return render_background(text_texture, metrics);
    }
    
    return text_texture;
}

Tensor CaptionRenderer::render_background(const Tensor& text_texture, const TextMetrics& metrics) const {
    // Create background texture
    Tensor background = zeros({
        metrics.height + style.padding * 2,
        metrics.width + style.padding * 2,
        4
    });
    
    // Fill background color (simplified)
    float* bg_data = background.data_ptr<float>();
    for (size_t i = 0; i < background.size(); i += 4) {
        bg_data[i] = style.background_color.data_ptr<float>()[0];
        bg_data[i + 1] = style.background_color.data_ptr<float>()[1];
        bg_data[i + 2] = style.background_color.data_ptr<float>()[2];
        bg_data[i + 3] = style.background_color.data_ptr<float>()[3];
    }
    
    // Add borders if needed
    if (style.border_width > 0.0f) {
        background = add_borders(background, style);
    }
    
    // Composite text over background (simplified)
    return background + text_texture;
}

Tensor CaptionRenderer::add_borders(const Tensor& background, const CaptionStyle& style) const {
    // Simplified border rendering
    // In a real implementation, this would draw proper rounded borders
    return background;
}

// DynamicText implementation
DynamicText::DynamicText() : TextRenderer() {
    name = "dynamic_text";
}

DynamicText::DynamicText(const std::string& initial_text) : TextRenderer(initial_text) {
    name = "dynamic_text";
}

void DynamicText::set_source(DataSource source_type, const std::string& path) {
    source = source_type;
    source_path = path;
}

void DynamicText::set_update_interval(float interval) {
    update_interval = std::max(0.1f, interval);
}

void DynamicText::watch_file(const std::string& file_path) {
    set_source(DataSource::FILE_WATCHER, file_path);
}

void DynamicText::stop_watching_file() {
    source_path.clear();
}

void DynamicText::fetch_from_url(const std::string& url) {
    set_source(DataSource::NETWORK, url);
}

void DynamicText::set_user_agent(const std::string& agent) {
    user_agent = agent;
}

void DynamicText::set_timeout(int seconds) {
    timeout_seconds = std::max(1, seconds);
}

void DynamicText::set_script_command(const std::string& command) {
    script_command = command;
    execute_on_update = !command.empty();
}

void DynamicText::execute_script() {
    if (execute_script_command()) {
        refresh();
    }
}

void DynamicText::update(float delta_time) {
    TextRenderer::update(delta_time);
    
    update_timer_ += delta_time;
    
    if (update_timer_ >= update_interval) {
        update_timer_ = 0.0f;
        update_from_source();
    }
}

void DynamicText::refresh() {
    update_from_source();
}

void DynamicText::update_from_source() {
    bool success = false;
    
    switch (source) {
        case DataSource::FILE_WATCHER:
            success = load_from_file();
            break;
        case DataSource::NETWORK:
            success = fetch_from_network();
            break;
        case DataSource::SCRIPT:
            if (execute_on_update) {
                success = execute_script_command();
            }
            break;
        case DataSource::STATIC:
        default:
            break;
    }
    
    if (success && is_content_changed(text)) {
        needs_update = true;
        invalidate_cache();
    }
}

bool DynamicText::load_from_file() {
    std::ifstream file(source_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    text = trim_whitespace(buffer.str());
    file.close();
    
    return true;
}

bool DynamicText::fetch_from_network() {
    // Simplified network fetching (would need proper HTTP client)
    // For now, just return false
    return false;
}

bool DynamicText::execute_script_command() {
    // Simplified script execution (would need proper process handling)
    // For now, just return false
    return false;
}

std::string DynamicText::trim_whitespace(const std::string& str) const {
    size_t start = 0;
    size_t end = str.length();
    
    while (start < end && std::isspace(str[start])) {
        start++;
    }
    
    while (end > start && std::isspace(str[end - 1])) {
        end--;
    }
    
    return str.substr(start, end - start);
}

bool DynamicText::is_content_changed(const std::string& new_content) const {
    return text != new_content;
}

} // namespace scene
} // namespace jin