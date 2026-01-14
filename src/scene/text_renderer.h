#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include "scene_graph.h"
#include "sprite.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace jin {
namespace scene {

/**
 * Font - Basic font definition and glyph management
 */
struct Font {
    std::string name;
    int size = 24;
    bool bold = false;
    bool italic = false;
    
    // Glyph data (simplified)
    std::map<char, Tensor> glyph_textures;  // Character -> texture data
    float line_height = 1.2f;                // Line height multiplier
    float letter_spacing = 0.0f;             // Additional letter spacing
};

/**
 * Glyph - Individual character glyph data
 */
struct Glyph {
    Tensor texture;          // Glyph texture [H, W, 4] RGBA
    float advance = 0.0f;     // Horizontal advance to next character
    float bearing_x = 0.0f;   // Horizontal bearing
    float bearing_y = 0.0f;   // Vertical bearing (baseline to top)
    int width = 0;           // Glyph width in pixels
    int height = 0;           // Glyph height in pixels
    
    // UV coordinates in font atlas (if using texture atlas)
    float u0 = 0.0f, v0 = 0.0f;  // Top-left UV
    float u1 = 1.0f, v1 = 1.0f;  // Bottom-right UV
};

/**
 * TextRenderer - Renders text and subtitles
 */
class TextRenderer : public SceneNode {
public:
    // Text content and properties
    std::string text;
    std::shared_ptr<Font> font;
    Tensor color = Tensor({1.0f, 1.0f, 1.0f, 1.0f});  // RGBA color
    
    // Layout properties
    enum class Alignment {
        LEFT,
        CENTER,
        RIGHT,
        JUSTIFY
    };
    
    Alignment alignment = Alignment::LEFT;
    float wrap_width = 0.0f;           // 0.0 = no wrapping
    float line_spacing = 1.0f;          // Multiplier for line height
    float letter_spacing = 0.0f;        // Additional letter spacing
    bool auto_size = false;             // Automatically size to fit content
    
    // Text effects
    enum class TextEffect {
        NONE,
        OUTLINE,
        SHADOW,
        GLOW,
        TYPEWRITER
    };
    
    TextEffect text_effect = TextEffect::NONE;
    Tensor outline_color = Tensor({0.0f, 0.0f, 0.0f, 1.0f}); // For outline effect
    float outline_width = 1.0f;                            // Outline thickness
    Tensor shadow_offset = Tensor({2.0f, 2.0f});           // Shadow offset
    Tensor shadow_color = Tensor({0.0f, 0.0f, 0.0f, 0.5f}); // Shadow color
    float glow_intensity = 1.0f;                          // Glow effect intensity
    
    // Animation properties
    bool typewriter_mode = false;
    float typewriter_speed = 50.0f;   // Characters per second
    int typewriter_position = 0;       // Current character position
    
    // Cached rendering
    Tensor cached_text_texture;       // Cached rendered text
    bool needs_update = true;
    
    TextRenderer();
    explicit TextRenderer(const std::string& initial_text);
    TextRenderer(const std::string& text, std::shared_ptr<Font> font);
    ~TextRenderer() = default;
    
    // Text content management
    void set_text(const std::string& new_text);
    std::string get_text() const { return text; }
    
    // Font and appearance
    void set_font(std::shared_ptr<Font> new_font);
    void set_font_size(int size);
    void set_bold(bool bold);
    void set_italic(bool italic);
    void set_color(const Tensor& new_color);
    void set_alignment(Alignment new_alignment);
    
    // Layout properties
    void set_wrap_width(float width);
    void set_line_spacing(float spacing);
    void set_letter_spacing(float spacing);
    void set_auto_size(bool auto_size_text);
    
    // Text effects
    void set_text_effect(TextEffect effect);
    void set_outline(const Tensor& color, float width);
    void set_shadow(const Tensor& color, const Tensor& offset);
    void set_glow(float intensity);
    
    // Typewriter animation
    void set_typewriter_mode(bool enabled);
    void set_typewriter_speed(float speed);
    void reset_typewriter();
    void skip_typewriter();
    
    // Animation control
    void play();
    void pause();
    void stop();
    void update(float delta_time) override;
    
    // Rendering
    Tensor render(int width, int height, const Camera2D& camera) const override;
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state) override;
    
    // Text measurement
    struct TextMetrics {
        int width = 0;          // Total text width
        int height = 0;         // Total text height
        int lines = 1;          // Number of lines
        int characters = 0;     // Character count
    };
    
    TextMetrics measure_text() const;
    int get_line_count() const;
    int get_character_count() const;
    
    // Utility methods
    void refresh_cache();
    void clear_cache();
    
private:
    // Glyph management
    std::map<char, Glyph> glyph_cache_;
    bool font_loaded_ = false;
    
    // Typewriter state
    bool playing_ = false;
    float typewriter_timer_ = 0.0f;
    
    // Cached metrics
    mutable TextMetrics cached_metrics_;
    mutable bool metrics_valid_ = false;
    
    // Text processing helpers
    std::vector<std::string> wrap_text(const std::string& text, float max_width) const;
    void update_glyph_cache();
    void ensure_font_loaded();
    Glyph get_glyph(char character);
    
    // Rendering helpers
    Tensor render_basic_text() const;
    Tensor render_with_outline() const;
    Tensor render_with_shadow() const;
    Tensor render_with_glow() const;
    Tensor render_typewriter() const;
    
    // Layout calculation
    void calculate_layout(std::vector<std::string>& lines, TextMetrics& metrics) const;
    void apply_effects(Tensor& base_texture) const;
    
    // Cached data
    void invalidate_cache();
    void update_cache() const;
};

/**
 * SubtitleRenderer - Specialized text renderer for subtitles
 */
class SubtitleRenderer : public TextRenderer {
public:
    // Subtitle-specific properties
    float fade_in_duration = 0.5f;    // Fade in duration in seconds
    float fade_out_duration = 0.5f;    // Fade out duration in seconds
    float display_duration = 3.0f;     // How long subtitle stays visible
    
    // Positioning
    enum class SubtitlePosition {
        BOTTOM_CENTER,
        TOP_CENTER,
        BOTTOM_LEFT,
        BOTTOM_RIGHT,
        CENTER
    };
    
    SubtitlePosition position = SubtitlePosition::BOTTOM_CENTER;
    Tensor position_offset = Tensor({0.0f, -50.0f});  // Offset from position
    
    // Subtitle content management
    void set_subtitle_text(const std::string& subtitle);
    void show_subtitle(const std::string& text, float duration = -1.0f);
    void hide_subtitle();
    void update_subtitles(float delta_time);
    
    // Animation states
    enum class SubtitleState {
        HIDDEN,
        FADING_IN,
        VISIBLE,
        FADING_OUT
    };
    
    SubtitleState current_state = SubtitleState::HIDDEN;
    float state_timer = 0.0f;
    
    // Positioning
    void set_position(SubtitlePosition new_position);
    void set_position_offset(const Tensor& offset);
    
    // Auto-positioning based on camera
    Tensor calculate_subtitle_position(int width, int height, const Camera2D& camera) const;
    
    // Rendering with subtitle positioning
    Tensor render(int width, int height, const Camera2D& camera) const override;
    
private:
    void update_state_machine(float delta_time);
    void transition_to_state(SubtitleState new_state);
    float get_current_opacity() const;
};

/**
 * CaptionRenderer - Text renderer for captions with styling
 */
class CaptionRenderer : public TextRenderer {
public:
    // Caption styling
    struct CaptionStyle {
        Tensor background_color = Tensor({0.0f, 0.0f, 0.0f, 0.7f});
        float padding = 10.0f;                    // Background padding
        float border_radius = 5.0f;                // Corner radius
        float border_width = 0.0f;                 // Border width
        Tensor border_color = Tensor({1.0f, 1.0f, 1.0f, 1.0f});
        bool drop_shadow = true;                   // Drop shadow on background
    };
    
    CaptionStyle style;
    bool show_background = true;
    
    // Caption management
    void set_style(const CaptionStyle& new_style);
    void set_show_background(bool show) { show_background = show; }
    
    // Styled rendering
    Tensor render_with_background(int width, int height, const Camera2D& camera) const;
    
private:
    Tensor render_background(const Tensor& text_texture, const TextMetrics& metrics) const;
    Tensor add_borders(const Tensor& background, const CaptionStyle& style) const;
};

/**
 * DynamicText - Text that can be updated dynamically
 */
class DynamicText : public TextRenderer {
public:
    // Dynamic content sources
    enum class DataSource {
        STATIC,
        FILE_WATCHER,     // Watch file for changes
        NETWORK,          // Fetch from network
        SCRIPT            // Execute script to get text
    };
    
    DataSource source = DataSource::STATIC;
    std::string source_path;           // File path or URL
    float update_interval = 1.0f;      // Update interval in seconds
    
    // Script execution (simplified)
    std::string script_command;         // Command to execute
    bool execute_on_update = false;
    
    // Network properties
    std::string user_agent = "Jin-TextRenderer/1.0";
    int timeout_seconds = 10;
    
    DynamicText();
    explicit DynamicText(const std::string& initial_text);
    ~DynamicText() = default;
    
    // Source management
    void set_source(DataSource source_type, const std::string& path);
    void set_update_interval(float interval);
    
    // File watching
    void watch_file(const std::string& file_path);
    void stop_watching_file();
    
    // Network fetching
    void fetch_from_url(const std::string& url);
    void set_user_agent(const std::string& agent);
    void set_timeout(int seconds);
    
    // Script execution
    void set_script_command(const std::string& command);
    void execute_script();
    
    // Update loop
    void update(float delta_time) override;
    
    // Manual refresh
    void refresh();
    
private:
    float update_timer_ = 0.0f;
    std::string last_content_;
    
    void update_from_source();
    bool load_from_file();
    bool fetch_from_network();
    bool execute_script_command();
    
    // Helper methods
    std::string trim_whitespace(const std::string& str) const;
    bool is_content_changed(const std::string& new_content) const;
};

} // namespace scene
} // namespace jin