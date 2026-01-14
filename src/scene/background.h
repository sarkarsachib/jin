#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include "scene_graph.h"
#include "sprite.h"
#include "layer_stack.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace jin {
namespace scene {

/**
 * ParallaxLayer - Individual background layer with parallax scrolling
 */
struct ParallaxLayer {
    Tensor image;                    // Background image [H, W, 4]
    float depth;                     // Parallax factor (0.0 = fixed, 1.0 = moves with camera)
    Tensor scroll_offset;             // Current scroll offset [x, y]
    bool visible = true;             // Layer visibility
    float opacity = 1.0f;            // Layer opacity
    std::string name;                // Layer name
    Tensor tiling_factor = {1.0f, 1.0f}; // How many times to tile the image
    
    // Animation properties
    bool animated = false;
    Tensor animation_speed = {0.0f, 0.0f};  // Auto-scroll speed [x, y]
    float animation_timer = 0.0f;           // Current animation time
    
    // Render properties
    enum class RenderMode {
        TILE,          // Tile the image to fill background
        STRETCH,       // Stretch image to fill background
        CENTER,        // Center the image
        FIT_WIDTH,     // Fit to width, maintain aspect ratio
        FIT_HEIGHT     // Fit to height, maintain aspect ratio
    };
    
    RenderMode render_mode = RenderMode::TILE;
    
    // Quality settings
    bool use_mipmaps = true;         // Use mipmaps for scaling
    float quality_factor = 1.0f;      // Quality multiplier (0.1 to 1.0)
};

/**
 * Background - Multi-layer parallax background system
 */
class Background : public SceneNode {
public:
    // Layer management
    std::vector<std::shared_ptr<ParallaxLayer>> layers;
    
    // Background properties
    Tensor background_color = {0.0f, 0.0f, 0.0f, 1.0f}; // Default background color
    Tensor viewport_size = {1280.0f, 720.0f};           // Target viewport size
    
    // Camera integration
    bool follow_camera = true;        // Whether background follows camera movement
    bool parallax_enabled = true;     // Enable/disable parallax effect
    
    // Performance settings
    bool pre_render_layers = true;   // Pre-render layers for performance
    bool use_layer_culling = true;   // Cull layers outside viewport
    float cull_margin = 50.0f;       // Margin for culling
    
    // Background dimensions
    int width = 1920;
    int height = 1080;
    bool dynamic_size = true;        // Automatically size to viewport
    
    // Animation properties
    float global_scroll_speed = 1.0f; // Global scroll speed multiplier
    bool loop_horizontally = true;   // Loop background horizontally
    bool loop_vertically = true;     // Loop background vertically
    
    Background();
    ~Background() = default;
    
    // Layer management
    void add_layer(const std::shared_ptr<ParallaxLayer>& layer);
    void remove_layer(const std::string& name);
    void remove_layer(size_t index);
    void clear_layers();
    std::shared_ptr<ParallaxLayer> get_layer(const std::string& name) const;
    std::shared_ptr<ParallaxLayer> get_layer(size_t index) const;
    
    // Background properties
    void set_background_color(const Tensor& color);
    void set_viewport_size(int width, int height);
    void set_dimensions(int width, int height);
    void set_dynamic_size(bool enabled);
    
    // Camera integration
    void set_follow_camera(bool follow);
    void set_parallax_enabled(bool enabled);
    void set_global_scroll_speed(float speed);
    
    // Looping
    void set_horizontal_looping(bool loop);
    void set_vertical_looping(bool loop);
    
    // Animation
    void update(float delta_time) override;
    void animate_layer(const std::string& name, float delta_time);
    void animate_all_layers(float delta_time);
    
    // Rendering
    Tensor render(int width, int height, const Camera2D& camera) const override;
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state) override;
    
    // Background creation helpers
    static std::shared_ptr<Background> create_sky_background(int width, int height);
    static std::shared_ptr<Background> create_forest_background(int width, int height);
    static std::shared_ptr<Background> create_city_background(int width, int height);
    static std::shared_ptr<Background> create_space_background(int width, int height);
    static std::shared_ptr<Background> create_underwater_background(int width, int height);
    
    // Utility methods
    void set_all_layers_visible(bool visible);
    void set_layer_opacity(const std::string& name, float opacity);
    void set_layer_depth(const std::string& name, float depth);
    void set_layer_render_mode(const std::string& name, ParallaxLayer::RenderMode mode);
    
    // Performance optimization
    void optimize_for_performance();
    void pre_render_all_layers();
    
private:
    // Cached rendering
    mutable Tensor cached_background_;
    mutable bool needs_update_ = true;
    
    // Layer rendering helpers
    Tensor render_layer(const ParallaxLayer& layer, int width, int height, const Camera2D& camera) const;
    Tensor apply_parallax(const ParallaxLayer& layer, const Tensor& camera_position) const;
    Tensor tile_image(const Tensor& image, int target_width, int target_height) const;
    Tensor stretch_image(const Tensor& image, int target_width, int target_height) const;
    Tensor fit_image(const Tensor& image, int target_width, int target_height, bool fit_width) const;
    
    // Culling and optimization
    bool should_cull_layer(const ParallaxLayer& layer, int width, int height, const Camera2D& camera) const;
    void update_cached_data() const;
    
    // Animation helpers
    void update_layer_animation(ParallaxLayer& layer, float delta_time);
    Tensor calculate_scroll_offset(const ParallaxLayer& layer, const Camera2D& camera) const;
};

/**
 * AnimatedBackground - Background with built-in animations
 */
class AnimatedBackground : public Background {
public:
    // Animation sequences
    enum class AnimationType {
        DAY_NIGHT_CYCLE,
        SEASONAL_CHANGE,
        WEATHER_CHANGE,
        TIME_LAPSE
    };
    
    AnimationType animation_type = AnimationType::DAY_NIGHT_CYCLE;
    float animation_speed = 1.0f;      // Animation speed multiplier
    bool loop_animation = true;       // Whether to loop the animation
    
    // Time-based properties
    float current_time = 0.0f;        // Current time in animation (0.0 to 1.0)
    float animation_duration = 60.0f;  // Duration of one full cycle in seconds
    
    // Layer transition properties
    std::map<std::string, float> layer_opacity_transitions;  // Layer name -> target opacity
    std::map<std::string, Tensor> layer_color_transitions;   // Layer name -> target color
    
    AnimatedBackground();
    ~AnimatedBackground() = default;
    
    // Animation control
    void set_animation_type(AnimationType type);
    void set_animation_speed(float speed);
    void set_animation_duration(float duration);
    void set_loop_animation(bool loop);
    
    // Time control
    void set_current_time(float time_normalized);
    void advance_time(float delta_time);
    
    // Layer transitions
    void add_layer_opacity_transition(const std::string& layer_name, float target_opacity, float transition_time);
    void add_layer_color_transition(const std::string& layer_name, const Tensor& target_color, float transition_time);
    
    // Built-in animations
    void setup_day_night_cycle();
    void setup_seasonal_change();
    void setup_weather_change();
    void setup_time_lapse();
    
    // Update loop
    void update(float delta_time) override;
    
private:
    void update_day_night_cycle(float delta_time);
    void update_seasonal_change(float delta_time);
    void update_weather_change(float delta_time);
    void update_time_lapse(float delta_time);
    
    // Transition helpers
    float interpolate_transition(const std::string& layer_name, const std::map<std::string, float>& transitions) const;
    Tensor interpolate_color_transition(const std::string& layer_name, const std::map<std::string, Tensor>& transitions) const;
};

/**
 * InteractiveBackground - Background that responds to user interaction
 */
class InteractiveBackground : public Background {
public:
    // Interaction properties
    bool mouse_reactive = false;       // Respond to mouse movement
    bool audio_reactive = false;      // Respond to audio
    bool physics_enabled = false;     // Enable physics interactions
    
    // Mouse interaction
    Tensor mouse_position = {0.0f, 0.0f};
    float mouse_influence_radius = 100.0f;
    float mouse_influence_strength = 0.5f;
    
    // Audio interaction
    float audio_level = 0.0f;
    Tensor frequency_spectrum;
    float audio_influence_strength = 1.0f;
    
    // Physics interaction
    Tensor gravity_center = {0.0f, 0.0f};
    float gravity_strength = 1.0f;
    float friction = 0.9f;
    
    InteractiveBackground();
    ~InteractiveBackground() = default;
    
    // Interaction control
    void set_mouse_reactive(bool reactive);
    void set_audio_reactive(bool reactive);
    void set_physics_enabled(bool enabled);
    
    // Mouse input
    void set_mouse_position(const Tensor& position);
    void set_mouse_influence(float radius, float strength);
    
    // Audio input
    void set_audio_level(float level);
    void set_frequency_spectrum(const Tensor& spectrum);
    void set_audio_influence_strength(float strength);
    
    // Physics input
    void set_gravity_center(const Tensor& center);
    void set_gravity_strength(float strength);
    void set_friction(float friction_value);
    
    // Update loop
    void update(float delta_time) override;
    
private:
    void apply_mouse_interaction(float delta_time);
    void apply_audio_interaction(float delta_time);
    void apply_physics_interaction(float delta_time);
    
    // Helper methods
    Tensor calculate_mouse_influence(const ParallaxLayer& layer) const;
    Tensor calculate_audio_influence(const ParallaxLayer& layer) const;
    Tensor calculate_physics_influence(const ParallaxLayer& layer) const;
};

} // namespace scene
} // namespace jin