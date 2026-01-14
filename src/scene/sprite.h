#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include "scene_graph.h"
#include <string>
#include <memory>
#include <functional>

namespace jin {
namespace scene {

/**
 * BlendMode - Different blending modes for sprites
 */
enum class BlendMode {
    NORMAL,     // Standard alpha blending
    ADD,        // Additive blending
    MULTIPLY,   // Multiply blending
    SCREEN,     // Screen blending
    OVERLAY,    // Overlay blending
    SOFT_LIGHT, // Soft light blending
    HARD_LIGHT  // Hard light blending
};

/**
 * Sprite - 2D image-based scene object
 */
class Sprite : public SceneNode {
public:
    // Sprite data
    Tensor image_data;           // RGBA image data [H, W, 4]
    Tensor mesh_data;            // Optional deformed mesh vertices
    Tensor uv_coordinates;       // UV mapping coordinates
    
    // Rendering properties
    BlendMode blend_mode = BlendMode::NORMAL;
    float opacity = 1.0f;
    bool use_alpha = true;
    bool use_mesh_deformation = false;
    
    // Animation data (from Layer 7 integration)
    Tensor current_animation_pose;  // Current character pose
    Tensor target_deformation;       // Target mesh deformation
    
    // Sprite dimensions
    int original_width = 0;
    int original_height = 0;
    
    // Constructors
    Sprite();
    explicit Sprite(const Tensor& image);
    Sprite(const Tensor& image, const std::string& name);
    ~Sprite() = default;
    
    // Image management
    void set_image(const Tensor& image);
    Tensor get_image() const { return image_data; }
    void update_image(const Tensor& new_image);
    
    // Mesh deformation (from Layer 7)
    void set_mesh_data(const Tensor& mesh);
    void apply_deformation(const Tensor& deformed_mesh);
    void set_deformation_mode(bool enabled) { use_mesh_deformation = enabled; }
    
    // UV coordinate management
    void set_uv_coordinates(const Tensor& uv_coords);
    Tensor get_uv_coordinates() const { return uv_coordinates; }
    
    // Animation integration (Layer 7)
    void set_animation_pose(const Tensor& pose);
    Tensor get_animation_pose() const { return current_animation_pose; }
    
    void set_target_deformation(const Tensor& deformation);
    Tensor get_target_deformation() const { return target_deformation; }
    
    // Rendering properties
    void set_blend_mode(BlendMode mode) { blend_mode = mode; }
    BlendMode get_blend_mode() const { return blend_mode; }
    
    void set_opacity(float opacity_value);
    float get_opacity() const { return opacity; }
    
    void set_use_alpha(bool use) { use_alpha = use; }
    bool get_use_alpha() const { return use_alpha; }
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state) override;
    temporal::FrameState get_frame_state() const override;
    
    // Rendering
    Tensor render(int width, int height, const Camera2D& camera) const override;
    
    // Utility methods
    void flip_horizontal();
    void flip_vertical();
    void rotate_90_degrees();
    
    // Diffusion integration (Layer 5)
    void update_from_diffusion_output(const Tensor& diffusion_result);
    void blend_with_diffusion_result(const Tensor& diffusion_result, float blend_factor);
    
    // Audio-reactive properties (Layer 6)
    void set_audio_reactive_mode(bool enabled);
    void update_audio_properties(float audio_level, float frequency_spectrum);
    
private:
    // Internal rendering helpers
    Tensor apply_blend_modes(const Tensor& source, const Tensor& destination) const;
    Tensor apply_opacity(const Tensor& image) const;
    Tensor apply_mesh_deformation(const Tensor& image) const;
    
    // Cached data
    mutable Tensor cached_rendered_image_;
    mutable bool needs_update_ = true;
    
    // Audio-reactive properties
    bool audio_reactive_ = false;
    float last_audio_level_ = 0.0f;
    float last_frequency_spectrum_ = 0.0f;
    
    void mark_needs_update();
    void update_cached_data() const;
};

/**
 * AnimatedSprite - Sprite with built-in animation support
 */
class AnimatedSprite : public Sprite {
public:
    // Animation frame management
    struct AnimationFrame {
        Tensor image;
        float duration;  // Duration in seconds
        std::string name;
    };
    
    std::vector<AnimationFrame> animation_frames;
    int current_frame = 0;
    float frame_timer = 0.0f;
    bool is_playing = false;
    bool loop_animation = true;
    
    AnimatedSprite();
    ~AnimatedSprite() = default;
    
    // Frame management
    void add_frame(const Tensor& frame_image, float duration, const std::string& name = "");
    void remove_frame(int frame_index);
    void clear_frames();
    
    // Animation control
    void play();
    void pause();
    void stop();
    void goto_frame(int frame_index);
    
    // Animation properties
    void set_loop(bool loop) { loop_animation = loop; }
    bool get_loop() const { return loop_animation; }
    
    int get_frame_count() const { return static_cast<int>(animation_frames.size()); }
    int get_current_frame() const { return current_frame; }
    
    // Update animation
    void update(float delta_time) override;
    
private:
    void advance_frame();
};

/**
 * SpriteSheet - Manages multiple sprites in a single texture
 */
class SpriteSheet : public Sprite {
public:
    // Sprite sheet configuration
    struct SpriteRegion {
        int x, y;           // Top-left corner in sheet
        int width, height;   // Region dimensions
        std::string name;    // Region name
    };
    
    std::vector<SpriteRegion> sprite_regions;
    int sheet_columns = 1;
    int sheet_rows = 1;
    int region_width = 0;
    int region_height = 0;
    
    SpriteSheet();
    ~SpriteSheet() = default;
    
    // Sheet configuration
    void configure_sheet(int columns, int rows, int region_width, int region_height);
    void add_region(int x, int y, int width, int height, const std::string& name);
    
    // Sprite extraction
    Tensor get_sprite_region(int region_index) const;
    Tensor get_sprite_region(const std::string& name) const;
    SpriteRegion* get_region(int region_index);
    SpriteRegion* get_region(const std::string& name);
    
    // Render specific region
    Tensor render_region(int region_index, int output_width, int output_height, const Camera2D& camera) const;
    Tensor render_region(const std::string& name, int output_width, int output_height, const Camera2D& camera) const;
    
private:
    Tensor extract_region(const SpriteRegion& region) const;
};

/**
 * CharacterSprite - Specialized sprite for character rendering
 */
class CharacterSprite : public AnimatedSprite {
public:
    // Character-specific properties
    enum CharacterState {
        IDLE,
        WALKING,
        RUNNING,
        TALKING,
        EMOTIONAL
    };
    
    CharacterState current_state = IDLE;
    std::string character_name;
    Tensor character_identity;  // Character identity vector for consistency
    
    // Emotion-based rendering
    void set_emotion_state(const std::string& emotion, float intensity);
    void update_expression(const Tensor& expression_vector);
    
    // Character consistency (Layer 3 integration)
    void set_character_consistency_data(const Tensor& identity_vector);
    Tensor get_character_consistency_data() const { return character_identity; }
    
    // Motion integration (Layer 7)
    void set_motion_data(const Tensor& motion_vector);
    void update_from_motion(const Tensor& motion_data);
    
    // Layer 5 integration for character generation
    void generate_from_prompt(const std::string& character_description);
    
    // Rendering with character-specific effects
    Tensor render(int width, int height, const Camera2D& camera) const override;
    
private:
    std::map<std::string, float> emotion_intensities_;
    Tensor current_expression_;
    
    void apply_emotion_effects();
    void ensure_character_consistency();
};

} // namespace scene
} // namespace jin