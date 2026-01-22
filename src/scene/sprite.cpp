#include "sprite.h"
#include <algorithm>
#include <cmath>

namespace jin {
namespace scene {

// Sprite implementation
Sprite::Sprite() {
    name = "sprite";
}

Sprite::Sprite(const Tensor& image) {
    name = "sprite";
    set_image(image);
}

Sprite::Sprite(const Tensor& image, const std::string& name) : name(name) {
    set_image(image);
}

void Sprite::set_image(const Tensor& image) {
    image_data = image;
    original_height = image.shape()[0];
    original_width = image.shape()[1];
    mark_needs_update();
}

void Sprite::update_image(const Tensor& new_image) {
    set_image(new_image);
}

void Sprite::set_mesh_data(const Tensor& mesh) {
    mesh_data = mesh;
    mark_needs_update();
}

void Sprite::apply_deformation(const Tensor& deformed_mesh) {
    target_deformation = deformed_mesh;
    if (use_mesh_deformation) {
        apply_deformation(deformed_mesh);
    }
}

void Sprite::set_uv_coordinates(const Tensor& uv_coords) {
    uv_coordinates = uv_coords;
    mark_needs_update();
}

void Sprite::set_animation_pose(const Tensor& pose) {
    current_animation_pose = pose;
    mark_needs_update();
}

void Sprite::set_target_deformation(const Tensor& deformation) {
    target_deformation = deformation;
    mark_needs_update();
}

void Sprite::set_opacity(float opacity_value) {
    opacity = std::clamp(opacity_value, 0.0f, 1.0f);
    mark_needs_update();
}

void Sprite::flip_horizontal() {
    // Flip image horizontally
    if (!image_data.size()) return;
    
    // This would require proper image manipulation
    // For now, just mark as needing update
    mark_needs_update();
}

void Sprite::flip_vertical() {
    // Flip image vertically
    if (!image_data.size()) return;
    
    // This would require proper image manipulation
    // For now, just mark as needing update
    mark_needs_update();
}

void Sprite::rotate_90_degrees() {
    // Rotate image 90 degrees
    if (!image_data.size()) return;
    
    // This would require proper image manipulation
    // For now, just mark as needing update
    mark_needs_update();
}

void Sprite::update_from_diffusion_output(const Tensor& diffusion_result) {
    // Update sprite image from diffusion output (Layer 5)
    set_image(diffusion_result);
}

void Sprite::blend_with_diffusion_result(const Tensor& diffusion_result, float blend_factor) {
    // Blend current image with diffusion result
    if (image_data.size() && diffusion_result.size()) {
        Tensor blended = image_data * (1.0f - blend_factor) + diffusion_result * blend_factor;
        set_image(blended);
    }
}

void Sprite::set_audio_reactive_mode(bool enabled) {
    audio_reactive_ = enabled;
}

void Sprite::update_audio_properties(float audio_level, float frequency_spectrum) {
    if (audio_reactive_) {
        last_audio_level_ = audio_level;
        last_frequency_spectrum_ = frequency_spectrum;
        
        // Apply audio-reactive effects
        if (audio_level > 0.5f) {
            // Simple glow effect based on audio level
            set_opacity(std::min(1.0f, opacity + audio_level * 0.2f));
        }
        
        mark_needs_update();
    }
}

void Sprite::set_frame_state(const temporal::FrameState& frame_state) {
    SceneNode::set_frame_state(frame_state);
    
    // Update sprite based on frame state
    if (frame_state.emotion_embedding.size() >= 3) {
        // Extract emotion colors (simplified)
        float* emotion_data = frame_state.emotion_embedding.data_ptr<float>();
        Tensor emotion_color = Tensor({emotion_data[0], emotion_data[1], emotion_data[2], 1.0f});
        
        // Apply emotion to sprite tint (would require proper image processing)
        mark_needs_update();
    }
}

temporal::FrameState Sprite::get_frame_state() const {
    return SceneNode::get_frame_state();
}

Tensor Sprite::render(int width, int height, const Camera2D& camera) const {
    update_cached_data();
    
    if (!image_data.size()) {
        return zeros({height, width, 4});
    }
    
    // Calculate sprite position in world space
    Transform2D world_transform = get_world_transform();
    Tensor world_pos = world_transform.position;
    
    // Convert to screen coordinates
    Tensor screen_pos = camera.world_to_screen(world_pos);
    
    // Calculate sprite size (considering scale)
    Tensor sprite_scale = world_transform.scale;
    float scale_x = sprite_scale.size() >= 2 ? sprite_scale.data_ptr<float>()[0] : 1.0f;
    float scale_y = sprite_scale.size() >= 2 ? sprite_scale.data_ptr<float>()[1] : 1.0f;
    
    int render_width = static_cast<int>(original_width * scale_x);
    int render_height = static_cast<int>(original_height * scale_y);
    
    // Create output frame
    Tensor output_frame = zeros({height, width, 4});
    
    // Calculate sprite bounds on screen
    int sprite_x = static_cast<int>(screen_pos.data_ptr<float>()[0] - render_width / 2.0f);
    int sprite_y = static_cast<int>(screen_pos.data_ptr<float>()[1] - render_height / 2.0f);
    
    // Clamp to screen bounds
    sprite_x = std::max(0, std::min(sprite_x, width - render_width));
    sprite_y = std::max(0, std::min(sprite_y, height - render_height));
    
    // Simple blit (this is a simplified implementation)
    // In a real implementation, this would involve proper texture mapping
    size_t pixels_to_copy = std::min(static_cast<size_t>(render_width * render_height),
                                   image_data.size() / 4);
    
    float* output_data = output_frame.data_ptr<float>();
    const float* sprite_data = image_data.data_ptr<float>();
    
    for (size_t y = 0; y < render_height && (sprite_y + y) < height; ++y) {
        for (size_t x = 0; x < render_width && (sprite_x + x) < width; ++x) {
            size_t sprite_idx = (y * render_width + x) * 4;
            size_t output_idx = ((sprite_y + y) * width + (sprite_x + x)) * 4;
            
            if (sprite_idx + 3 < image_data.size() && output_idx + 3 < output_frame.size()) {
                // Copy RGBA with opacity
                float alpha = sprite_data[sprite_idx + 3] * opacity;
                output_data[output_idx] = sprite_data[sprite_idx];         // R
                output_data[output_idx + 1] = sprite_data[sprite_idx + 1]; // G
                output_data[output_idx + 2] = sprite_data[sprite_idx + 2]; // B
                output_data[output_idx + 3] = alpha;                      // A
            }
        }
    }
    
    // Apply blend mode
    return apply_blend_modes(output_frame, zeros({height, width, 4}));
}

Tensor Sprite::apply_blend_modes(const Tensor& source, const Tensor& destination) const {
    // Simplified blend mode implementation
    // In a real implementation, this would handle all blend modes properly
    
    switch (blend_mode) {
        case BlendMode::NORMAL:
            return source;
            
        case BlendMode::ADD:
            // Additive blending: result = source + destination
            return source + destination;
            
        case BlendMode::MULTIPLY:
            // Multiply blending: result = source * destination
            return source * destination;
            
        case BlendMode::SCREEN:
            // Screen blending: result = 1 - (1-source) * (1-destination)
            return 1.0f - (1.0f - source) * (1.0f - destination);
            
        default:
            return source;
    }
}

Tensor Sprite::apply_opacity(const Tensor& image) const {
    if (opacity >= 1.0f) return image;
    
    Tensor result = image.clone();
    if (result.size() >= 4) {
        size_t pixel_count = result.size() / 4;
        float* data = result.data_ptr<float>();
        
        for (size_t i = 0; i < pixel_count; ++i) {
            data[i * 4 + 3] *= opacity; // Multiply alpha channel
        }
    }
    
    return result;
}

Tensor Sprite::apply_mesh_deformation(const Tensor& image) const {
    // Simplified mesh deformation
    // In a real implementation, this would warp the image based on mesh vertices
    
    if (!use_mesh_deformation || !mesh_data.size()) {
        return image;
    }
    
    // For now, just return the original image
    // A real implementation would warp the image based on the mesh
    return image;
}

void Sprite::mark_needs_update() {
    needs_update_ = true;
}

void Sprite::update_cached_data() const {
    if (needs_update_) {
        // Update cached data based on current state
        // This would involve re-processing the image with current effects
        needs_update_ = false;
    }
}

// AnimatedSprite implementation
AnimatedSprite::AnimatedSprite() : Sprite() {
    name = "animated_sprite";
}

void AnimatedSprite::add_frame(const Tensor& frame_image, float duration, const std::string& name) {
    AnimationFrame frame;
    frame.image = frame_image;
    frame.duration = duration;
    frame.name = name.empty() ? "frame_" + std::to_string(animation_frames.size()) : name;
    
    animation_frames.push_back(frame);
}

void AnimatedSprite::remove_frame(int frame_index) {
    if (frame_index >= 0 && frame_index < static_cast<int>(animation_frames.size())) {
        animation_frames.erase(animation_frames.begin() + frame_index);
    }
}

void AnimatedSprite::clear_frames() {
    animation_frames.clear();
    current_frame = 0;
    frame_timer = 0.0f;
}

void AnimatedSprite::play() {
    is_playing = true;
}

void AnimatedSprite::pause() {
    is_playing = false;
}

void AnimatedSprite::stop() {
    is_playing = false;
    current_frame = 0;
    frame_timer = 0.0f;
}

void AnimatedSprite::goto_frame(int frame_index) {
    if (frame_index >= 0 && frame_index < static_cast<int>(animation_frames.size())) {
        current_frame = frame_index;
        frame_timer = 0.0f;
        
        if (animation_frames[current_frame].image.size()) {
            set_image(animation_frames[current_frame].image);
        }
    }
}

void AnimatedSprite::update(float delta_time) {
    Sprite::update(delta_time);
    
    if (is_playing && !animation_frames.empty()) {
        frame_timer += delta_time;
        
        if (frame_timer >= animation_frames[current_frame].duration) {
            advance_frame();
        }
    }
}

void AnimatedSprite::advance_frame() {
    frame_timer = 0.0f;
    current_frame++;
    
    if (current_frame >= static_cast<int>(animation_frames.size())) {
        if (loop_animation) {
            current_frame = 0;
        } else {
            current_frame = static_cast<int>(animation_frames.size()) - 1;
            is_playing = false;
        }
    }
    
    if (current_frame < static_cast<int>(animation_frames.size())) {
        if (animation_frames[current_frame].image.size()) {
            set_image(animation_frames[current_frame].image);
        }
    }
}

// SpriteSheet implementation
SpriteSheet::SpriteSheet() : Sprite() {
    name = "sprite_sheet";
}

void SpriteSheet::configure_sheet(int columns, int rows, int region_width, int region_height) {
    sheet_columns = std::max(1, columns);
    sheet_rows = std::max(1, rows);
    region_width = std::max(1, region_width);
    region_height = std::max(1, region_height);
}

void SpriteSheet::add_region(int x, int y, int width, int height, const std::string& name) {
    SpriteRegion region;
    region.x = x;
    region.y = y;
    region.width = width;
    region.height = height;
    region.name = name.empty() ? "region_" + std::to_string(sprite_regions.size()) : name;
    
    sprite_regions.push_back(region);
}

Tensor SpriteSheet::get_sprite_region(int region_index) const {
    if (region_index >= 0 && region_index < static_cast<int>(sprite_regions.size())) {
        return extract_region(sprite_regions[region_index]);
    }
    return Tensor();
}

Tensor SpriteSheet::get_sprite_region(const std::string& name) const {
    for (const auto& region : sprite_regions) {
        if (region.name == name) {
            return extract_region(region);
        }
    }
    return Tensor();
}

SpriteSheet::SpriteRegion* SpriteSheet::get_region(int region_index) {
    if (region_index >= 0 && region_index < static_cast<int>(sprite_regions.size())) {
        return &sprite_regions[region_index];
    }
    return nullptr;
}

SpriteSheet::SpriteRegion* SpriteSheet::get_region(const std::string& name) {
    for (auto& region : sprite_regions) {
        if (region.name == name) {
            return &region;
        }
    }
    return nullptr;
}

Tensor SpriteSheet::render_region(int region_index, int output_width, int output_height, const Camera2D& camera) const {
    Tensor region_image = get_sprite_region(region_index);
    if (!region_image.size()) {
        return zeros({output_height, output_width, 4});
    }
    
    // Resize region to output size
    // This is a simplified implementation
    // In a real implementation, this would use proper image scaling
    
    Tensor scaled_region = region_image;
    return scaled_region;
}

Tensor SpriteSheet::render_region(const std::string& name, int output_width, int output_height, const Camera2D& camera) const {
    Tensor region_image = get_sprite_region(name);
    if (!region_image.size()) {
        return zeros({output_height, output_width, 4});
    }
    
    // Resize region to output size
    Tensor scaled_region = region_image;
    return scaled_region;
}

Tensor SpriteSheet::extract_region(const SpriteRegion& region) const {
    if (!image_data.size() || region.width <= 0 || region.height <= 0) {
        return Tensor();
    }
    
    // Extract region from sprite sheet
    // This is a simplified implementation
    // In a real implementation, this would properly crop the image
    
    Tensor extracted = zeros({region.height, region.width, 4});
    
    size_t max_height = std::min(static_cast<size_t>(region.height), image_data.shape()[0] - region.y);
    size_t max_width = std::min(static_cast<size_t>(region.width), image_data.shape()[1] - region.x);
    
    const float* source_data = image_data.data_ptr<float>();
    float* extracted_data = extracted.data_ptr<float>();
    
    for (size_t y = 0; y < max_height; ++y) {
        for (size_t x = 0; x < max_width; ++x) {
            size_t source_idx = ((region.y + y) * image_data.shape()[1] + (region.x + x)) * 4;
            size_t extracted_idx = (y * max_width + x) * 4;
            
            for (int c = 0; c < 4 && source_idx + c < image_data.size() && extracted_idx + c < extracted.size(); ++c) {
                extracted_data[extracted_idx + c] = source_data[source_idx + c];
            }
        }
    }
    
    return extracted;
}

// CharacterSprite implementation
CharacterSprite::CharacterSprite() : AnimatedSprite() {
    name = "character_sprite";
}

void CharacterSprite::set_emotion_state(const std::string& emotion, float intensity) {
    emotion_intensities_[emotion] = std::clamp(intensity, 0.0f, 1.0f);
    apply_emotion_effects();
}

void CharacterSprite::update_expression(const Tensor& expression_vector) {
    current_expression_ = expression_vector;
    apply_emotion_effects();
}

void CharacterSprite::set_character_consistency_data(const Tensor& identity_vector) {
    character_identity = identity_vector;
    ensure_character_consistency();
}

void CharacterSprite::set_motion_data(const Tensor& motion_vector) {
    // Update character sprite based on motion data
    // This would integrate with Layer 7 animation system
    mark_needs_update();
}

void CharacterSprite::generate_from_prompt(const std::string& character_description) {
    // This would integrate with Layer 5 diffusion system
    // For now, just mark as needing update
    mark_needs_update();
}

Tensor CharacterSprite::render(int width, int height, const Camera2D& camera) const {
    // Apply character-specific effects before rendering
    apply_emotion_effects();
    ensure_character_consistency();
    
    return AnimatedSprite::render(width, height, camera);
}

void CharacterSprite::apply_emotion_effects() {
    // Apply emotion-based visual effects
    // This would modify the sprite appearance based on emotions
    mark_needs_update();
}

void CharacterSprite::ensure_character_consistency() {
    // Ensure character appearance remains consistent
    // This would integrate with Layer 3 neural primitives
    mark_needs_update();
}

} // namespace scene
} // namespace jin