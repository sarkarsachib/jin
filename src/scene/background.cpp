#include "background.h"
#include <algorithm>
#include <cmath>

namespace jin {
namespace scene {

// ParallaxLayer implementation (inline in header)

// Background implementation
Background::Background() {
    name = "background";
}

void Background::add_layer(const std::shared_ptr<ParallaxLayer>& layer) {
    if (layer) {
        layers.push_back(layer);
        if (layer->name.empty()) {
            layer->name = "layer_" + std::to_string(layers.size());
        }
        needs_update_ = true;
    }
}

void Background::remove_layer(const std::string& name) {
    auto it = std::find_if(layers.begin(), layers.end(),
        [&name](const std::shared_ptr<ParallaxLayer>& layer) {
            return layer->name == name;
        });
    
    if (it != layers.end()) {
        layers.erase(it);
        needs_update_ = true;
    }
}

void Background::remove_layer(size_t index) {
    if (index < layers.size()) {
        layers.erase(layers.begin() + index);
        needs_update_ = true;
    }
}

void Background::clear_layers() {
    layers.clear();
    needs_update_ = true;
}

std::shared_ptr<ParallaxLayer> Background::get_layer(const std::string& name) const {
    for (const auto& layer : layers) {
        if (layer->name == name) {
            return layer;
        }
    }
    return nullptr;
}

std::shared_ptr<ParallaxLayer> Background::get_layer(size_t index) const {
    if (index < layers.size()) {
        return layers[index];
    }
    return nullptr;
}

void Background::set_background_color(const Tensor& color) {
    background_color = color;
    needs_update_ = true;
}

void Background::set_viewport_size(int width, int height) {
    viewport_size = Tensor({static_cast<float>(width), static_cast<float>(height)});
    needs_update_ = true;
}

void Background::set_dimensions(int width, int height) {
    this->width = width;
    this->height = height;
    needs_update_ = true;
}

void Background::set_dynamic_size(bool enabled) {
    dynamic_size = enabled;
    needs_update_ = true;
}

void Background::set_follow_camera(bool follow) {
    follow_camera = follow;
}

void Background::set_parallax_enabled(bool enabled) {
    parallax_enabled = enabled;
}

void Background::set_global_scroll_speed(float speed) {
    global_scroll_speed = std::max(0.0f, speed);
}

void Background::set_horizontal_looping(bool loop) {
    loop_horizontally = loop;
}

void Background::set_vertical_looping(bool loop) {
    loop_vertically = loop;
}

void Background::update(float delta_time) {
    SceneNode::update(delta_time);
    
    // Update layer animations
    animate_all_layers(delta_time);
    
    needs_update_ = true;
}

void Background::animate_layer(const std::string& name, float delta_time) {
    auto layer = get_layer(name);
    if (layer) {
        update_layer_animation(*layer, delta_time);
        needs_update_ = true;
    }
}

void Background::animate_all_layers(float delta_time) {
    for (auto& layer : layers) {
        update_layer_animation(*layer, delta_time);
    }
}

Tensor Background::render(int width, int height, const Camera2D& camera) const {
    update_cached_data();
    
    if (dynamic_size) {
        viewport_size = Tensor({static_cast<float>(width), static_cast<float>(height)});
    }
    
    // Start with background color
    Tensor result = zeros({height, width, 4});
    
    // Fill with background color
    if (background_color.size() >= 4) {
        float* bg_color = background_color.data_ptr<float>();
        float* result_data = result.data_ptr<float>();
        size_t pixel_count = result.size() / 4;
        
        for (size_t i = 0; i < pixel_count; ++i) {
            result_data[i * 4] = bg_color[0];     // R
            result_data[i * 4 + 1] = bg_color[1]; // G
            result_data[i * 4 + 2] = bg_color[2]; // B
            result_data[i * 4 + 3] = bg_color[3]; // A
        }
    }
    
    // Render each layer in order (back to front)
    for (const auto& layer : layers) {
        if (!layer->visible) continue;
        
        if (use_layer_culling && should_cull_layer(*layer, width, height, camera)) {
            continue;
        }
        
        Tensor layer_render = render_layer(*layer, width, height, camera);
        
        // Composite layer with opacity
        if (layer->opacity < 1.0f) {
            // Apply opacity (simplified)
            // In a real implementation, this would properly blend with alpha
        }
        
        result = result + layer_render; // Simple additive composite
    }
    
    return result;
}

void Background::set_frame_state(const temporal::FrameState& frame_state) {
    SceneNode::set_frame_state(frame_state);
    
    // Update layers based on frame state
    // This could be used for scene transitions, weather effects, etc.
    needs_update_ = true;
}

std::shared_ptr<Background> Background::create_sky_background(int width, int height) {
    auto background = std::make_shared<Background>();
    background->set_dimensions(width, height);
    
    // Create sky gradient layer
    auto sky_layer = std::make_shared<ParallaxLayer>();
    sky_layer->depth = 1.0f;
    sky_layer->render_mode = ParallaxLayer::RenderMode::STRETCH;
    
    // Create gradient sky (simplified)
    Tensor sky_image = zeros({height, width, 4});
    for (int y = 0; y < height; ++y) {
        float gradient_factor = static_cast<float>(y) / height;
        for (int x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 4;
            sky_image.data_ptr<float>()[idx] = 0.5f + gradient_factor * 0.3f;     // R
            sky_image.data_ptr<float>()[idx + 1] = 0.7f + gradient_factor * 0.2f; // G
            sky_image.data_ptr<float>()[idx + 2] = 0.9f;                          // B
            sky_image.data_ptr<float>()[idx + 3] = 1.0f;                          // A
        }
    }
    sky_layer->image = sky_image;
    
    // Add clouds layer
    auto cloud_layer = std::make_shared<ParallaxLayer>();
    cloud_layer->depth = 0.8f;
    cloud_layer->render_mode = ParallaxLayer::RenderMode::TILE;
    cloud_layer->animated = true;
    cloud_layer->animation_speed = Tensor({10.0f, 0.0f}); // Slow horizontal movement
    
    background->add_layer(sky_layer);
    background->add_layer(cloud_layer);
    
    return background;
}

std::shared_ptr<Background> Background::create_forest_background(int width, int height) {
    auto background = std::make_shared<Background>();
    background->set_dimensions(width, height);
    
    // Create forest layers (simplified)
    auto sky_layer = std::make_shared<ParallaxLayer>();
    sky_layer->depth = 1.0f;
    sky_layer->render_mode = ParallaxLayer::RenderMode::STRETCH;
    sky_layer->image = ones({height, width, 4}) * Tensor({0.6f, 0.8f, 0.9f, 1.0f}); // Light blue sky
    
    auto far_trees = std::make_shared<ParallaxLayer>();
    far_trees->depth = 0.7f;
    far_trees->render_mode = ParallaxLayer::RenderMode::TILE;
    far_trees->image = ones({height, width, 4}) * Tensor({0.2f, 0.5f, 0.3f, 1.0f}); // Dark green
    
    auto near_trees = std::make_shared<ParallaxLayer>();
    near_trees->depth = 0.5f;
    near_trees->render_mode = ParallaxLayer::RenderMode::TILE;
    near_trees->image = ones({height, width, 4}) * Tensor({0.1f, 0.4f, 0.2f, 1.0f}); // Darker green
    
    background->add_layer(sky_layer);
    background->add_layer(far_trees);
    background->add_layer(near_trees);
    
    return background;
}

std::shared_ptr<Background> Background::create_city_background(int width, int height) {
    auto background = std::make_shared<Background>();
    background->set_dimensions(width, height);
    
    // Create city layers (simplified)
    auto sky_layer = std::make_shared<ParallaxLayer>();
    sky_layer->depth = 1.0f;
    sky_layer->render_mode = ParallaxLayer::RenderMode::STRETCH;
    sky_layer->image = ones({height, width, 4}) * Tensor({0.7f, 0.8f, 0.9f, 1.0f}); // Light gray sky
    
    auto buildings_layer = std::make_shared<ParallaxLayer>();
    buildings_layer->depth = 0.6f;
    buildings_layer->render_mode = ParallaxLayer::RenderMode::TILE;
    buildings_layer->image = ones({height, width, 4}) * Tensor({0.3f, 0.3f, 0.4f, 1.0f}); // Dark gray buildings
    
    background->add_layer(sky_layer);
    background->add_layer(buildings_layer);
    
    return background;
}

std::shared_ptr<Background> Background::create_space_background(int width, int height) {
    auto background = std::make_shared<Background>();
    background->set_dimensions(width, height);
    background->set_background_color({0.0f, 0.0f, 0.1f, 1.0f}); // Deep space blue
    
    // Create starfield (simplified)
    auto star_layer = std::make_shared<ParallaxLayer>();
    star_layer->depth = 1.0f;
    star_layer->render_mode = ParallaxLayer::RenderMode::TILE;
    
    // Generate starfield
    Tensor star_image = zeros({height, width, 4});
    for (int i = 0; i < 100; ++i) { // Random stars
        int x = rand() % width;
        int y = rand() % height;
        size_t idx = (y * width + x) * 4;
        star_image.data_ptr<float>()[idx] = 1.0f;     // R
        star_image.data_ptr<float>()[idx + 1] = 1.0f; // G
        star_image.data_ptr<float>()[idx + 2] = 1.0f; // B
        star_image.data_ptr<float>()[idx + 3] = 1.0f; // A
    }
    
    star_layer->image = star_image;
    background->add_layer(star_layer);
    
    return background;
}

std::shared_ptr<Background> Background::create_underwater_background(int width, int height) {
    auto background = std::make_shared<Background>();
    background->set_dimensions(width, height);
    background->set_background_color({0.0f, 0.1f, 0.2f, 1.0f}); // Deep blue water
    
    // Create water layers
    auto water_layer = std::make_shared<ParallaxLayer>();
    water_layer->depth = 1.0f;
    water_layer->render_mode = ParallaxLayer::RenderMode::STRETCH;
    water_layer->animated = true;
    water_layer->animation_speed = Tensor({0.0f, 5.0f}); // Gentle vertical movement
    
    // Water gradient (simplified)
    Tensor water_image = zeros({height, width, 4});
    for (int y = 0; y < height; ++y) {
        float depth_factor = static_cast<float>(y) / height;
        for (int x = 0; x < width; ++x) {
            size_t idx = (y * width + x) * 4;
            water_image.data_ptr<float>()[idx] = 0.1f + depth_factor * 0.1f;     // R
            water_image.data_ptr<float>()[idx + 1] = 0.3f + depth_factor * 0.3f; // G
            water_image.data_ptr<float>()[idx + 2] = 0.6f + depth_factor * 0.3f; // B
            water_image.data_ptr<float>()[idx + 3] = 1.0f;                       // A
        }
    }
    water_layer->image = water_image;
    
    background->add_layer(water_layer);
    
    return background;
}

void Background::set_all_layers_visible(bool visible) {
    for (auto& layer : layers) {
        layer->visible = visible;
    }
    needs_update_ = true;
}

void Background::set_layer_opacity(const std::string& name, float opacity) {
    auto layer = get_layer(name);
    if (layer) {
        layer->opacity = std::clamp(opacity, 0.0f, 1.0f);
        needs_update_ = true;
    }
}

void Background::set_layer_depth(const std::string& name, float depth) {
    auto layer = get_layer(name);
    if (layer) {
        layer->depth = std::clamp(depth, 0.0f, 1.0f);
        needs_update_ = true;
    }
}

void Background::set_layer_render_mode(const std::string& name, ParallaxLayer::RenderMode mode) {
    auto layer = get_layer(name);
    if (layer) {
        layer->render_mode = mode;
        needs_update_ = true;
    }
}

void Background::optimize_for_performance() {
    // Set performance-oriented settings
    pre_render_layers = true;
    use_layer_culling = true;
    
    for (auto& layer : layers) {
        layer->use_mipmaps = false;
        layer->quality_factor = 0.8f;
    }
}

void Background::pre_render_all_layers() {
    if (!pre_render_layers) return;
    
    // Pre-render layers for current viewport
    for (auto& layer : layers) {
        if (layer->visible) {
            // This would involve rendering the layer to a texture
            // For now, just mark as needing update
        }
    }
}

Tensor Background::render_layer(const ParallaxLayer& layer, int width, int height, const Camera2D& camera) const {
    if (!layer.visible || layer.image.size() == 0) {
        return zeros({height, width, 4});
    }
    
    Tensor layer_result;
    
    // Apply parallax effect
    Tensor camera_pos = camera.position;
    if (parallax_enabled && follow_camera) {
        layer_result = apply_parallax(layer, camera_pos);
    } else {
        layer_result = layer.image;
    }
    
    // Render based on mode
    switch (layer.render_mode) {
        case ParallaxLayer::RenderMode::TILE:
            return tile_image(layer_result, width, height);
        case ParallaxLayer::RenderMode::STRETCH:
            return stretch_image(layer_result, width, height);
        case ParallaxLayer::RenderMode::CENTER:
            // Simplified center rendering
            return stretch_image(layer_result, width, height);
        case ParallaxLayer::RenderMode::FIT_WIDTH:
            return fit_image(layer_result, width, height, true);
        case ParallaxLayer::RenderMode::FIT_HEIGHT:
            return fit_image(layer_result, width, height, false);
        default:
            return stretch_image(layer_result, width, height);
    }
}

Tensor Background::apply_parallax(const ParallaxLayer& layer, const Tensor& camera_position) const {
    if (!parallax_enabled || !follow_camera) {
        return layer.image;
    }
    
    // Calculate parallax offset
    Tensor parallax_offset = camera_position * (1.0f - layer.depth);
    Tensor offset_image = layer.image;
    
    // Apply offset (simplified)
    // In a real implementation, this would properly scroll the image
    
    return offset_image;
}

Tensor Background::tile_image(const Tensor& image, int target_width, int target_height) const {
    if (image.size() == 0) {
        return zeros({target_height, target_width, 4});
    }
    
    Tensor result = zeros({target_height, target_width, 4});
    
    int img_height = image.shape()[0];
    int img_width = image.shape()[1];
    
    // Simple tiling
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            int img_y = y % img_height;
            int img_x = x % img_width;
            
            size_t src_idx = (img_y * img_width + img_x) * 4;
            size_t dst_idx = (y * target_width + x) * 4;
            
            if (src_idx + 3 < image.size() && dst_idx + 3 < result.size()) {
                const float* src_data = image.data_ptr<float>();
                float* dst_data = result.data_ptr<float>();
                
                for (int c = 0; c < 4; ++c) {
                    dst_data[dst_idx + c] = src_data[src_idx + c];
                }
            }
        }
    }
    
    return result;
}

Tensor Background::stretch_image(const Tensor& image, int target_width, int target_height) const {
    if (image.size() == 0) {
        return zeros({target_height, target_width, 4});
    }
    
    // Simple nearest-neighbor scaling
    Tensor result = zeros({target_height, target_width, 4});
    
    int src_height = image.shape()[0];
    int src_width = image.shape()[1];
    
    for (int y = 0; y < target_height; ++y) {
        for (int x = 0; x < target_width; ++x) {
            int src_y = (y * src_height) / target_height;
            int src_x = (x * src_width) / target_width;
            
            src_y = std::min(src_y, src_height - 1);
            src_x = std::min(src_x, src_width - 1);
            
            size_t src_idx = (src_y * src_width + src_x) * 4;
            size_t dst_idx = (y * target_width + x) * 4;
            
            if (src_idx + 3 < image.size() && dst_idx + 3 < result.size()) {
                const float* src_data = image.data_ptr<float>();
                float* dst_data = result.data_ptr<float>();
                
                for (int c = 0; c < 4; ++c) {
                    dst_data[dst_idx + c] = src_data[src_idx + c];
                }
            }
        }
    }
    
    return result;
}

Tensor Background::fit_image(const Tensor& image, int target_width, int target_height, bool fit_width) const {
    if (image.size() == 0) {
        return zeros({target_height, target_width, 4});
    }
    
    // Aspect-ratio preserving fit
    int src_height = image.shape()[0];
    int src_width = image.shape()[1];
    
    float src_aspect = static_cast<float>(src_width) / src_height;
    float target_aspect = static_cast<float>(target_width) / target_height;
    
    int render_width, render_height;
    int offset_x = 0, offset_y = 0;
    
    if (fit_width) {
        render_width = target_width;
        render_height = static_cast<int>(target_width / src_aspect);
        offset_y = (target_height - render_height) / 2;
    } else {
        render_height = target_height;
        render_width = static_cast<int>(target_height * src_aspect);
        offset_x = (target_width - render_width) / 2;
    }
    
    // First scale the image
    Tensor scaled = stretch_image(image, render_width, render_height);
    
    // Then create result with proper offset
    Tensor result = zeros({target_height, target_width, 4});
    
    for (int y = 0; y < render_height && y + offset_y < target_height; ++y) {
        for (int x = 0; x < render_width && x + offset_x < target_width; ++x) {
            size_t src_idx = (y * render_width + x) * 4;
            size_t dst_idx = ((y + offset_y) * target_width + (x + offset_x)) * 4;
            
            if (src_idx + 3 < scaled.size() && dst_idx + 3 < result.size()) {
                const float* src_data = scaled.data_ptr<float>();
                float* dst_data = result.data_ptr<float>();
                
                for (int c = 0; c < 4; ++c) {
                    dst_data[dst_idx + c] = src_data[src_idx + c];
                }
            }
        }
    }
    
    return result;
}

bool Background::should_cull_layer(const ParallaxLayer& layer, int width, int height, const Camera2D& camera) const {
    if (!use_layer_culling) return false;
    
    // Simple culling based on layer depth and camera position
    // In a real implementation, this would check actual screen bounds
    
    float distance = layer.depth;
    return distance < 0.1f || distance > 2.0f;
}

void Background::update_cached_data() const {
    if (!needs_update_) return;
    
    // Update cached background data
    // This would involve pre-rendering or other optimizations
    needs_update_ = false;
}

void Background::update_layer_animation(ParallaxLayer& layer, float delta_time) {
    if (!layer.animated) return;
    
    layer.animation_timer += delta_time;
    
    // Apply animation speed
    if (layer.animation_speed.size() >= 2) {
        layer.scroll_offset = layer.scroll_offset + layer.animation_speed * delta_time * global_scroll_speed;
    }
}

// AnimatedBackground implementation
AnimatedBackground::AnimatedBackground() : Background() {
    name = "animated_background";
}

void AnimatedBackground::set_animation_type(AnimationType type) {
    animation_type = type;
}

void AnimatedBackground::set_animation_speed(float speed) {
    animation_speed = std::max(0.0f, speed);
}

void AnimatedBackground::set_animation_duration(float duration) {
    animation_duration = std::max(1.0f, duration);
}

void AnimatedBackground::set_loop_animation(bool loop) {
    loop_animation = loop;
}

void AnimatedBackground::set_current_time(float time_normalized) {
    current_time = std::clamp(time_normalized, 0.0f, 1.0f);
}

void AnimatedBackground::advance_time(float delta_time) {
    float time_advance = (delta_time / animation_duration) * animation_speed;
    current_time += time_advance;
    
    if (loop_animation) {
        while (current_time > 1.0f) {
            current_time -= 1.0f;
        }
    } else {
        current_time = std::min(current_time, 1.0f);
    }
}

void AnimatedBackground::add_layer_opacity_transition(const std::string& layer_name, float target_opacity, float transition_time) {
    // Store transition parameters (simplified)
    // In a real implementation, this would manage complex transitions
}

void AnimatedBackground::add_layer_color_transition(const std::string& layer_name, const Tensor& target_color, float transition_time) {
    // Store transition parameters (simplified)
    // In a real implementation, this would manage complex transitions
}

void AnimatedBackground::setup_day_night_cycle() {
    set_animation_type(AnimationType::DAY_NIGHT_CYCLE);
    
    // Configure sky layers for day/night transition
    if (layers.size() >= 1) {
        // Add color transitions for day/night cycle
        Tensor day_color = Tensor({0.5f, 0.7f, 0.9f, 1.0f}); // Light blue
        Tensor night_color = Tensor({0.1f, 0.1f, 0.3f, 1.0f}); // Dark blue
        
        // This would be implemented with proper transitions
    }
}

void AnimatedBackground::setup_seasonal_change() {
    set_animation_type(AnimationType::SEASONAL_CHANGE);
    animation_duration = 120.0f; // 2 minutes for full cycle
}

void AnimatedBackground::setup_weather_change() {
    set_animation_type(AnimationType::WEATHER_CHANGE);
    animation_duration = 60.0f; // 1 minute for weather cycle
}

void AnimatedBackground::setup_time_lapse() {
    set_animation_type(AnimationType::TIME_LAPSE);
    animation_duration = 30.0f; // 30 seconds for time lapse
}

void AnimatedBackground::update(float delta_time) {
    Background::update(delta_time);
    
    advance_time(delta_time);
    
    switch (animation_type) {
        case AnimationType::DAY_NIGHT_CYCLE:
            update_day_night_cycle(delta_time);
            break;
        case AnimationType::SEASONAL_CHANGE:
            update_seasonal_change(delta_time);
            break;
        case AnimationType::WEATHER_CHANGE:
            update_weather_change(delta_time);
            break;
        case AnimationType::TIME_LAPSE:
            update_time_lapse(delta_time);
            break;
    }
}

void AnimatedBackground::update_day_night_cycle(float delta_time) {
    // Update sky color based on time of day
    if (layers.size() >= 1) {
        auto& sky_layer = layers[0];
        
        // Interpolate between day and night colors
        float day_factor = std::sin(current_time * 2.0f * 3.14159f); // Day-night cycle
        day_factor = (day_factor + 1.0f) / 2.0f; // Normalize to 0-1
        
        Tensor day_color = Tensor({0.5f, 0.7f, 0.9f, 1.0f});
        Tensor night_color = Tensor({0.1f, 0.1f, 0.3f, 1.0f});
        
        Tensor current_color = day_color * day_factor + night_color * (1.0f - day_factor);
        
        // Apply to layer (simplified)
        // sky_layer->color = current_color;
    }
}

void AnimatedBackground::update_seasonal_change(float delta_time) {
    // Update colors for seasonal changes
    // Spring: green, Summer: bright green, Autumn: orange/brown, Winter: white/gray
}

void AnimatedBackground::update_weather_change(float delta_time) {
    // Update for weather changes (rain, snow, clouds, etc.)
}

void AnimatedBackground::update_time_lapse(float delta_time) {
    // Speed up all animations for time lapse effect
    global_scroll_speed = 5.0f * animation_speed;
}

float AnimatedBackground::interpolate_transition(const std::string& layer_name, const std::map<std::string, float>& transitions) const {
    // Simplified transition interpolation
    auto it = transitions.find(layer_name);
    return (it != transitions.end()) ? it->second : 1.0f;
}

Tensor AnimatedBackground::interpolate_color_transition(const std::string& layer_name, const std::map<std::string, Tensor>& transitions) const {
    // Simplified color transition interpolation
    auto it = transitions.find(layer_name);
    return (it != transitions.end()) ? it->second : Tensor({1.0f, 1.0f, 1.0f, 1.0f});
}

// InteractiveBackground implementation
InteractiveBackground::InteractiveBackground() : Background() {
    name = "interactive_background";
}

void InteractiveBackground::set_mouse_reactive(bool reactive) {
    mouse_reactive = reactive;
}

void InteractiveBackground::set_audio_reactive(bool reactive) {
    audio_reactive = reactive;
}

void InteractiveBackground::set_physics_enabled(bool enabled) {
    physics_enabled = enabled;
}

void InteractiveBackground::set_mouse_position(const Tensor& position) {
    mouse_position = position;
}

void InteractiveBackground::set_mouse_influence(float radius, float strength) {
    mouse_influence_radius = std::max(0.0f, radius);
    mouse_influence_strength = std::max(0.0f, strength);
}

void InteractiveBackground::set_audio_level(float level) {
    audio_level = std::clamp(level, 0.0f, 1.0f);
}

void InteractiveBackground::set_frequency_spectrum(const Tensor& spectrum) {
    frequency_spectrum = spectrum;
}

void InteractiveBackground::set_audio_influence_strength(float strength) {
    audio_influence_strength = std::max(0.0f, strength);
}

void InteractiveBackground::set_gravity_center(const Tensor& center) {
    gravity_center = center;
}

void InteractiveBackground::set_gravity_strength(float strength) {
    gravity_strength = std::max(0.0f, strength);
}

void InteractiveBackground::set_friction(float friction_value) {
    friction = std::clamp(friction_value, 0.0f, 1.0f);
}

void InteractiveBackground::update(float delta_time) {
    Background::update(delta_time);
    
    if (mouse_reactive) {
        apply_mouse_interaction(delta_time);
    }
    
    if (audio_reactive) {
        apply_audio_interaction(delta_time);
    }
    
    if (physics_enabled) {
        apply_physics_interaction(delta_time);
    }
}

void InteractiveBackground::apply_mouse_interaction(float delta_time) {
    // Apply mouse influence to nearby layers
    for (auto& layer : layers) {
        if (!layer->visible) continue;
        
        Tensor influence = calculate_mouse_influence(*layer);
        if (influence.size() >= 2) {
            // Apply mouse influence to layer properties
            layer->scroll_offset = layer->scroll_offset + influence * delta_time;
        }
    }
}

void InteractiveBackground::apply_audio_interaction(float delta_time) {
    // Apply audio influence to layers
    for (auto& layer : layers) {
        if (!layer->visible) continue;
        
        Tensor influence = calculate_audio_influence(*layer);
        if (influence.size() >= 2) {
            // Apply audio influence
            layer->animation_speed = layer->animation_speed + influence * audio_influence_strength;
        }
    }
}

void InteractiveBackground::apply_physics_interaction(float delta_time) {
    // Apply physics forces to layers
    for (auto& layer : layers) {
        if (!layer->visible) continue;
        
        Tensor influence = calculate_physics_influence(*layer);
        if (influence.size() >= 2) {
            // Apply physics influence
            layer->scroll_offset = layer->scroll_offset + influence * delta_time;
        }
    }
}

Tensor InteractiveBackground::calculate_mouse_influence(const ParallaxLayer& layer) const {
    // Calculate mouse influence on this layer
    Tensor influence = Tensor({0.0f, 0.0f});
    
    // Simplified: distance-based influence
    float distance = layer.depth * mouse_influence_radius;
    float strength = mouse_influence_strength * (1.0f - layer.depth);
    
    // This would calculate actual mouse position influence
    return influence * strength;
}

Tensor InteractiveBackground::calculate_audio_influence(const ParallaxLayer& layer) const {
    // Calculate audio influence on this layer
    Tensor influence = Tensor({0.0f, 0.0f});
    
    // Simplified: audio level affects animation speed
    influence = Tensor({audio_level * 0.1f, audio_level * 0.1f});
    
    return influence * audio_influence_strength * layer.depth;
}

Tensor InteractiveBackground::calculate_physics_influence(const ParallaxLayer& layer) const {
    // Calculate physics influence (gravity, etc.)
    Tensor influence = Tensor({0.0f, 0.0f});
    
    // Simplified gravity effect
    Tensor direction = gravity_center - layer.scroll_offset;
    float distance = std::sqrt(direction.data_ptr<float>()[0] * direction.data_ptr<float>() [0] + 
                              direction.data_ptr<float>()[1] * direction.data_ptr<float>()[1]);
    
    if (distance > 0.001f) {
        influence = direction / distance * gravity_strength * layer.depth;
    }
    
    return influence;
}

} // namespace scene
} // namespace jin