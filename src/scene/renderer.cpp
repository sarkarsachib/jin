#include "renderer.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <future>
#include <chrono>

namespace jin {
namespace scene {

// RenderBuffer implementation
RenderBuffer::RenderBuffer() {
    width = 0;
    height = 0;
    use_depth_buffer = false;
    use_stencil_buffer = false;
}

RenderBuffer::RenderBuffer(int width, int height, bool use_depth, bool use_stencil) 
    : width(width), height(height), use_depth_buffer(use_depth), use_stencil_buffer(use_stencil) {
    color_buffer = zeros({height, width, 4});
    
    if (use_depth_buffer) {
        depth_buffer = zeros({height, width});
    }
    
    if (use_stencil_buffer) {
        stencil_buffer = zeros({height, width});
    }
}

void RenderBuffer::resize(int new_width, int new_height) {
    width = new_width;
    height = new_height;
    
    color_buffer = zeros({height, width, 4});
    
    if (use_depth_buffer) {
        depth_buffer = zeros({height, width});
    }
    
    if (use_stencil_buffer) {
        stencil_buffer = zeros({height, width});
    }
}

void RenderBuffer::clear(const Tensor& color) {
    if (color.size() >= 4) {
        color_buffer = color_buffer.fill(color.data_ptr<float>()[0]);
        color_buffer = color_buffer.fill(color.data_ptr<float>()[1]);
        color_buffer = color_buffer.fill(color.data_ptr<float>()[2]);
        color_buffer = color_buffer.fill(color.data_ptr<float>()[3]);
    }
}

void RenderBuffer::clear_depth(float depth_value) {
    if (use_depth_buffer && depth_buffer.size() > 0) {
        depth_buffer = depth_buffer.fill(depth_value);
    }
}

void RenderBuffer::clear_stencil(int stencil_value) {
    if (use_stencil_buffer && stencil_buffer.size() > 0) {
        stencil_buffer = stencil_buffer.fill(static_cast<float>(stencil_value));
    }
}

// Renderer implementation
Renderer::Renderer() {
    quality = RenderQuality::QUALITY;
    enable_antialiasing = true;
    multisample_count = 4;
    use_caching = true;
    use_threading = false;
    worker_threads = 4;
}

void Renderer::initialize(int width, int height) {
    default_width = width;
    default_height = height;
    
    state.current_target = RenderBuffer(width, height);
    state.current_camera = Camera2D();
    
    if (use_threading) {
        // Initialize worker threads
        worker_threads = std::max(1, std::thread::hardware_concurrency() / 2);
    }
}

void Renderer::shutdown() {
    // Cleanup resources
    texture_cache_.clear();
}

void Renderer::set_render_target(RenderBuffer& target) {
    state.current_target = target;
}

void Renderer::clear_current_target(const Tensor& color) {
    state.current_target.clear(color);
}

void Renderer::set_current_camera(const Camera2D& camera) {
    state.current_camera = camera;
    camera_matrix_valid_ = false;
}

void Renderer::draw_sprite(const Tensor& sprite_image, const Tensor& world_transform, float opacity, BlendMode blend_mode) {
    if (sprite_image.size() == 0) return;
    
    Transform2D transform;
    if (world_transform.size() >= 6) {
        transform.position = world_transform.view({2}).slice(0, 2);
        transform.rotation = world_transform.view({1}).slice(2, 3);
        transform.scale = world_transform.view({2}).slice(3, 5);
    }
    
    render_sprite_with_filtering(sprite_image, transform, opacity);
}

void Renderer::draw_rectangle(const Tensor& position, const Tensor& size, const Tensor& color) {
    if (position.size() < 2 || size.size() < 2 || color.size() < 4) return;
    
    float* pos_data = position.data_ptr<float>();
    float* size_data = size.data_ptr<float>();
    float* color_data = color.data_ptr<float>();
    
    // Calculate rectangle corners
    Tensor v0 = Tensor({pos_data[0], pos_data[1]});                    // Top-left
    Tensor v1 = Tensor({pos_data[0] + size_data[0], pos_data[1]});     // Top-right
    Tensor v2 = Tensor({pos_data[0] + size_data[0], pos_data[1] - size_data[1]}); // Bottom-right
    Tensor v3 = Tensor({pos_data[0], pos_data[1] - size_data[1]});     // Bottom-left
    
    // Draw two triangles to form the rectangle
    render_triangle(v0, v1, v2, color);
    render_triangle(v0, v2, v3, color);
}

void Renderer::draw_circle(const Tensor& center, float radius, const Tensor& color) {
    if (center.size() < 2 || color.size() < 4) return;
    
    float* center_data = center.data_ptr<float>();
    float* color_data = color.data_ptr<float>();
    
    // Approximate circle with triangles
    int segments = 16;
    Tensor prev_vertex = Tensor({center_data[0] + radius, center_data[1]});
    
    for (int i = 1; i <= segments; ++i) {
        float angle = (2.0f * 3.14159f * i) / segments;
        Tensor current_vertex = Tensor({
            center_data[0] + std::cos(angle) * radius,
            center_data[1] + std::sin(angle) * radius
        });
        
        render_triangle(Tensor(center_data[0], center_data[1]), prev_vertex, current_vertex, color);
        prev_vertex = current_vertex;
    }
}

void Renderer::draw_line(const Tensor& start, const Tensor& end, float width, const Tensor& color) {
    if (start.size() < 2 || end.size() < 2 || color.size() < 4) return;
    
    float* start_data = start.data_ptr<float>();
    float* end_data = end.data_ptr<float>();
    float* color_data = color.data_ptr<float>();
    
    // Calculate line direction
    Tensor direction = end - start;
    float line_length = std::sqrt(direction.data_ptr<float>()[0] * direction.data_ptr<float>()[0] + 
                                  direction.data_ptr<float>()[1] * direction.data_ptr<float>()[1]);
    
    if (line_length < 0.001f) return;
    
    // Normalize direction
    direction = direction / line_length;
    
    // Calculate perpendicular direction for line width
    Tensor perpendicular = Tensor({-direction.data_ptr<float>()[1], direction.data_ptr<float>()[0]});
    Tensor half_width = perpendicular * (width / 2.0f);
    
    // Calculate rectangle corners
    Tensor v0 = start + half_width;
    Tensor v1 = start - half_width;
    Tensor v2 = end - half_width;
    Tensor v3 = end + half_width;
    
    // Draw the line as a quad (two triangles)
    render_triangle(v0, v1, v2, color);
    render_triangle(v0, v2, v3, color);
}

void Renderer::draw_triangle(const Tensor& v0, const Tensor& v1, const Tensor& v2, const Tensor& color) {
    render_triangle(v0, v1, v2, color);
}

void Renderer::bind_texture(const Tensor& texture) {
    // Bind texture for rendering
    // In a real implementation, this would set up texture parameters
}

void Renderer::unbind_texture() {
    // Unbind current texture
}

Tensor Renderer::create_texture(int width, int height, const Tensor& data) {
    if (data.size() != static_cast<size_t>(width * height * 4)) {
        // Invalid data size
        return Tensor();
    }
    
    // Create texture and add to cache
    Tensor texture = data.clone();
    
    // Add to texture cache
    size_t hash = texture.storage()->data() ? 
        std::hash<std::string>{}(std::string(static_cast<const char*>(texture.storage()->data()), texture.size() * sizeof(float))) : 
        0;
    texture_cache_[hash] = texture;
    
    return texture;
}

void Renderer::update_texture(const Tensor& texture, const Tensor& data) {
    // Update texture data
}

void Renderer::delete_texture(const Tensor& texture) {
    // Remove from cache
    auto it = std::find_if(texture_cache_.begin(), texture_cache_.end(),
        [&texture](const auto& pair) {
            return pair.second.storage() == texture.storage();
        });
    
    if (it != texture_cache_.end()) {
        texture_cache_.erase(it);
    }
}

void Renderer::set_blend_mode(BlendMode mode) {
    state.current_blend_mode = mode;
}

void Renderer::set_render_quality(RenderQuality quality) {
    state.current_quality = quality;
}

void Renderer::enable_scissor_test(bool enabled) {
    state.scissor_test_enabled = enabled;
}

void Renderer::set_scissor_rect(const Tensor& rect) {
    state.scissor_rect = rect;
}

Tensor Renderer::render_frame(SceneGraph& scene, int width, int height, const Tensor& background) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear target
    clear_current_target(background);
    
    // Render scene graph
    Tensor result = zeros({height, width, 4});
    
    // Update render stats
    render_stats_.frames_rendered++;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    render_stats_.last_frame_time = duration.count() / 1000.0f; // Convert to milliseconds
    
    // Update average frame time
    render_stats_.average_frame_time = (render_stats_.average_frame_time * (render_stats_.frames_rendered - 1) + 
                                      render_stats_.last_frame_time) / render_stats_.frames_rendered;
    
    return result;
}

Tensor Renderer::render_layered_scene(LayerStack& layer_stack, const Camera2D& camera, int width, int height) {
    Tensor result = zeros({height, width, 4});
    
    auto render_order = layer_stack.get_render_order();
    
    for (Layer* layer : render_order) {
        if (layer->visible && layer->node) {
            Tensor layer_render = layer->node->render(width, height, camera);
            result = blend_textures(result, layer_render, layer->blend_mode, layer->opacity);
        }
    }
    
    return result;
}

Tensor Renderer::render_particle_system(ParticleSystem& particle_system, const Camera2D& camera, int width, int height) {
    return particle_system.render(width, height, camera);
}

Tensor Renderer::render_text(TextRenderer& text_renderer, const Camera2D& camera, int width, int height) {
    return text_renderer.render(width, height, camera);
}

Tensor Renderer::render_texture(const Tensor& texture, int x, int y, int width, int height) {
    Tensor result = zeros({height, width, 4});
    
    // Simple texture rendering
    if (texture.size() > 0) {
        // Calculate scaling factors
        float scale_x = static_cast<float>(width) / texture.shape()[1];
        float scale_y = static_cast<float>(height) / texture.shape()[0];
        
        // Render texture with scaling
        for (int py = 0; py < height && py < this->default_height; ++py) {
            for (int px = 0; px < width && px < this->default_width; ++px) {
                int src_x = static_cast<int>(px / scale_x);
                int src_y = static_cast<int>(py / scale_y);
                
                src_x = std::min(src_x, texture.shape()[1] - 1);
                src_y = std::min(src_y, texture.shape()[0] - 1);
                
                size_t src_idx = (src_y * texture.shape()[1] + src_x) * 4;
                size_t dst_idx = (py * width + px) * 4;
                
                if (src_idx + 3 < texture.size() && dst_idx + 3 < result.size()) {
                    const float* src_data = texture.data_ptr<float>();
                    float* dst_data = result.data_ptr<float>();
                    
                    for (int c = 0; c < 4; ++c) {
                        dst_data[dst_idx + c] = src_data[src_idx + c];
                    }
                }
            }
        }
    }
    
    return result;
}

Tensor Renderer::render_texture_with_transform(const Tensor& texture, const Transform2D& transform) {
    // Apply transform to texture rendering
    // This is a simplified implementation
    return texture;
}

Tensor Renderer::render_textured_quad(const Tensor& texture, const Tensor& quad_vertices) {
    // Render textured quad
    // Simplified implementation
    return texture;
}

void Renderer::set_frame_state(const temporal::FrameState& frame_state) {
    frame_state_ = frame_state;
}

temporal::FrameState Renderer::get_frame_state() const {
    return frame_state_;
}

Renderer::RenderStats Renderer::get_render_stats() const {
    return render_stats_;
}

void Renderer::reset_render_stats() {
    render_stats_ = RenderStats{};
}

void Renderer::enable_debug_mode(bool enabled) {
    debug_mode_ = enabled;
}

void Renderer::set_wireframe_mode(bool enabled) {
    wireframe_mode_ = enabled;
}

void Renderer::render_debug_info() {
    // Render debug information overlay
    if (debug_mode_) {
        // Draw FPS, render stats, etc.
    }
}

Tensor Renderer::apply_camera_transform(const Tensor& world_position) const {
    return state.current_camera.world_to_screen(world_position);
}

Tensor Renderer::apply_blend_modes(const Tensor& source, const Tensor& destination, BlendMode blend_mode) const {
    return apply_blend_mode(source, destination, blend_mode);
}

Tensor Renderer::apply_texture_filtering(const Tensor& texture, RenderQuality quality) const {
    // Apply texture filtering based on quality setting
    switch (quality) {
        case RenderQuality::FASTEST:
            // Nearest neighbor filtering
            return texture;
        case RenderQuality::FAST:
            // Bilinear filtering
            return texture;
        case RenderQuality::QUALITY:
            // Trilinear filtering
            return texture;
        case RenderQuality::ULTRA:
            // Anisotropic filtering
            return texture;
        default:
            return texture;
    }
}

void Renderer::render_triangle(const Tensor& v0, const Tensor& v1, const Tensor& v2, const Tensor& color) {
    if (wireframe_mode_) {
        // Render as wireframe
        draw_line(v0, v1, 1.0f, color);
        draw_line(v1, v2, 1.0f, color);
        draw_line(v2, v0, 1.0f, color);
        return;
    }
    
    // Apply camera transform to vertices
    Tensor screen_v0 = apply_camera_transform(v0);
    Tensor screen_v1 = apply_camera_transform(v1);
    Tensor screen_v2 = apply_camera_transform(v2);
    
    // Rasterize triangle
    if (enable_antialiasing) {
        render_triangle_antialiased(screen_v0, screen_v1, screen_v2, color);
    } else {
        render_triangle_aliased(screen_v0, screen_v1, screen_v2, color);
    }
}

void Renderer::render_sprite_with_filtering(const Tensor& sprite, const Transform2D& transform, float opacity) {
    Tensor world_transform_tensor = transform.get_transform_matrix();
    Tensor sprite_image = apply_texture_filtering(sprite, state.current_quality);
    
    // Apply opacity
    if (opacity < 1.0f) {
        sprite_image = sprite_image * opacity;
    }
    
    // Draw sprite as textured quad
    render_texture_with_transform(sprite_image, transform);
}

void Renderer::render_particle_batch(const std::vector<Particle>& particles, const Camera2D& camera) {
    // Batch render particles for performance
    for (const auto& particle : particles) {
        if (!should_cull_particle(particle, camera)) {
            Tensor screen_pos = apply_camera_transform(particle.position);
            draw_circle(screen_pos, particle.size, particle.color);
        }
    }
}

void Renderer::update_texture_cache() {
    // Update texture cache for performance
    if (!use_caching) return;
    
    // Remove unused textures from cache
    // This would involve tracking texture usage
}

void Renderer::optimize_render_calls() {
    // Optimize render calls for better performance
    // Group similar render calls, batch operations, etc.
}

Tensor Renderer::normalize_coordinates(const Tensor& position, int width, int height) const {
    // Convert screen coordinates to normalized device coordinates
    Tensor normalized = position;
    if (position.size() >= 2) {
        normalized.data_ptr<float>()[0] = (position.data_ptr<float>()[0] / width) * 2.0f - 1.0f;
        normalized.data_ptr<float>()[1] = (position.data_ptr<float>()[1] / height) * 2.0f - 1.0f;
    }
    return normalized;
}

Tensor Renderer::denormalize_coordinates(const Tensor& position, int width, int height) const {
    // Convert normalized device coordinates to screen coordinates
    Tensor denormalized = position;
    if (position.size() >= 2) {
        denormalized.data_ptr<float>()[0] = (position.data_ptr<float>()[0] + 1.0f) * width / 2.0f;
        denormalized.data_ptr<float>()[1] = (position.data_ptr<float>()[1] + 1.0f) * height / 2.0f;
    }
    return denormalized;
}

bool Renderer::is_point_visible(const Tensor& point, int width, int height) const {
    if (point.size() < 2) return false;
    
    float x = point.data_ptr<float>()[0];
    float y = point.data_ptr<float>()[1];
    
    return (x >= 0.0f && x < width && y >= 0.0f && y < height);
}

Tensor Renderer::blend_normal(const Tensor& src, const Tensor& dst, float alpha) const {
    Tensor result = dst.clone();
    size_t pixel_count = std::min(src.size(), dst.size()) / 4;
    const float* src_data = src.data_ptr<float>();
    float* dst_data = result.data_ptr<float>();
    
    for (size_t i = 0; i < pixel_count; ++i) {
        float src_alpha = src_data[i * 4 + 3] * alpha;
        dst_data[i * 4] = src_data[i * 4] * src_alpha + dst_data[i * 4] * (1.0f - src_alpha);
        dst_data[i * 4 + 1] = src_data[i * 4 + 1] * src_alpha + dst_data[i * 4 + 1] * (1.0f - src_alpha);
        dst_data[i * 4 + 2] = src_data[i * 4 + 2] * src_alpha + dst_data[i * 4 + 2] * (1.0f - src_alpha);
        dst_data[i * 4 + 3] = std::max(src_data[i * 4 + 3], dst_data[i * 4 + 3]);
    }
    
    return result;
}

Tensor Renderer::blend_add(const Tensor& src, const Tensor& dst) const {
    Tensor result = dst.clone();
    size_t pixel_count = std::min(src.size(), dst.size()) / 4;
    const float* src_data = src.data_ptr<float>();
    float* dst_data = result.data_ptr<float>();
    
    for (size_t i = 0; i < pixel_count; ++i) {
        for (int c = 0; c < 4; ++c) {
            dst_data[i * 4 + c] += src_data[i * 4 + c];
        }
    }
    
    return result;
}

Tensor Renderer::blend_multiply(const Tensor& src, const Tensor& dst) const {
    Tensor result = dst.clone();
    size_t pixel_count = std::min(src.size(), dst.size()) / 4;
    const float* src_data = src.data_ptr<float>();
    float* dst_data = result.data_ptr<float>();
    
    for (size_t i = 0; i < pixel_count; ++i) {
        for (int c = 0; c < 4; ++c) {
            dst_data[i * 4 + c] *= src_data[i * 4 + c];
        }
    }
    
    return result;
}

Tensor Renderer::blend_screen(const Tensor& src, const Tensor& dst) const {
    Tensor result = dst.clone();
    size_t pixel_count = std::min(src.size(), dst.size()) / 4;
    const float* src_data = src.data_ptr<float>();
    float* dst_data = result.data_ptr<float>();
    
    for (size_t i = 0; i < pixel_count; ++i) {
        for (int c = 0; c < 4; ++c) {
            dst_data[i * 4 + c] = 1.0f - (1.0f - src_data[i * 4 + c]) * (1.0f - dst_data[i * 4 + c]);
        }
    }
    
    return result;
}

Tensor Renderer::blend_overlay(const Tensor& src, const Tensor& dst) const {
    Tensor result = dst.clone();
    size_t pixel_count = std::min(src.size(), dst.size()) / 4;
    const float* src_data = src.data_ptr<float>();
    float* dst_data = result.data_ptr<float>();
    
    for (size_t i = 0; i < pixel_count; ++i) {
        for (int c = 0; c < 3; ++c) { // Only RGB, not alpha
            float dst_val = dst_data[i * 4 + c];
            if (dst_val < 0.5f) {
                dst_data[i * 4 + c] = 2.0f * src_data[i * 4 + c] * dst_val;
            } else {
                dst_data[i * 4 + c] = 1.0f - 2.0f * (1.0f - src_data[i * 4 + c]) * (1.0f - dst_val);
            }
        }
    }
    
    return result;
}

Tensor Renderer::blend_soft_light(const Tensor& src, const Tensor& dst) const {
    Tensor result = dst.clone();
    size_t pixel_count = std::min(src.size(), dst.size()) / 4;
    const float* src_data = src.data_ptr<float>();
    float* dst_data = result.data_ptr<float>();
    
    for (size_t i = 0; i < pixel_count; ++i) {
        for (int c = 0; c < 3; ++c) { // Only RGB, not alpha
            float dst_val = dst_data[i * 4 + c];
            float src_val = src_data[i * 4 + c];
            
            if (src_val < 0.5f) {
                dst_data[i * 4 + c] = dst_val - (1.0f - 2.0f * src_val) * dst_val * (1.0f - dst_val);
            } else {
                float new_val = dst_val;
                if (dst_val < 0.25f) {
                    new_val = ((16.0f * dst_val - 12.0f) * dst_val + 4.0f) * dst_val;
                } else {
                    new_val = std::sqrt(dst_val);
                }
                dst_data[i * 4 + c] = dst_val + (2.0f * src_val - 1.0f) * (new_val - dst_val);
            }
        }
    }
    
    return result;
}

Tensor Renderer::blend_hard_light(const Tensor& src, const Tensor& dst) const {
    Tensor result = dst.clone();
    size_t pixel_count = std::min(src.size(), dst.size()) / 4;
    const float* src_data = src.data_ptr<float>();
    float* dst_data = result.data_ptr<float>();
    
    for (size_t i = 0; i < pixel_count; ++i) {
        for (int c = 0; c < 3; ++c) { // Only RGB, not alpha
            float dst_val = dst_data[i * 4 + c];
            float src_val = src_data[i * 4 + c];
            
            if (src_val < 0.5f) {
                dst_data[i * 4 + c] = 2.0f * src_val * dst_val;
            } else {
                dst_data[i * 4 + c] = 1.0f - 2.0f * (1.0f - src_val) * (1.0f - dst_val);
            }
        }
    }
    
    return result;
}

void Renderer::render_triangle_aliased(const Tensor& v0, const Tensor& v1, const Tensor& v2, const Tensor& color) {
    // Aliased triangle rasterization
    // Simplified implementation - fill bounding box and check barycentric coordinates
}

void Renderer::render_triangle_antialiased(const Tensor& v0, const Tensor& v1, const Tensor& v2, const Tensor& color) {
    // Antialiased triangle rasterization
    // Use multisampling for smooth edges
}

bool Renderer::should_cull_particle(const Particle& particle, const Camera2D& camera) const {
    Tensor screen_pos = apply_camera_transform(particle.position);
    return !is_point_visible(screen_pos, default_width, default_height);
}

// HardwareRenderer implementation
HardwareRenderer::HardwareRenderer() : Renderer() {
    detect_hardware_capabilities();
    initialize_shaders();
}

void HardwareRenderer::create_shader_program(const std::string& name, const std::string& vertex_shader, const std::string& fragment_shader) {
    ShaderProgram program;
    program.vertex_shader = vertex_shader;
    program.fragment_shader = fragment_shader;
    program.compiled = true; // Simplified - assume compilation succeeds
    
    shader_programs[name] = program;
}

void HardwareRenderer::bind_shader_program(const std::string& name) {
    auto it = shader_programs.find(name);
    if (it != shader_programs.end() && it->second.compiled) {
        // Bind shader program
    }
}

void HardwareRenderer::delete_shader_program(const std::string& name) {
    shader_programs.erase(name);
}

Tensor HardwareRenderer::create_vertex_buffer(const Tensor& vertex_data) {
    // Create hardware vertex buffer
    return vertex_data.clone();
}

void HardwareRenderer::bind_vertex_buffer(const Tensor& buffer) {
    // Bind vertex buffer for rendering
}

void HardwareRenderer::update_vertex_buffer(const Tensor& buffer, const Tensor& data) {
    // Update vertex buffer data
}

void HardwareRenderer::delete_vertex_buffer(const Tensor& buffer) {
    // Delete vertex buffer
}

Tensor HardwareRenderer::create_index_buffer(const Tensor& index_data) {
    // Create hardware index buffer
    return index_data.clone();
}

void HardwareRenderer::bind_index_buffer(const Tensor& buffer) {
    // Bind index buffer for rendering
}

void HardwareRenderer::delete_index_buffer(const Tensor& buffer) {
    // Delete index buffer
}

Tensor HardwareRenderer::create_uniform_buffer(const Tensor& uniform_data) {
    // Create hardware uniform buffer
    return uniform_data.clone();
}

void HardwareRenderer::update_uniform_buffer(const Tensor& buffer, const Tensor& data) {
    // Update uniform buffer data
}

void HardwareRenderer::delete_uniform_buffer(const Tensor& buffer) {
    // Delete uniform buffer
}

void HardwareRenderer::render_instanced(const Tensor& vertex_buffer, const Tensor& index_buffer, int instance_count) {
    // Render multiple instances efficiently
}

void HardwareRenderer::render_with_compute_shader(const std::string& shader_name, const Tensor& compute_data) {
    // Render using compute shader
}

void HardwareRenderer::enable_gpu_culling(bool enabled) {
    // Enable GPU-based culling for performance
}

void HardwareRenderer::set_async_rendering(bool enabled) {
    // Enable asynchronous rendering
}

void HardwareRenderer::detect_hardware_capabilities() {
    // Detect available GPU APIs
    supports_opengl = true; // Assume OpenGL support for now
    supports_direct3d = false; // Would detect based on platform
    supports_vulkan = false; // Would detect based on platform
    supports_metal = false; // Would detect based on platform
}

void HardwareRenderer::initialize_shaders() {
    // Initialize default shader programs
    create_shader_program("default", 
        "#version 330 core\n"
        "layout(location = 0) in vec3 aPos;\n"
        "void main() {\n"
        "    gl_Position = vec4(aPos, 1.0);\n"
        "}\n",
        "#version 330 core\n"
        "out vec4 FragColor;\n"
        "void main() {\n"
        "    FragColor = vec4(1.0, 0.5, 0.2, 1.0);\n"
        "}\n"
    );
}

// SoftwareRenderer implementation
SoftwareRenderer::SoftwareRenderer() : Renderer() {
    use_simd_instructions = true;
    use_multithreading = true;
    use_growing_buffers = true;
    max_texture_memory = 256 * 1024 * 1024; // 256 MB
}

void SoftwareRenderer::set_texture_memory_limit(size_t limit_bytes) {
    max_texture_memory = limit_bytes;
}

void SoftwareRenderer::clear_unused_textures() {
    // Remove unused textures from cache
    manage_texture_memory();
}

void SoftwareRenderer::compact_texture_memory() {
    // Compact texture memory by moving data
    cleanup_unused_resources();
}

void SoftwareRenderer::enable_simd_optimizations(bool enabled) {
    use_simd_instructions = enabled;
}

void SoftwareRenderer::set_thread_count(int thread_count) {
    worker_threads = std::max(1, thread_count);
}

void SoftwareRenderer::enable_growing_buffers(bool enabled) {
    use_growing_buffers = enabled;
}

void SoftwareRenderer::set_initial_buffer_size(size_t size) {
    // Set initial size for growing buffers
}

void SoftwareRenderer::render_software_sprite(const Tensor& sprite, const Transform2D& transform, float opacity) {
    // Optimized sprite rendering for software rasterizer
    render_sprite_with_filtering(sprite, transform, opacity);
}

void SoftwareRenderer::render_software_particles(const std::vector<Particle>& particles, const Camera2D& camera) {
    if (use_multithreading && particles.size() > 100) {
        // Multi-threaded particle rendering
        render_particle_batch_multithreaded(particles, camera);
    } else {
        // Single-threaded particle rendering
        render_particle_batch(particles, camera);
    }
}

Tensor SoftwareRenderer::apply_software_filtering(const Tensor& input, const std::string& filter_type) {
    // Apply various filters to textures
    if (filter_type == "gaussian_blur") {
        return apply_gaussian_blur(input);
    } else if (filter_type == "sharpen") {
        return apply_sharpen_filter(input);
    } else if (filter_type == "edge_detect") {
        return apply_edge_detection(input);
    }
    
    return input;
}

size_t SoftwareRenderer::get_texture_memory_usage() const {
    return current_texture_memory;
}

float SoftwareRenderer::get_cpu_usage() const {
    // Return CPU usage percentage
    return 0.0f; // Simplified
}

int SoftwareRenderer::get_active_threads() const {
    return use_multithreading ? worker_threads : 1;
}

void SoftwareRenderer::process_pixels_simd(const Tensor& source, Tensor& destination, const std::function<void(float*, float*, size_t)>& operation) {
    // SIMD-optimized pixel processing
    // This would use SIMD instructions like SSE, AVX, etc.
}

void SoftwareRenderer::render_with_multithreading(const Tensor& source, Tensor& destination, const std::function<void(const Tensor&, Tensor&)>& render_func) {
    if (!use_multithreading || source.shape()[0] < 100) {
        // Single-threaded rendering
        render_func(source, destination);
        return;
    }
    
    // Multi-threaded rendering
    int num_threads = worker_threads;
    int height_per_thread = source.shape()[0] / num_threads;
    
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < num_threads; ++i) {
        int start_y = i * height_per_thread;
        int end_y = (i == num_threads - 1) ? source.shape()[0] : (i + 1) * height_per_thread;
        
        futures.push_back(std::async(std::launch::async, [this, &source, &destination, &render_func, start_y, end_y]() {
            // Process rows start_y to end_y
            for (int y = start_y; y < end_y; ++y) {
                // Process each pixel in the row
                // This is a simplified implementation
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
}

void SoftwareRenderer::manage_texture_memory() {
    // Manage texture memory usage
    if (current_texture_memory > max_texture_memory) {
        // Remove least recently used textures
        cleanup_unused_resources();
    }
}

void SoftwareRenderer::cleanup_unused_resources() {
    // Cleanup unused textures and buffers
    // This would involve reference counting and garbage collection
}

void SoftwareRenderer::render_particle_batch_multithreaded(const std::vector<Particle>& particles, const Camera2D& camera) {
    // Multi-threaded particle batch rendering
    if (use_multithreading) {
        render_with_multithreading(Tensor(), Tensor(), [&](const Tensor& source, Tensor& dest) {
            render_particle_batch(particles, camera);
        });
    } else {
        render_particle_batch(particles, camera);
    }
}

Tensor SoftwareRenderer::apply_gaussian_blur(const Tensor& input) {
    // Apply Gaussian blur filter
    // Simplified implementation
    return input;
}

Tensor SoftwareRenderer::apply_sharpen_filter(const Tensor& input) {
    // Apply sharpen filter
    // Simplified implementation
    return input;
}

Tensor SoftwareRenderer::apply_edge_detection(const Tensor& input) {
    // Apply edge detection filter
    // Simplified implementation
    return input;
}

} // namespace scene
} // namespace jin