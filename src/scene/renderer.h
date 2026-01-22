#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include "scene_graph.h"
#include "sprite.h"
#include "layer_stack.h"
#include "compositor.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace jin {
namespace scene {

/**
 * RenderBuffer - Manages render target textures
 */
struct RenderBuffer {
    Tensor color_buffer;      // RGBA color buffer
    Tensor depth_buffer;     // Depth buffer (for 2D, may not be used)
    Tensor stencil_buffer;    // Stencil buffer
    int width = 0;
    int height = 0;
    bool use_depth_buffer = false;
    bool use_stencil_buffer = false;
    
    RenderBuffer();
    RenderBuffer(int width, int height, bool use_depth = false, bool use_stencil = false);
    
    void resize(int new_width, int new_height);
    void clear(const Tensor& color);
    void clear_depth(float depth_value);
    void clear_stencil(int stencil_value);
    
    Tensor get_color_texture() const { return color_buffer; }
};

/**
 * BlitOperations - Hardware-accelerated blit operations
 */
struct BlitOperation {
    enum class Operation {
        COPY,
        BLIT,
        BLEND,
        SCALE,
        ROTATE,
        CONVOLVE
    } operation;
    
    Tensor source_texture;
    Tensor destination_texture;
    Tensor parameters; // Additional parameters for the operation
    
    // Blending parameters
    BlendMode blend_mode = BlendMode::NORMAL;
    float blend_factor = 1.0f;
    
    // Transform parameters
    float rotation_angle = 0.0f;
    Tensor scale_factor = {1.0f, 1.0f};
    Tensor translation = {0.0f, 0.0f};
};

/**
 * Renderer - Simple CPU-based rasterizer
 */
class Renderer {
public:
    // Renderer configuration
    int default_width = 1280;
    int default_height = 720;
    bool enable_vsync = false;
    bool enable_multisampling = false;
    
    // Performance settings
    bool use_caching = true;
    bool use_threading = false;
    int worker_threads = 4;
    
    // Render quality
    enum class RenderQuality {
        FASTEST,      // Nearest neighbor, no filtering
        FAST,         // Bilinear filtering
        QUALITY,     // Trilinear filtering
        ULTRA         // Anisotropic filtering
    };
    
    RenderQuality quality = RenderQuality::QUALITY;
    
    // Antialiasing settings
    bool enable_antialiasing = true;
    int multisample_count = 4;
    
    // Renderer state
    struct RendererState {
        RenderBuffer current_target;
        Camera2D current_camera;
        Transform2D current_transform;
        BlendMode current_blend_mode = BlendMode::NORMAL;
        RenderQuality current_quality = RenderQuality::QUALITY;
        bool scissor_test_enabled = false;
        Tensor scissor_rect = {0.0f, 0.0f, 1.0f, 1.0f}; // x, y, width, height
        bool depth_test_enabled = false;
    };
    
    RendererState state;
    
    Renderer();
    ~Renderer() = default;
    
    // Initialization
    void initialize(int width, int height);
    void shutdown();
    
    // Render target management
    void set_render_target(RenderBuffer& target);
    RenderBuffer& get_current_target() { return state.current_target; }
    void clear_current_target(const Tensor& color);
    
    // Camera management
    void set_current_camera(const Camera2D& camera);
    Camera2D& get_current_camera() { return state.current_camera; }
    
    // Drawing primitives
    void draw_sprite(const Tensor& sprite_image, const Tensor& world_transform, float opacity, BlendMode blend_mode);
    void draw_rectangle(const Tensor& position, const Tensor& size, const Tensor& color);
    void draw_circle(const Tensor& center, float radius, const Tensor& color);
    void draw_line(const Tensor& start, const Tensor& end, float width, const Tensor& color);
    void draw_triangle(const Tensor& v0, const Tensor& v1, const Tensor& v2, const Tensor& color);
    
    // Texture management
    void bind_texture(const Tensor& texture);
    void unbind_texture();
    Tensor create_texture(int width, int height, const Tensor& data);
    void update_texture(const Tensor& texture, const Tensor& data);
    void delete_texture(const Tensor& texture);
    
    // Blending and effects
    void set_blend_mode(BlendMode mode);
    void set_render_quality(RenderQuality quality);
    
    // Scissor and clipping
    void enable_scissor_test(bool enabled);
    void set_scissor_rect(const Tensor& rect);
    
    // Main rendering methods
    Tensor render_frame(SceneGraph& scene, int width, int height, const Tensor& background);
    Tensor render_layered_scene(LayerStack& layer_stack, const Camera2D& camera, int width, int height);
    Tensor render_particle_system(ParticleSystem& particle_system, const Camera2D& camera, int width, int height);
    Tensor render_text(TextRenderer& text_renderer, const Camera2D& camera, int width, int height);
    
    // Utility rendering
    Tensor render_texture(const Tensor& texture, int x, int y, int width, int height);
    Tensor render_texture_with_transform(const Tensor& texture, const Transform2D& transform);
    Tensor render_textured_quad(const Tensor& texture, const Tensor& quad_vertices);
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state);
    temporal::FrameState get_frame_state() const;
    
    // Performance monitoring
    struct RenderStats {
        float last_frame_time = 0.0f;
        float average_frame_time = 0.0f;
        int frames_rendered = 0;
        int draw_calls = 0;
        int triangles_rendered = 0;
        int pixels_rendered = 0;
    };
    
    RenderStats get_render_stats() const;
    void reset_render_stats();
    
    // Debug and profiling
    void enable_debug_mode(bool enabled);
    void set_wireframe_mode(bool enabled);
    void render_debug_info();
    
private:
    temporal::FrameState frame_state_;
    RenderStats render_stats_;
    bool debug_mode_ = false;
    bool wireframe_mode_ = false;
    
    // Cached data
    Tensor cached_camera_matrix_;
    bool camera_matrix_valid_ = false;
    
    // Texture cache
    std::map<size_t, Tensor> texture_cache_;
    
    // Helper methods
    Tensor apply_camera_transform(const Tensor& world_position) const;
    Tensor apply_blend_modes(const Tensor& source, const Tensor& destination, BlendMode blend_mode) const;
    Tensor apply_texture_filtering(const Tensor& texture, RenderQuality quality) const;
    
    // Rendering helpers
    void render_triangle(const Tensor& v0, const Tensor& v1, const Tensor& v2, const Tensor& color);
    void render_sprite_with_filtering(const Tensor& sprite, const Transform2D& transform, float opacity);
    void render_particle_batch(const std::vector<Particle>& particles, const Camera2D& camera);
    
    // Optimizations
    void update_texture_cache();
    void optimize_render_calls();
    
    // Math helpers
    Tensor normalize_coordinates(const Tensor& position, int width, int height) const;
    Tensor denormalize_coordinates(const Tensor& position, int width, int height) const;
    bool is_point_visible(const Tensor& point, int width, int height) const;
    
    // Blending implementations
    Tensor blend_normal(const Tensor& src, const Tensor& dst, float alpha) const;
    Tensor blend_add(const Tensor& src, const Tensor& dst) const;
    Tensor blend_multiply(const Tensor& src, const Tensor& dst) const;
    Tensor blend_screen(const Tensor& src, const Tensor& dst) const;
    Tensor blend_overlay(const Tensor& src, const Tensor& dst) const;
    Tensor blend_soft_light(const Tensor& src, const Tensor& dst) const;
    Tensor blend_hard_light(const Tensor& src, const Tensor& dst) const;
};

/**
 * HardwareRenderer - Hardware-accelerated renderer interface
 */
class HardwareRenderer : public Renderer {
public:
    // Hardware-specific features
    bool supports_compute_shaders = false;
    bool supports_geometry_shaders = false;
    bool supports_tessellation = false;
    bool supports_direct3d = false;
    bool supports_opengl = false;
    bool supports_vulkan = false;
    bool supports_metal = false;
    
    // Shader programs (simplified)
    struct ShaderProgram {
        std::string vertex_shader;
        std::string fragment_shader;
        std::string geometry_shader;
        bool compiled = false;
    };
    
    std::map<std::string, ShaderProgram> shader_programs;
    
    HardwareRenderer();
    ~HardwareRenderer() = default;
    
    // Shader management
    void create_shader_program(const std::string& name, const std::string& vertex_shader, const std::string& fragment_shader);
    void bind_shader_program(const std::string& name);
    void delete_shader_program(const std::string& name);
    
    // Vertex buffers
    Tensor create_vertex_buffer(const Tensor& vertex_data);
    void bind_vertex_buffer(const Tensor& buffer);
    void update_vertex_buffer(const Tensor& buffer, const Tensor& data);
    void delete_vertex_buffer(const Tensor& buffer);
    
    // Index buffers
    Tensor create_index_buffer(const Tensor& index_data);
    void bind_index_buffer(const Tensor& buffer);
    void delete_index_buffer(const Tensor& buffer);
    
    // Uniform buffers
    Tensor create_uniform_buffer(const Tensor& uniform_data);
    void update_uniform_buffer(const Tensor& buffer, const Tensor& data);
    void delete_uniform_buffer(const Tensor& buffer);
    
    // Hardware-specific rendering
    void render_instanced(const Tensor& vertex_buffer, const Tensor& index_buffer, int instance_count);
    void render_with_compute_shader(const std::string& shader_name, const Tensor& compute_data);
    
    // Performance optimizations
    void enable_gpu_culling(bool enabled);
    void set_async_rendering(bool enabled);
    
private:
    void detect_hardware_capabilities();
    void initialize_shaders();
};

/**
 * SoftwareRenderer - Pure software renderer
 */
class SoftwareRenderer : public Renderer {
public:
    // Software rendering optimizations
    bool use_simd_instructions = true;
    bool use_multithreading = true;
    bool use_growing_buffers = true;
    int cache_line_size = 64;
    
    // Memory management
    size_t max_texture_memory = 256 * 1024 * 1024; // 256 MB
    size_t current_texture_memory = 0;
    
    SoftwareRenderer();
    ~SoftwareRenderer() = default;
    
    // Memory optimization
    void set_texture_memory_limit(size_t limit_bytes);
    void clear_unused_textures();
    void compact_texture_memory();
    
    // SIMD optimizations
    void enable_simd_optimizations(bool enabled);
    void set_thread_count(int thread_count);
    
    // Growing buffer management
    void enable_growing_buffers(bool enabled);
    void set_initial_buffer_size(size_t size);
    
    // Software-specific rendering
    void render_software_sprite(const Tensor& sprite, const Transform2D& transform, float opacity);
    void render_software_particles(const std::vector<Particle>& particles, const Camera2D& camera);
    Tensor apply_software_filtering(const Tensor& input, const std::string& filter_type);
    
    // Performance monitoring
    size_t get_texture_memory_usage() const;
    float get_cpu_usage() const;
    int get_active_threads() const;
    
private:
    // SIMD operations (if available)
    void process_pixels_simd(const Tensor& source, Tensor& destination, const std::function<void(float*, float*, size_t)>& operation);
    
    // Threading
    void render_with_multithreading(const Tensor& source, Tensor& destination, const std::function<void(const Tensor&, Tensor&)>& render_func);
    
    // Memory management
    std::vector<Tensor> texture_pool_;
    std::map<size_t, std::vector<Tensor>> free_list_;
    
    void manage_texture_memory();
    void cleanup_unused_resources();
};

} // namespace scene
} // namespace jin