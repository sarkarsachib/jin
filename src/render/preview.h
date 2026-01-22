#pragma once

#include "../core/tensor.h"
#include <string>
#include <vector>
#include <map>
#include <functional>

namespace jin {
namespace render {

// Performance statistics for preview rendering
struct PerformanceStats {
    float avg_frame_time_ms;           // Average frame render time
    float min_frame_time_ms;           // Minimum frame render time
    float max_frame_time_ms;           // Maximum frame render time
    float current_fps;                 // Current rendering FPS
    int frames_dropped;               // Number of frames dropped
    float gpu_memory_mb;              // GPU memory usage (if available)
    int total_frames_rendered;        // Total frames rendered
    float peak_memory_usage_mb;       // Peak memory usage
};

// Overlay types for debug visualization
enum class OverlayType {
    NONE,
    MOTION_VECTORS,
    QUALITY_SCORES,
    SYNC_INFO,
    PERFORMANCE_INFO,
    TIMING_INFO
};

// Preview window configuration
struct PreviewConfig {
    int window_width;                  // Window width in pixels
    int window_height;                 // Window height in pixels
    std::string window_title;          // Window title
    bool enable_vsync;                 // Enable vsync
    bool enable_fullscreen;            // Enable fullscreen mode
    float playback_speed;              // Playback speed multiplier
    bool show_overlay;                 // Show debug overlay
    
    PreviewConfig()
        : window_width(1280), window_height(720),
          window_title("Jin Animation Preview"),
          enable_vsync(true), enable_fullscreen(false),
          playback_speed(1.0f), show_overlay(false) {}
};

// Real-time preview renderer
class PreviewRenderer {
public:
    PreviewRenderer();
    explicit PreviewRenderer(const PreviewConfig& config);
    ~PreviewRenderer();
    
    // Window management
    bool create_preview_window();
    bool create_preview_window(int width, int height, const std::string& title);
    void close_preview_window();
    bool is_window_open() const;
    
    // Frame display
    void display_frame(const Tensor& frame);
    void display_frame_with_overlay(const Tensor& frame, OverlayType overlay_type);
    void display_frame_sequence(const std::vector<Tensor>& frames);
    
    // Playback control
    void play();
    void pause();
    void stop();
    void seek_to_frame(int frame_idx);
    void set_playback_speed(float speed);
    
    // Configuration
    void set_config(const PreviewConfig& config);
    const PreviewConfig& get_config() const;
    void set_resolution(int width, int height);
    std::pair<int, int> get_resolution() const;
    
    // Performance monitoring
    PerformanceStats get_performance_stats() const;
    void reset_performance_stats();
    float get_current_fps() const;
    void record_performance_stats();
    
    // Overlay control
    void enable_overlay(OverlayType overlay_type);
    void disable_overlay();
    void set_overlay_opacity(float opacity);
    
    // Rendering control
    void set_frame_interval(float interval_ms);
    void enable_auto_frame_skip(bool enable);
    void set_max_fps(float max_fps);
    
    // Callbacks
    void set_frame_callback(std::function<void(const Tensor&, int)> callback);
    void set_error_callback(std::function<void(const std::string&)> callback);
    
    // Threading support
    void run_preview_thread();
    void stop_preview_thread();
    bool is_preview_running() const;
    
    // Memory management
    void clear_frame_buffer();
    void reserve_frame_buffer(size_t num_frames);
    
private:
    PreviewConfig config_;
    bool window_open_;
    bool preview_running_;
    bool playback_active_;
    float playback_speed_;
    int current_frame_;
    int total_frames_;
    
    // Frame buffer
    std::vector<Tensor> frame_buffer_;
    size_t buffer_capacity_;
    
    // Performance tracking
    PerformanceStats performance_stats_;
    std::vector<float> frame_times_;
    static constexpr size_t FPS_WINDOW_SIZE = 60; // Rolling window for FPS calculation
    
    // Callbacks
    std::function<void(const Tensor&, int)> frame_callback_;
    std::function<void(const std::string&)> error_callback_;
    
    // Threading
    std::thread preview_thread_;
    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    bool thread_stop_requested_;
    
    // Overlay rendering
    OverlayType current_overlay_;
    float overlay_opacity_;
    
    // Helper methods
    bool initialize_sdl();             // Initialize SDL2
    bool initialize_opengl();          // Initialize OpenGL
    void render_frame_to_window(const Tensor& frame);
    void render_overlay(OverlayType overlay_type);
    void update_performance_stats(float frame_time_ms);
    void calculate_fps();
    
    // Texture management
    uint32_t create_texture_from_tensor(const Tensor& tensor);
    void update_texture(uint32_t texture_id, const Tensor& tensor);
    void delete_texture(uint32_t texture_id);
    
    // Window and GL state
    void* gl_context_;
    void* sdl_window_;
    
    // Shader programs (simplified)
    uint32_t create_default_shader();
    void use_shader(uint32_t shader_id);
    
    // Error handling
    void handle_opengl_error(const std::string& operation);
    void log_error(const std::string& message);
    
    // Utility methods
    Tensor preprocess_frame_for_display(const Tensor& frame);
    std::vector<uint8_t> tensor_to_rgba_bytes(const Tensor& tensor);
    std::vector<float> generate_motion_vectors_overlay(const Tensor& frame);
    std::string format_performance_info() const;
};

// Simplified OpenGL wrapper for preview
class OpenGLWrapper {
public:
    static bool initialize();
    static void cleanup();
    static uint32_t create_texture_2d(int width, int height);
    static void update_texture_data(uint32_t texture_id, const std::vector<uint8_t>& data);
    static void render_texture(uint32_t texture_id, int x, int y, int width, int height);
    static void set_viewport(int x, int y, int width, int height);
    
private:
    static bool initialized_;
};

// Frame buffer management
class FrameBuffer {
public:
    FrameBuffer(size_t capacity = 1000);
    ~FrameBuffer();
    
    void add_frame(const Tensor& frame);
    Tensor get_frame(size_t index) const;
    size_t size() const;
    void clear();
    void reserve(size_t capacity);
    size_t capacity() const { return capacity_; }
    
    // Thread-safe access
    void thread_safe_add_frame(const Tensor& frame);
    Tensor thread_safe_get_frame(size_t index) const;
    
private:
    std::vector<Tensor> frames_;
    size_t capacity_;
    mutable std::mutex mutex_;
};

} // namespace render
} // namespace jin