#include "preview.h"
#include "../utils/math.h"
#include "../autodiff/tape.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>

namespace jin {
namespace render {

// Static member initialization
bool OpenGLWrapper::initialized_ = false;

PreviewRenderer::PreviewRenderer() : PreviewRenderer(PreviewConfig{}) {}

PreviewRenderer::PreviewRenderer(const PreviewConfig& config) 
    : config_(config), window_open_(false), preview_running_(false), playback_active_(false),
      playback_speed_(1.0f), current_frame_(0), total_frames_(0),
      buffer_capacity_(1000), performance_stats_{}, thread_stop_requested_(false),
      current_overlay_(OverlayType::NONE), overlay_opacity_(1.0f) {
    
    // Initialize performance stats
    performance_stats_.avg_frame_time_ms = 0.0f;
    performance_stats_.min_frame_time_ms = 0.0f;
    performance_stats_.max_frame_time_ms = 0.0f;
    performance_stats_.current_fps = 0.0f;
    performance_stats_.frames_dropped = 0;
    performance_stats_.gpu_memory_mb = 0.0f;
    performance_stats_.total_frames_rendered = 0;
    performance_stats_.peak_memory_usage_mb = 0.0f;
    
    frame_times_.reserve(FPS_WINDOW_SIZE);
}

PreviewRenderer::~PreviewRenderer() {
    close_preview_window();
    stop_preview_thread();
}

bool PreviewRenderer::create_preview_window() {
    return create_preview_window(config_.window_width, config_.window_height, config_.window_title);
}

bool PreviewRenderer::create_preview_window(int width, int height, const std::string& title) {
    if (window_open_) {
        std::cerr << "Window already exists!" << std::endl;
        return false;
    }
    
    // Simplified window creation - in a full implementation this would use SDL2
    std::cout << "Creating preview window: " << width << "x" << height << " - " << title << std::endl;
    
    // Simulate window creation success
    window_open_ = true;
    current_frame_ = 0;
    
    // Initialize OpenGL
    if (!initialize_opengl()) {
        std::cerr << "Failed to initialize OpenGL" << std::endl;
        return false;
    }
    
    return true;
}

void PreviewRenderer::close_preview_window() {
    if (window_open_) {
        std::cout << "Closing preview window" << std::endl;
        window_open_ = false;
        
        // Cleanup OpenGL resources
        cleanup_opengl();
    }
}

bool PreviewRenderer::is_window_open() const {
    return window_open_;
}

void PreviewRenderer::display_frame(const Tensor& frame) {
    if (!window_open_) {
        std::cerr << "Window is not open!" << std::endl;
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Preprocess frame for display
    Tensor processed_frame = preprocess_frame_for_display(frame);
    
    // Render frame to window
    render_frame_to_window(processed_frame);
    
    // Update performance stats
    auto end_time = std::chrono::high_resolution_clock::now();
    float frame_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    update_performance_stats(frame_time_ms);
    
    // Render overlay if enabled
    if (config_.show_overlay && current_overlay_ != OverlayType::NONE) {
        render_overlay(current_overlay_);
    }
    
    // Swap buffers (simplified)
    swap_buffers();
}

void PreviewRenderer::display_frame_with_overlay(const Tensor& frame, OverlayType overlay_type) {
    current_overlay_ = overlay_type;
    display_frame(frame);
}

void PreviewRenderer::display_frame_sequence(const std::vector<Tensor>& frames) {
    if (frames.empty()) {
        std::cerr << "No frames to display!" << std::endl;
        return;
    }
    
    for (size_t i = 0; i < frames.size(); ++i) {
        display_frame(frames[i]);
        
        // Frame timing
        float frame_interval = 1000.0f / config_.fps / playback_speed_;
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(frame_interval)));
        
        if (!playback_active_) {
            break;
        }
    }
}

void PreviewRenderer::play() {
    playback_active_ = true;
    std::cout << "Playing animation at " << playback_speed_ << "x speed" << std::endl;
}

void PreviewRenderer::pause() {
    playback_active_ = false;
    std::cout << "Paused playback" << std::endl;
}

void PreviewRenderer::stop() {
    playback_active_ = false;
    current_frame_ = 0;
    std::cout << "Stopped playback" << std::endl;
}

void PreviewRenderer::seek_to_frame(int frame_idx) {
    if (frame_idx >= 0 && frame_idx < static_cast<int>(frame_buffer_.size())) {
        current_frame_ = frame_idx;
        std::cout << "Seeked to frame " << frame_idx << std::endl;
    }
}

void PreviewRenderer::set_playback_speed(float speed) {
    playback_speed_ = std::clamp(speed, 0.1f, 5.0f);
    std::cout << "Playback speed set to " << playback_speed_ << "x" << std::endl;
}

PerformanceStats PreviewRenderer::get_performance_stats() const {
    return performance_stats_;
}

void PreviewRenderer::reset_performance_stats() {
    performance_stats_ = PerformanceStats{};
    frame_times_.clear();
}

float PreviewRenderer::get_current_fps() const {
    return performance_stats_.current_fps;
}

void PreviewRenderer::enable_overlay(OverlayType overlay_type) {
    current_overlay_ = overlay_type;
    config_.show_overlay = true;
}

void PreviewRenderer::disable_overlay() {
    current_overlay_ = OverlayType::NONE;
    config_.show_overlay = false;
}

void PreviewRenderer::run_preview_thread() {
    if (preview_running_) {
        return;
    }
    
    preview_running_ = true;
    thread_stop_requested_ = false;
    
    preview_thread_ = std::thread([this]() {
        while (!thread_stop_requested_) {
            if (playback_active_ && current_frame_ < static_cast<int>(frame_buffer_.size())) {
                // Get current frame
                Tensor current_frame_tensor;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex_);
                    if (current_frame_ < static_cast<int>(frame_buffer_.size())) {
                        current_frame_tensor = frame_buffer_[current_frame_];
                    }
                }
                
                if (!current_frame_tensor.is_empty()) {
                    display_frame(current_frame_tensor);
                    current_frame_++;
                }
                
                // Frame timing
                float frame_interval = 1000.0f / config_.fps / playback_speed_;
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(frame_interval)));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
            }
        }
    });
}

void PreviewRenderer::stop_preview_thread() {
    if (preview_running_) {
        thread_stop_requested_ = true;
        if (preview_thread_.joinable()) {
            preview_thread_.join();
        }
        preview_running_ = false;
    }
}

bool PreviewRenderer::is_preview_running() const {
    return preview_running_;
}

bool PreviewRenderer::initialize_opengl() {
    std::cout << "Initializing OpenGL (simplified)" << std::endl;
    // In a full implementation, this would initialize SDL2 + OpenGL
    return true;
}

void PreviewRenderer::cleanup_opengl() {
    std::cout << "Cleaning up OpenGL resources" << std::endl;
    // Cleanup OpenGL resources
}

void PreviewRenderer::render_frame_to_window(const Tensor& frame) {
    // Simplified frame rendering
    std::cout << "Rendering frame " << current_frame_ << " to window" << std::endl;
    
    performance_stats_.total_frames_rendered++;
    
    // Call frame callback if set
    if (frame_callback_) {
        frame_callback_(frame, current_frame_);
    }
}

void PreviewRenderer::render_overlay(OverlayType overlay_type) {
    if (overlay_type == OverlayType::PERFORMANCE_INFO) {
        std::string overlay_text = format_performance_info();
        std::cout << "Overlay: " << overlay_text << std::endl;
    } else if (overlay_type == OverlayType::QUALITY_SCORES) {
        std::cout << "Quality overlay rendering..." << std::endl;
    } else if (overlay_type == OverlayType::MOTION_VECTORS) {
        std::cout << "Motion vectors overlay rendering..." << std::endl;
    }
}

void PreviewRenderer::update_performance_stats(float frame_time_ms) {
    // Update rolling window of frame times
    frame_times_.push_back(frame_time_ms);
    if (frame_times_.size() > FPS_WINDOW_SIZE) {
        frame_times_.erase(frame_times_.begin());
    }
    
    // Calculate statistics
    if (!frame_times_.empty()) {
        float total_time = std::accumulate(frame_times_.begin(), frame_times_.end(), 0.0f);
        performance_stats_.avg_frame_time_ms = total_time / frame_times_.size();
        performance_stats_.min_frame_time_ms = *std::min_element(frame_times_.begin(), frame_times_.end());
        performance_stats_.max_frame_time_ms = *std::max_element(frame_times_.begin(), frame_times_.end());
        
        // Calculate FPS
        calculate_fps();
    }
}

void PreviewRenderer::calculate_fps() {
    if (!frame_times_.empty()) {
        float avg_frame_time = performance_stats_.avg_frame_time_ms;
        if (avg_frame_time > 0) {
            performance_stats_.current_fps = 1000.0f / avg_frame_time;
        }
    }
}

Tensor PreviewRenderer::preprocess_frame_for_display(const Tensor& frame) {
    // Ensure frame is in displayable format
    Tensor processed = frame.clone();
    
    // Normalize to [0, 1] if needed
    if (processed.max().item<float>() > 1.0f) {
        processed = processed / 255.0f;
    }
    
    // Ensure correct dimensions (NCHW or NHWC to display format)
    // This is a simplified preprocessing
    return processed;
}

void PreviewRenderer::swap_buffers() {
    // Simplified buffer swap
    // In a full implementation, this would swap OpenGL buffers
}

std::string PreviewRenderer::format_performance_info() const {
    std::stringstream ss;
    ss << "FPS: " << performance_stats_.current_fps << " | "
       << "Frame Time: " << performance_stats_.avg_frame_time_ms << "ms | "
       << "Frames: " << performance_stats_.total_frames_rendered;
    return ss.str();
}

// OpenGLWrapper implementation
bool OpenGLWrapper::initialize() {
    if (initialized_) {
        return true;
    }
    
    std::cout << "Initializing OpenGL wrapper (simplified)" << std::endl;
    // In a full implementation, this would initialize actual OpenGL
    initialized_ = true;
    return true;
}

void OpenGLWrapper::cleanup() {
    if (initialized_) {
        std::cout << "Cleaning up OpenGL wrapper" << std::endl;
        initialized_ = false;
    }
}

uint32_t OpenGLWrapper::create_texture_2d(int width, int height) {
    // Simplified texture creation
    return 1; // Return fake texture ID
}

void OpenGLWrapper::update_texture_data(uint32_t texture_id, const std::vector<uint8_t>& data) {
    std::cout << "Updating texture data for texture " << texture_id << std::endl;
}

void OpenGLWrapper::render_texture(uint32_t texture_id, int x, int y, int width, int height) {
    std::cout << "Rendering texture " << texture_id << " at (" << x << "," << y << ") " 
              << width << "x" << height << std::endl;
}

void OpenGLWrapper::set_viewport(int x, int y, int width, int height) {
    std::cout << "Setting viewport to " << width << "x" << height << std::endl;
}

// FrameBuffer implementation
FrameBuffer::FrameBuffer(size_t capacity) : capacity_(capacity) {}

FrameBuffer::~FrameBuffer() = default;

void FrameBuffer::add_frame(const Tensor& frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frames_.size() >= capacity_) {
        frames_.erase(frames_.begin()); // Remove oldest frame
    }
    
    frames_.push_back(frame);
}

Tensor FrameBuffer::get_frame(size_t index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (index < frames_.size()) {
        return frames_[index];
    }
    
    return Tensor::empty();
}

size_t FrameBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_.size();
}

void FrameBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    frames_.clear();
}

void FrameBuffer::reserve(size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
}

void FrameBuffer::thread_safe_add_frame(const Tensor& frame) {
    add_frame(frame);
}

Tensor FrameBuffer::thread_safe_get_frame(size_t index) const {
    return get_frame(index);
}

} // namespace render
} // namespace jin