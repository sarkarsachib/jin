#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include "scene_graph.h"
#include "layer_stack.h"
#include "sprite.h"
#include "particle_system.h"
#include "text_renderer.h"
#include "background.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace jin {
namespace scene {

/**
 * RenderPass - Individual rendering pass for compositing
 */
struct RenderPass {
    std::string name;
    Tensor input_texture;       // Input texture for this pass
    Tensor output_texture;      // Output texture after processing
    enum class PassType {
        BACKGROUND,
        SPRITES,
        PARTICLES,
        TEXT,
        EFFECTS,
        FINAL
    } pass_type;
    
    // Pass properties
    bool enabled = true;
    float opacity = 1.0f;
    BlendMode blend_mode = BlendMode::NORMAL;
    
    // Shader-like parameters (simplified)
    std::map<std::string, float> float_params;
    std::map<std::string, Tensor> tensor_params;
};

/**
 * Compositor - Final frame composition engine
 */
class Compositor {
public:
    // Render configuration
    int output_width = 1280;
    int output_height = 720;
    float frame_rate = 30.0f;
    
    // Rendering settings
    bool enable_post_processing = true;
    bool enable_color_grading = true;
    bool enable_bloom = true;
    bool enable_anti_aliasing = false;
    
    // Quality settings
    enum class QualityLevel {
        LOW,
        MEDIUM,
        HIGH,
        ULTRA
    };
    
    QualityLevel quality = QualityLevel::HIGH;
    
    // Compositing pipeline
    std::vector<RenderPass> render_passes;
    
    // Post-processing effects
    struct PostProcessingSettings {
        // Bloom effect
        bool bloom_enabled = true;
        float bloom_intensity = 0.5f;
        float bloom_threshold = 0.8f;
        float bloom_radius = 2.0f;
        
        // Color grading
        bool color_grading_enabled = true;
        Tensor exposure = {1.0f, 1.0f, 1.0f};    // RGB exposure
        Tensor contrast = {1.0f, 1.0f, 1.0f};     // RGB contrast
        Tensor saturation = {1.0f, 1.0f, 1.0f};   // RGB saturation
        Tensor temperature = {0.0f, 0.0f, 0.0f};   // Color temperature adjustment
        Tensor tint = {0.0f, 0.0f, 0.0f};         // Color tint adjustment
        
        // Tone mapping
        enum class ToneMapping {
            NONE,
            REINHARD,
            ACES,
            FILMIC
        } tone_mapping = ToneMapping::FILMIC;
        
        // Gamma correction
        float gamma = 2.2f;
        
        // Vignette
        bool vignette_enabled = true;
        float vignette_intensity = 0.3f;
        float vignette_radius = 0.8f;
        
        // Chromatic aberration
        bool chromatic_aberration_enabled = false;
        float aberration_intensity = 1.0f;
        Tensor aberration_direction = {1.0f, 0.0f};
    };
    
    PostProcessingSettings post_processing;
    
    // Performance optimization
    bool use_multithreading = false;
    bool cache_rendered_frames = false;
    int max_cache_size = 10;
    
    Compositor();
    ~Compositor() = default;
    
    // Configuration
    void set_output_size(int width, int height);
    void set_frame_rate(float fps);
    void set_quality(QualityLevel quality_level);
    
    // Scene composition
    Tensor composite(
        SceneGraph& scene,
        LayerStack& layer_stack,
        Background& background,
        Camera2D& camera,
        int width,
        int height
    );
    
    // Individual render passes
    Tensor render_background_pass(Background& background, const Camera2D& camera, int width, int height);
    Tensor render_sprites_pass(LayerStack& layer_stack, const Camera2D& camera, int width, int height);
    Tensor render_particles_pass(std::vector<std::shared_ptr<ParticleSystem>>& particle_systems, const Camera2D& camera, int width, int height);
    Tensor render_text_pass(std::vector<std::shared_ptr<TextRenderer>>& text_renderers, const Camera2D& camera, int width, int height);
    
    // Post-processing pipeline
    Tensor apply_post_processing(const Tensor& input_frame);
    Tensor apply_bloom(const Tensor& input_frame);
    Tensor apply_color_grading(const Tensor& input_frame);
    Tensor apply_tone_mapping(const Tensor& input_frame);
    Tensor apply_gamma_correction(const Tensor& input_frame, float gamma);
    Tensor apply_vignette(const Tensor& input_frame);
    Tensor apply_chromatic_aberration(const Tensor& input_frame);
    
    // Render pass management
    void add_render_pass(const RenderPass& pass);
    void remove_render_pass(const std::string& pass_name);
    void clear_render_passes();
    void enable_pass(const std::string& pass_name, bool enabled);
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state);
    temporal::FrameState get_frame_state() const;
    
    // Performance monitoring
    struct PerformanceMetrics {
        float total_render_time = 0.0f;
        float background_pass_time = 0.0f;
        float sprites_pass_time = 0.0f;
        float particles_pass_time = 0.0f;
        float text_pass_time = 0.0f;
        float post_processing_time = 0.0f;
        int frames_rendered = 0;
    };
    
    PerformanceMetrics get_performance_metrics() const;
    void reset_performance_metrics();
    
    // Utility methods
    void set_post_processing_settings(const PostProcessingSettings& settings);
    PostProcessingSettings get_post_processing_settings() const;
    
    // Debug and profiling
    void enable_profiling(bool enabled);
    void set_debug_output(bool enabled);
    
private:
    temporal::FrameState frame_state_;
    PerformanceMetrics performance_metrics_;
    bool profiling_enabled_ = false;
    bool debug_output_ = false;
    
    // Cached textures for optimization
    Tensor cached_background_;
    std::map<std::string, Tensor> cached_passes_;
    
    // Helper methods
    Tensor blend_textures(const Tensor& background, const Tensor& foreground, BlendMode blend_mode, float opacity) const;
    Tensor apply_blend_mode(const Tensor& source, const Tensor& destination, BlendMode mode) const;
    
    // Performance tracking
    void start_timing(const std::string& pass_name);
    void end_timing(const std::string& pass_name);
    
    // Quality-dependent settings
    void apply_quality_settings();
    Tensor get_bloom_kernel(float radius, float intensity) const;
    Tensor get_color_lut(int size) const;
};

/**
 * RealTimeCompositor - Compositor optimized for real-time rendering
 */
class RealTimeCompositor : public Compositor {
public:
    // Real-time specific settings
    bool use_frame_prediction = true;     // Predict next frame for smoother rendering
    bool use_adaptive_quality = true;      // Adapt quality based on frame time
    float target_frame_time = 16.67f;      // 60 FPS target (16.67ms)
    
    // Adaptive quality settings
    struct AdaptiveQualitySettings {
        float min_frame_time = 10.0f;      // Below this, increase quality
        float max_frame_time = 25.0f;      // Above this, decrease quality
        int quality_adjustment_interval = 30; // Adjust quality every N frames
        int current_adjustment_counter = 0;
    };
    
    AdaptiveQualitySettings adaptive_quality;
    
    RealTimeCompositor();
    ~RealTimeCompositor() = default;
    
    // Override composite for real-time optimization
    Tensor composite(
        SceneGraph& scene,
        LayerStack& layer_stack,
        Background& background,
        Camera2D& camera,
        int width,
        int height
    ) override;
    
    // Real-time specific methods
    void set_target_frame_time(float frame_time_ms);
    void enable_adaptive_quality(bool enabled);
    void set_adaptive_quality_settings(const AdaptiveQualitySettings& settings);
    
    // Frame prediction
    void enable_frame_prediction(bool enabled);
    Tensor predict_next_frame(const Tensor& current_frame, float delta_time);
    
    // Performance optimization
    void set_max_render_time(float max_time_ms);
    void set_quality_levels(QualityLevel min_quality, QualityLevel max_quality);
    
private:
    void adjust_quality_based_on_performance();
    void update_frame_prediction(float delta_time);
    
    Tensor predicted_frame_;
    bool use_prediction_ = true;
    QualityLevel min_quality_ = QualityLevel::MEDIUM;
    QualityLevel max_quality_ = QualityLevel::ULTRA;
};

/**
 * OfflineCompositor - Compositor optimized for offline rendering (higher quality)
 */
class OfflineCompositor : public Compositor {
public:
    // Offline rendering settings
    bool use_ray_tracing = false;         // Enable ray tracing effects
    bool use_path_tracing = false;       // Enable path tracing
    bool use_advanced_sampling = true;    // Use advanced sampling techniques
    int samples_per_pixel = 16;           // Number of samples per pixel
    
    // Advanced features
    bool use_global_illumination = true;
    bool use_ambient_occlusion = true;
    bool use_reflections = true;
    bool use_refractions = true;
    
    // Quality settings
    int max_bounces = 4;                  // Maximum light bounces
    float roughness_threshold = 0.1f;    // Roughness threshold for effects
    
    OfflineCompositor();
    ~OfflineCompositor() = default;
    
    // Override composite for offline quality
    Tensor composite(
        SceneGraph& scene,
        LayerStack& layer_stack,
        Background& background,
        Camera2D& camera,
        int width,
        int height
    ) override;
    
    // Advanced rendering methods
    Tensor render_with_global_illumination(const Tensor& scene_data);
    Tensor render_with_ambient_occlusion(const Tensor& scene_data);
    Tensor render_with_reflections(const Tensor& scene_data);
    Tensor render_with_refractions(const Tensor& scene_data);
    
    // Batch rendering for animations
    std::vector<Tensor> render_animation_sequence(
        SceneGraph& scene,
        LayerStack& layer_stack,
        Background& background,
        Camera2D& camera,
        int width,
        int height,
        int start_frame,
        int end_frame
    );
    
    // Quality settings
    void set_samples_per_pixel(int samples);
    void set_max_light_bounces(int bounces);
    void enable_ray_tracing_effects(bool enabled);
    
private:
    void apply_advanced_sampling();
    Tensor denoise_rendered_frame(const Tensor& noisy_frame);
};

} // namespace scene
} // namespace jin