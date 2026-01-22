#pragma once

#include "renderer.h"
#include "video_export.h"
#include "quality.h"
#include "preview.h"
#include "../audio/processor.h"
#include "../audio/sync.h"
#include "../temporal/diffusion_loop.h"
#include "../core/tensor.h"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <filesystem>

namespace jin {
namespace render {

// Progress callback type
using ProgressCallback = std::function<void(int current, int total)>;
using ProgressCallbackWithStage = std::function<void(const std::string& stage, int current, int total)>;

// Pipeline stages enumeration
enum class PipelineStage {
    INITIALIZATION,
    LOADING_FRAMES,
    LOADING_AUDIO,
    AUDIO_VISUAL_SYNC,
    FRAME_RENDERING,
    QUALITY_ASSESSMENT,
    VIDEO_ENCODING,
    METADATA_EMBEDDING,
    FINAL_VALIDATION,
    COMPLETION
};

// Pipeline configuration
struct PipelineConfig {
    std::vector<Tensor> animation_frames;    // Input animation frames
    std::string audio_path;                  // Optional audio file path
    std::string output_video_path;           // Output video file path
    RenderConfig render_config;              // Frame rendering configuration
    ExportConfig export_config;             // Video export configuration
    
    enum class QualityTarget {
        FAST, BALANCED, HIGH_QUALITY
    } quality_target;
    
    bool enable_audio_sync;                  // Enable audio-visual synchronization
    bool enable_preview;                     // Enable real-time preview
    bool enable_quality_assessment;         // Enable quality metrics computation
    bool save_checkpoints;                  // Save intermediate results
    std::string checkpoint_dir;              // Directory for checkpoints
    
    ProgressCallbackWithStage progress_callback; // Progress callback
    
    PipelineConfig()
        : quality_target(QualityTarget::BALANCED),
          enable_audio_sync(true), enable_preview(false),
          enable_quality_assessment(true),
          save_checkpoints(false), checkpoint_dir(""),
          progress_callback(nullptr) {}
};

// Checkpoint data structure
struct CheckpointData {
    std::string stage_name;                 // Current pipeline stage
    int progress;                          // Progress within stage [0-100]
    std::vector<Tensor> processed_frames;   // Frames processed so far
    Tensor processed_audio;                // Processed audio data
    std::map<std::string, std::string> metadata; // Additional metadata
    std::chrono::system_clock::time_point timestamp; // When checkpoint was created
};

// Quality report from pipeline
struct PipelineQualityReport {
    QualityReport render_quality;           // Overall render quality
    QualityReport temporal_quality;         // Temporal quality metrics
    float audio_sync_score;                 // Audio-visual sync score
    std::vector<float> frame_quality_scores; // Per-frame quality scores
    std::map<std::string, float> performance_metrics; // Performance statistics
    bool overall_success;                   // Whether pipeline completed successfully
    std::vector<std::string> warnings;      // Any warnings during processing
};

// Time estimation result
struct TimeEstimation {
    float estimated_total_seconds;           // Total estimated time
    std::map<PipelineStage, float> stage_estimates; // Time per stage
    float confidence;                       // Estimation confidence [0, 1]
};

// Main render pipeline class - high-level API for end-to-end processing
class RenderPipeline {
public:
    RenderPipeline();
    explicit RenderPipeline(const PipelineConfig& config);
    
    // Main pipeline execution
    bool render_full_pipeline();
    bool render_with_preview();
    bool resume_interrupted_render(const std::string& checkpoint_path);
    
    // Quality assessment
    PipelineQualityReport get_quality_report() const { return quality_report_; }
    TimeEstimation estimate_render_time() const;
    
    // Configuration management
    void set_config(const PipelineConfig& config) { config_ = config; }
    const PipelineConfig& get_config() const { return config_; }
    void update_render_config(const RenderConfig& render_config);
    void update_export_config(const ExportConfig& export_config);
    
    // Progress and cancellation
    void set_progress_callback(ProgressCallbackWithStage callback);
    void cancel_render();
    bool is_cancelled() const { return cancelled_; }
    bool is_running() const { return running_; }
    
    // Checkpoint management
    bool save_checkpoint(const std::string& path);
    bool load_checkpoint(const std::string& path);
    void enable_checkpoints(const std::string& directory);
    void disable_checkpoints();
    
    // Preview integration
    void enable_preview_rendering();
    void disable_preview_rendering();
    PreviewRenderer* get_preview_renderer() { return &preview_renderer_; }
    
    // Export options
    bool export_image_sequence(const std::string& output_dir, const std::string& format = "png");
    bool export_hdri_sequence(const std::string& output_dir, const std::string& format = "exr");
    
    // Utility methods
    std::vector<std::string> get_supported_video_formats() const;
    std::vector<std::string> get_supported_audio_formats() const;
    bool validate_configuration() const;
    void print_configuration() const;

private:
    PipelineConfig config_;
    bool running_;
    bool cancelled_;
    
    // Component instances
    Renderer renderer_;
    VideoExporter video_exporter_;
    QualityMetrics quality_metrics_;
    PreviewRenderer preview_renderer_;
    audio::AudioProcessor audio_processor_;
    audio::AudioSync audio_sync_;
    
    // State tracking
    PipelineStage current_stage_;
    int stage_progress_;
    PipelineQualityReport quality_report_;
    std::chrono::system_clock::time_point start_time_;
    
    // Pipeline stage methods
    bool initialize_pipeline();
    bool load_animation_frames();
    bool load_and_process_audio();
    bool perform_audio_visual_sync();
    bool render_frames();
    bool assess_quality();
    bool encode_and_export_video();
    bool embed_metadata();
    bool validate_final_output();
    bool cleanup_pipeline();
    
    // Stage execution helpers
    bool execute_stage(PipelineStage stage, std::function<bool()> stage_func);
    void update_progress(const std::string& stage_name, int current, int total);
    void handle_stage_error(const std::string& stage, const std::string& error);
    
    // Audio processing pipeline
    Tensor load_audio_file(const std::string& filepath);
    Tensor preprocess_audio(const Tensor& audio);
    std::vector<float> extract_sync_features(const Tensor& audio);
    
    // Frame processing pipeline
    std::vector<Tensor> preprocess_frames(const std::vector<Tensor>& frames);
    std::vector<Tensor> apply_quality_settings(const std::vector<Tensor>& frames);
    
    // Quality assessment pipeline
    QualityReport assess_render_quality(const std::vector<Tensor>& frames);
    QualityReport assess_temporal_quality(const std::vector<Tensor>& frames);
    float assess_audio_sync_quality(const std::vector<Tensor>& frames, const Tensor& audio);
    
    // Export pipeline
    bool prepare_video_export();
    bool export_with_codec_optimization();
    bool add_post_processing_effects();
    
    // Checkpoint system
    bool save_stage_checkpoint(PipelineStage stage);
    bool load_stage_checkpoint(PipelineStage stage);
    std::string get_checkpoint_path(PipelineStage stage) const;
    void cleanup_old_checkpoints();
    
    // Preview integration
    bool setup_preview_integration();
    bool update_preview_during_render();
    void handle_preview_events();
    
    // Error handling and recovery
    bool handle_encoding_failure();
    bool handle_memory_limit();
    bool handle_corrupted_frames();
    
    // Performance optimization
    void optimize_for_target_quality();
    void enable_parallel_processing();
    void setup_memory_management();
    
    // Utility methods
    std::string get_current_stage_name() const;
    std::vector<PipelineStage> get_pipeline_stages() const;
    bool check_system_resources() const;
    void log_performance_metrics();
    
    // Constants
    static constexpr float DEFAULT_ESTIMATION_CONFIDENCE = 0.8f;
    static constexpr int MAX_RETRIES = 3;
    static constexpr size_t MEMORY_LIMIT_MB = 2048; // 2GB memory limit
};

// Configuration helper functions
namespace config {

// Create high-quality configuration
PipelineConfig create_high_quality_config(
    const std::vector<Tensor>& frames,
    const std::string& output_path,
    const std::string& audio_path = ""
);

// Create balanced configuration
PipelineConfig create_balanced_config(
    const std::vector<Tensor>& frames,
    const std::string& output_path,
    const std::string& audio_path = ""
);

// Create fast configuration
PipelineConfig create_fast_config(
    const std::vector<Tensor>& frames,
    const std::string& output_path,
    const std::string& audio_path = ""
);

// Create preview configuration
PipelineConfig create_preview_config(
    const std::vector<Tensor>& frames,
    const std::string& audio_path = ""
);

// Validation helpers
bool validate_frame_sequence(const std::vector<Tensor>& frames);
bool validate_audio_file(const std::string& filepath);
bool validate_output_path(const std::string& output_path);
bool validate_render_config(const RenderConfig& config);
bool validate_export_config(const ExportConfig& config);

} // namespace config

} // namespace render
} // namespace jin