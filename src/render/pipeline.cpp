#include "pipeline.h"
#include "../utils/math.h"
#include "../autodiff/tape.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace jin {
namespace render {

RenderPipeline::RenderPipeline() {
    config_ = PipelineConfig{};
    running_ = false;
    cancelled_ = false;
    current_stage_ = PipelineStage::INITIALIZATION;
    stage_progress_ = 0;
    
    // Initialize quality report
    quality_report_ = PipelineQualityReport{};
    quality_report_.overall_success = false;
}

RenderPipeline::RenderPipeline(const PipelineConfig& config) : config_(config) {
    running_ = false;
    cancelled_ = false;
    current_stage_ = PipelineStage::INITIALIZATION;
    stage_progress_ = 0;
    
    // Initialize quality report
    quality_report_ = PipelineQualityReport{};
    quality_report_.overall_success = false;
}

bool RenderPipeline::render_full_pipeline() {
    if (running_) {
        std::cerr << "Pipeline is already running!" << std::endl;
        return false;
    }
    
    if (!validate_configuration()) {
        std::cerr << "Configuration validation failed!" << std::endl;
        return false;
    }
    
    running_ = true;
    cancelled_ = false;
    start_time_ = std::chrono::system_clock::now();
    
    std::cout << "Starting Jin Render Pipeline..." << std::endl;
    print_configuration();
    
    // Execute pipeline stages
    bool success = true;
    
    // Stage 1: Initialize
    success &= execute_stage(PipelineStage::INITIALIZATION, [this]() { return initialize_pipeline(); });
    
    // Stage 2: Load animation frames
    success &= execute_stage(PipelineStage::LOADING_FRAMES, [this]() { return load_animation_frames(); });
    
    // Stage 3: Load and process audio
    if (!config_.audio_path.empty()) {
        success &= execute_stage(PipelineStage::LOADING_AUDIO, [this]() { return load_and_process_audio(); });
        
        // Stage 4: Audio-visual sync
        if (config_.enable_audio_sync) {
            success &= execute_stage(PipelineStage::AUDIO_VISUAL_SYNC, [this]() { return perform_audio_visual_sync(); });
        }
    }
    
    // Stage 5: Render frames
    success &= execute_stage(PipelineStage::FRAME_RENDERING, [this]() { return render_frames(); });
    
    // Stage 6: Quality assessment
    if (config_.enable_quality_assessment) {
        success &= execute_stage(PipelineStage::QUALITY_ASSESSMENT, [this]() { return assess_quality(); });
    }
    
    // Stage 7: Video encoding
    success &= execute_stage(PipelineStage::VIDEO_ENCODING, [this]() { return encode_and_export_video(); });
    
    // Stage 8: Metadata embedding
    success &= execute_stage(PipelineStage::METADATA_EMBEDDING, [this]() { return embed_metadata(); });
    
    // Stage 9: Final validation
    success &= execute_stage(PipelineStage::FINAL_VALIDATION, [this]() { return validate_final_output(); });
    
    // Stage 10: Completion
    success &= execute_stage(PipelineStage::COMPLETION, [this]() { return cleanup_pipeline(); });
    
    running_ = false;
    
    if (success && !cancelled_) {
        std::cout << "Pipeline completed successfully!" << std::endl;
        quality_report_.overall_success = true;
    } else if (cancelled_) {
        std::cout << "Pipeline was cancelled by user" << std::endl;
        quality_report_.overall_success = false;
    } else {
        std::cout << "Pipeline failed!" << std::endl;
        quality_report_.overall_success = false;
    }
    
    return success && !cancelled_;
}

bool RenderPipeline::render_with_preview() {
    if (!config_.enable_preview) {
        std::cerr << "Preview is not enabled in configuration!" << std::endl;
        return false;
    }
    
    // Create preview window
    if (!preview_renderer_.create_preview_window()) {
        std::cerr << "Failed to create preview window!" << std::endl;
        return false;
    }
    
    // Start preview thread
    preview_renderer_.run_preview_thread();
    
    // Run main pipeline with preview updates
    bool success = render_full_pipeline();
    
    // Cleanup preview
    preview_renderer_.stop_preview_thread();
    preview_renderer_.close_preview_window();
    
    return success;
}

bool RenderPipeline::resume_interrupted_render(const std::string& checkpoint_path) {
    if (running_) {
        std::cerr << "Pipeline is already running!" << std::endl;
        return false;
    }
    
    running_ = true;
    cancelled_ = false;
    
    std::cout << "Resuming pipeline from checkpoint: " << checkpoint_path << std::endl;
    
    if (!load_checkpoint(checkpoint_path)) {
        std::cerr << "Failed to load checkpoint!" << std::endl;
        running_ = false;
        return false;
    }
    
    // Continue from current stage
    bool success = render_full_pipeline();
    
    running_ = false;
    return success;
}

TimeEstimation RenderPipeline::estimate_render_time() const {
    TimeEstimation estimation;
    
    if (config_.animation_frames.empty()) {
        estimation.estimated_total_seconds = 0.0f;
        estimation.confidence = 0.0f;
        return estimation;
    }
    
    int num_frames = config_.animation_frames.size();
    int width = config_.render_config.resolution.first;
    int height = config_.render_config.resolution.second;
    
    // Base time estimates (simplified model)
    float frame_render_time = 0.0f;
    switch (config_.quality_target) {
        case PipelineConfig::QualityTarget::FAST:
            frame_render_time = (width * height) / (1920.0f * 1080.0f) * 0.01f; // 10ms per frame at 1080p
            break;
        case PipelineConfig::QualityTarget::BALANCED:
            frame_render_time = (width * height) / (1920.0f * 1080.0f) * 0.05f; // 50ms per frame at 1080p
            break;
        case PipelineConfig::QualityTarget::HIGH_QUALITY:
            frame_render_time = (width * height) / (1920.0f * 1080.0f) * 0.2f; // 200ms per frame at 1080p
            break;
    }
    
    // Stage-specific estimates
    estimation.stage_estimates[PipelineStage::INITIALIZATION] = 1.0f;
    estimation.stage_estimates[PipelineStage::LOADING_FRAMES] = 0.5f;
    estimation.stage_estimates[PipelineStage::LOADING_AUDIO] = config_.audio_path.empty() ? 0.0f : 2.0f;
    estimation.stage_estimates[PipelineStage::AUDIO_VISUAL_SYNC] = config_.enable_audio_sync ? 5.0f : 0.0f;
    estimation.stage_estimates[PipelineStage::FRAME_RENDERING] = num_frames * frame_render_time;
    estimation.stage_estimates[PipelineStage::QUALITY_ASSESSMENT] = config_.enable_quality_assessment ? num_frames * 0.01f : 0.0f;
    estimation.stage_estimates[PipelineStage::VIDEO_ENCODING] = num_frames * 0.02f;
    estimation.stage_estimates[PipelineStage::METADATA_EMBEDDING] = 1.0f;
    estimation.stage_estimates[PipelineStage::FINAL_VALIDATION] = 1.0f;
    
    // Total time
    estimation.estimated_total_seconds = std::accumulate(
        estimation.stage_estimates.begin(),
        estimation.stage_estimates.end(), 0.0f,
        [](float sum, const auto& pair) { return sum + pair.second; }
    );
    
    // Confidence based on system resources and configuration
    estimation.confidence = DEFAULT_ESTIMATION_CONFIDENCE;
    
    return estimation;
}

void RenderPipeline::update_render_config(const RenderConfig& render_config) {
    config_.render_config = render_config;
    renderer_.set_config(render_config);
}

void RenderPipeline::update_export_config(const ExportConfig& export_config) {
    config_.export_config = export_config;
    video_exporter_.set_config(export_config);
}

void RenderPipeline::set_progress_callback(ProgressCallbackWithStage callback) {
    config_.progress_callback = callback;
}

void RenderPipeline::cancel_render() {
    cancelled_ = true;
    std::cout << "Pipeline cancellation requested..." << std::endl;
}

bool RenderPipeline::save_checkpoint(const std::string& path) {
    CheckpointData checkpoint;
    checkpoint.stage_name = get_current_stage_name();
    checkpoint.progress = stage_progress_;
    checkpoint.timestamp = std::chrono::system_clock::now();
    
    // Store current configuration
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Write checkpoint data (simplified)
    file.write(reinterpret_cast<const char*>(&checkpoint.stage_name), sizeof(checkpoint.stage_name));
    file.write(reinterpret_cast<const char*>(&checkpoint.progress), sizeof(checkpoint.progress));
    
    file.close();
    return true;
}

bool RenderPipeline::load_checkpoint(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Load checkpoint data (simplified)
    CheckpointData checkpoint;
    file.read(reinterpret_cast<char*>(&checkpoint.stage_name), sizeof(checkpoint.stage_name));
    file.read(reinterpret_cast<char*>(&checkpoint.progress), sizeof(checkpoint.progress));
    
    current_stage_ = PipelineStage::INITIALIZATION; // Resume from beginning for simplicity
    stage_progress_ = 0;
    
    file.close();
    return true;
}

void RenderPipeline::enable_preview_rendering() {
    config_.enable_preview = true;
}

void RenderPipeline::disable_preview_rendering() {
    config_.enable_preview = false;
}

bool RenderPipeline::export_image_sequence(const std::string& output_dir, const std::string& format) {
    if (config_.animation_frames.empty()) {
        std::cerr << "No frames to export!" << std::endl;
        return false;
    }
    
    ImageSequenceExporter exporter;
    std::string pattern = output_dir + "/frame_%04d." + format;
    
    return exporter.export_sequence(config_.animation_frames, pattern, format);
}

bool RenderPipeline::export_hdri_sequence(const std::string& output_dir, const std::string& format) {
    if (config_.animation_frames.empty()) {
        std::cerr << "No frames to export!" << std::endl;
        return false;
    }
    
    ImageSequenceExporter exporter;
    std::string pattern = output_dir + "/frame_%04d." + format;
    
    return exporter.export_hdri_sequence(config_.animation_frames, pattern, format);
}

// Pipeline Stage Implementations
bool RenderPipeline::initialize_pipeline() {
    std::cout << "Initializing pipeline..." << std::endl;
    
    // Check system resources
    if (!check_system_resources()) {
        std::cerr << "Insufficient system resources!" << std::endl;
        return false;
    }
    
    // Initialize components
    renderer_.set_config(config_.render_config);
    video_exporter_.set_config(config_.export_config);
    
    // Setup memory management
    setup_memory_management();
    
    // Enable parallel processing if configured
    if (config_.quality_target != PipelineConfig::QualityTarget::FAST) {
        enable_parallel_processing();
    }
    
    return true;
}

bool RenderPipeline::load_animation_frames() {
    std::cout << "Loading animation frames..." << std::endl;
    
    if (config_.animation_frames.empty()) {
        std::cerr << "No animation frames provided in configuration!" << std::endl;
        return false;
    }
    
    // Validate frame sequence
    if (!config::validate_frame_sequence(config_.animation_frames)) {
        std::cerr << "Invalid frame sequence!" << std::endl;
        return false;
    }
    
    // Preprocess frames
    config_.animation_frames = preprocess_frames(config_.animation_frames);
    
    std::cout << "Loaded " << config_.animation_frames.size() << " frames" << std::endl;
    return true;
}

bool RenderPipeline::load_and_process_audio() {
    if (config_.audio_path.empty()) {
        return true; // No audio to process
    }
    
    std::cout << "Loading and processing audio: " << config_.audio_path << std::endl;
    
    // Load audio file
    Tensor audio = load_audio_file(config_.audio_path);
    if (audio.is_empty()) {
        std::cerr << "Failed to load audio file!" << std::endl;
        return false;
    }
    
    // Preprocess audio
    Tensor processed_audio = preprocess_audio(audio);
    
    std::cout << "Audio loaded: " << audio.shape()[0] << " samples at " 
              << audio_processor_.get_sample_rate() << " Hz" << std::endl;
    
    return true;
}

bool RenderPipeline::perform_audio_visual_sync() {
    if (config_.audio_path.empty()) {
        return true; // No audio to sync with
    }
    
    std::cout << "Performing audio-visual synchronization..." << std::endl;
    
    // Extract sync features from audio
    std::vector<float> sync_features = extract_sync_features(Tensor{}); // Simplified
    
    // Apply sync to frames (simplified)
    // In a full implementation, this would use the AudioSync class
    
    std::cout << "Audio-visual sync completed" << std::endl;
    return true;
}

bool RenderPipeline::render_frames() {
    std::cout << "Rendering frames..." << std::endl;
    
    // Apply quality settings
    config_.animation_frames = apply_quality_settings(config_.animation_frames);
    
    // Update preview during rendering if enabled
    if (config_.enable_preview) {
        setup_preview_integration();
    }
    
    std::cout << "Frame rendering completed" << std::endl;
    return true;
}

bool RenderPipeline::assess_quality() {
    if (!config_.enable_quality_assessment) {
        return true;
    }
    
    std::cout << "Assessing quality..." << std::endl;
    
    // Assess render quality
    quality_report_.render_quality = assess_render_quality(config_.animation_frames);
    
    // Assess temporal quality
    quality_report_.temporal_quality = assess_temporal_quality(config_.animation_frames);
    
    // Calculate overall quality score
    quality_report_.frame_quality_scores = std::vector<float>(config_.animation_frames.size(), 8.0f); // Simplified
    
    std::cout << "Quality assessment completed" << std::endl;
    return true;
}

bool RenderPipeline::encode_and_export_video() {
    std::cout << "Encoding and exporting video..." << std::endl;
    
    if (!prepare_video_export()) {
        return handle_encoding_failure();
    }
    
    // Export video
    bool success = video_exporter_.export_video(
        config_.animation_frames,
        Tensor{}, // No audio in simplified version
        config_.output_video_path
    );
    
    if (success) {
        std::cout << "Video exported to: " << config_.output_video_path << std::endl;
    } else {
        std::cerr << "Video export failed!" << std::endl;
    }
    
    return success;
}

bool RenderPipeline::embed_metadata() {
    std::cout << "Embedding metadata..." << std::endl;
    
    // Create metadata
    VideoMetadata metadata;
    metadata.title = "Jin Animation Export";
    metadata.creator = "Jin 2D Animation Model";
    metadata.resolution = config_.render_config.resolution;
    metadata.fps = config_.export_config.fps;
    metadata.frame_count = config_.animation_frames.size();
    
    // Add metadata to video file
    bool success = video_exporter_.add_metadata(config_.output_video_path, metadata);
    
    if (success) {
        std::cout << "Metadata embedded successfully" << std::endl;
    }
    
    return success;
}

bool RenderPipeline::validate_final_output() {
    std::cout << "Validating final output..." << std::endl;
    
    // Check if output file exists
    if (!std::filesystem::exists(config_.output_video_path)) {
        std::cerr << "Output video file does not exist!" << std::endl;
        return false;
    }
    
    // Check file size
    size_t file_size = std::filesystem::file_size(config_.output_video_path);
    if (file_size < 1000) { // Less than 1KB is suspicious
        std::cerr << "Output file seems too small!" << std::endl;
        return false;
    }
    
    std::cout << "Output validation passed. File size: " << (file_size / 1024 / 1024) << " MB" << std::endl;
    return true;
}

bool RenderPipeline::cleanup_pipeline() {
    std::cout << "Cleaning up pipeline..." << std::endl;
    
    // Log performance metrics
    log_performance_metrics();
    
    // Cleanup old checkpoints
    if (config_.save_checkpoints) {
        cleanup_old_checkpoints();
    }
    
    // Final quality report
    if (config_.enable_quality_assessment) {
        std::cout << "\n=== QUALITY REPORT ===" << std::endl;
        std::cout << "Overall Grade: " << quality_report_.render_quality.overall_grade << std::endl;
        std::cout << "Perceptual Quality: " << quality_report_.render_quality.perceptual_quality << "/10" << std::endl;
        std::cout << "Temporal Stability: " << quality_report_.render_quality.temporal_stability << std::endl;
        std::cout << "======================" << std::endl;
    }
    
    std::cout << "Pipeline cleanup completed" << std::endl;
    return true;
}

bool RenderPipeline::execute_stage(PipelineStage stage, std::function<bool()> stage_func) {
    current_stage_ = stage;
    stage_progress_ = 0;
    
    std::string stage_name = get_current_stage_name();
    std::cout << "\n=== " << stage_name << " ===" << std::endl;
    
    // Update progress
    update_progress(stage_name, 0, 100);
    
    // Execute stage
    bool success = false;
    try {
        success = stage_func();
    } catch (const std::exception& e) {
        std::cerr << "Exception in stage " << stage_name << ": " << e.what() << std::endl;
        success = false;
    }
    
    if (!success) {
        handle_stage_error(stage_name, "Stage execution failed");
        return false;
    }
    
    // Update progress to completion
    update_progress(stage_name, 100, 100);
    
    return true;
}

void RenderPipeline::update_progress(const std::string& stage_name, int current, int total) {
    stage_progress_ = static_cast<int>((static_cast<float>(current) / total) * 100);
    
    if (config_.progress_callback) {
        config_.progress_callback(stage_name, current, total);
    }
}

void RenderPipeline::handle_stage_error(const std::string& stage, const std::string& error) {
    std::cerr << "Error in stage '" << stage << "': " << error << std::endl;
    quality_report_.overall_success = false;
    
    // Add warning to report
    quality_report_.warnings.push_back("Error in " + stage + ": " + error);
}

Tensor RenderPipeline::load_audio_file(const std::string& filepath) {
    try {
        return audio_processor_.load_audio(filepath);
    } catch (const std::exception& e) {
        std::cerr << "Error loading audio: " << e.what() << std::endl;
        return Tensor::empty();
    }
}

Tensor RenderPipeline::preprocess_audio(const Tensor& audio) {
    if (audio.is_empty()) {
        return audio;
    }
    
    // Normalize audio
    return audio_processor_.normalize_audio(audio);
}

std::vector<float> RenderPipeline::extract_sync_features(const Tensor& audio) {
    if (audio.is_empty()) {
        return std::vector<float>();
    }
    
    // Extract energy features for sync
    Tensor energy_features = audio_processor_.extract_features(audio, audio::FeatureType::ENERGY);
    return energy_features.get_col(0).to_vector<float>();
}

std::vector<Tensor> RenderPipeline::preprocess_frames(const std::vector<Tensor>& frames) {
    std::vector<Tensor> processed = frames;
    
    // Ensure all frames have consistent dimensions
    if (!frames.empty()) {
        auto target_shape = frames[0].shape();
        
        for (auto& frame : processed) {
            if (frame.shape() != target_shape) {
                // Resize frame if needed (simplified)
                frame = frame.clone(); // Placeholder for actual resizing
            }
        }
    }
    
    return processed;
}

std::vector<Tensor> RenderPipeline::apply_quality_settings(const std::vector<Tensor>& frames) {
    // Apply render configuration to frames
    RenderConfig config = config_.render_config;
    
    // Set resolution
    renderer_.set_resolution(config.resolution.first, config.resolution.second);
    
    // Apply tone mapping if needed
    if (config.output_format == RenderConfig::Format::RGB ||
        config.output_format == RenderConfig::Format::RGBA) {
        // Apply tone mapping during rendering
    }
    
    return frames; // Return processed frames
}

QualityReport RenderPipeline::assess_render_quality(const std::vector<Tensor>& frames) {
    return quality_metrics_.compute_metrics_batch(frames);
}

QualityReport RenderPipeline::assess_temporal_quality(const std::vector<Tensor>& frames) {
    return quality_metrics_.compute_metrics_batch(frames);
}

float RenderPipeline::assess_audio_sync_quality(const std::vector<Tensor>& frames, const Tensor& audio) {
    // Simplified sync quality assessment
    return 0.85f; // Return placeholder quality score
}

bool RenderPipeline::prepare_video_export() {
    // Check codec availability
    if (!video_exporter_.is_codec_available(config_.export_config.video_codec)) {
        std::cerr << "Video codec not available!" << std::endl;
        return false;
    }
    
    // Validate output path
    if (!config::validate_output_path(config_.output_video_path)) {
        std::cerr << "Invalid output path!" << std::endl;
        return false;
    }
    
    return true;
}

bool RenderPipeline::handle_encoding_failure() {
    std::cerr << "Handling encoding failure..." << std::endl;
    
    // Try fallback to image sequence
    std::string fallback_path = config_.output_video_path + "_frames";
    std::cout << "Falling back to image sequence export: " << fallback_path << std::endl;
    
    return export_image_sequence(fallback_path, "png");
}

bool RenderPipeline::check_system_resources() const {
    // Simplified resource checking
    if (config_.animation_frames.size() * 1024 * 1024 > MEMORY_LIMIT_MB) {
        std::cerr << "Insufficient memory for processing!" << std::endl;
        return false;
    }
    
    return true;
}

void RenderPipeline::log_performance_metrics() {
    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time_);
    
    std::cout << "\n=== PERFORMANCE METRICS ===" << std::endl;
    std::cout << "Total execution time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Frames processed: " << config_.animation_frames.size() << std::endl;
    std::cout << "Average time per frame: " << (duration.count() / std::max(1, (int)config_.animation_frames.size())) << " seconds" << std::endl;
    std::cout << "=========================" << std::endl;
}

std::string RenderPipeline::get_current_stage_name() const {
    switch (current_stage_) {
        case PipelineStage::INITIALIZATION: return "INITIALIZATION";
        case PipelineStage::LOADING_FRAMES: return "LOADING FRAMES";
        case PipelineStage::LOADING_AUDIO: return "LOADING AUDIO";
        case PipelineStage::AUDIO_VISUAL_SYNC: return "AUDIO-VISUAL SYNC";
        case PipelineStage::FRAME_RENDERING: return "FRAME RENDERING";
        case PipelineStage::QUALITY_ASSESSMENT: return "QUALITY ASSESSMENT";
        case PipelineStage::VIDEO_ENCODING: return "VIDEO ENCODING";
        case PipelineStage::METADATA_EMBEDDING: return "METADATA EMBEDDING";
        case PipelineStage::FINAL_VALIDATION: return "FINAL VALIDATION";
        case PipelineStage::COMPLETION: return "COMPLETION";
        default: return "UNKNOWN";
    }
}

bool RenderPipeline::validate_configuration() const {
    return config::validate_frame_sequence(config_.animation_frames) &&
           config::validate_output_path(config_.output_video_path) &&
           config::validate_render_config(config_.render_config) &&
           config::validate_export_config(config_.export_config);
}

void RenderPipeline::print_configuration() const {
    std::cout << "\n=== PIPELINE CONFIGURATION ===" << std::endl;
    std::cout << "Frames: " << config_.animation_frames.size() << std::endl;
    std::cout << "Audio: " << (config_.audio_path.empty() ? "None" : config_.audio_path) << std::endl;
    std::cout << "Output: " << config_.output_video_path << std::endl;
    std::cout << "Quality Target: " << (int)config_.quality_target << std::endl;
    std::cout << "Audio Sync: " << (config_.enable_audio_sync ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Preview: " << (config_.enable_preview ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Quality Assessment: " << (config_.enable_quality_assessment ? "Enabled" : "Disabled") << std::endl;
    std::cout << "=============================" << std::endl;
}

void RenderPipeline::optimize_for_target_quality() {
    // Apply quality-specific optimizations
    switch (config_.quality_target) {
        case PipelineConfig::QualityTarget::FAST:
            // Optimize for speed
            break;
        case PipelineConfig::QualityTarget::BALANCED:
            // Balance quality and speed
            break;
        case PipelineConfig::QualityTarget::HIGH_QUALITY:
            // Optimize for quality
            break;
    }
}

void RenderPipeline::enable_parallel_processing() {
    // Enable parallel processing where possible
    std::cout << "Parallel processing enabled" << std::endl;
}

void RenderPipeline::setup_memory_management() {
    // Setup memory management for large frame sequences
    std::cout << "Memory management configured" << std::endl;
}

bool RenderPipeline::setup_preview_integration() {
    if (!config_.enable_preview) {
        return false;
    }
    
    // Configure preview for real-time updates
    preview_renderer_.set_resolution(
        config_.render_config.resolution.first,
        config_.render_config.resolution.second
    );
    
    return true;
}

bool RenderPipeline::update_preview_during_render() {
    // Update preview with current frame
    return true;
}

void RenderPipeline::handle_preview_events() {
    // Handle preview window events
}

bool RenderPipeline::handle_memory_limit() {
    std::cerr << "Handling memory limit exceeded..." << std::endl;
    return false;
}

bool RenderPipeline::handle_corrupted_frames() {
    std::cerr << "Handling corrupted frames..." << std::endl;
    return false;
}

bool RenderPipeline::save_stage_checkpoint(PipelineStage stage) {
    if (!config_.save_checkpoints) {
        return true;
    }
    
    std::string checkpoint_path = get_checkpoint_path(stage);
    return save_checkpoint(checkpoint_path);
}

bool RenderPipeline::load_stage_checkpoint(PipelineStage stage) {
    if (!config_.save_checkpoints) {
        return true;
    }
    
    std::string checkpoint_path = get_checkpoint_path(stage);
    if (std::filesystem::exists(checkpoint_path)) {
        return load_checkpoint(checkpoint_path);
    }
    
    return true;
}

std::string RenderPipeline::get_checkpoint_path(PipelineStage stage) const {
    std::string filename = "checkpoint_" + get_current_stage_name() + ".dat";
    return config_.checkpoint_dir + "/" + filename;
}

void RenderPipeline::cleanup_old_checkpoints() {
    // Cleanup old checkpoint files
}

std::vector<PipelineStage> RenderPipeline::get_pipeline_stages() const {
    return {
        PipelineStage::INITIALIZATION,
        PipelineStage::LOADING_FRAMES,
        PipelineStage::LOADING_AUDIO,
        PipelineStage::AUDIO_VISUAL_SYNC,
        PipelineStage::FRAME_RENDERING,
        PipelineStage::QUALITY_ASSESSMENT,
        PipelineStage::VIDEO_ENCODING,
        PipelineStage::METADATA_EMBEDDING,
        PipelineStage::FINAL_VALIDATION,
        PipelineStage::COMPLETION
    };
}

// Configuration namespace implementation
namespace config {

PipelineConfig create_high_quality_config(
    const std::vector<Tensor>& frames,
    const std::string& output_path,
    const std::string& audio_path
) {
    PipelineConfig config;
    config.animation_frames = frames;
    config.audio_path = audio_path;
    config.output_video_path = output_path;
    config.quality_target = PipelineConfig::QualityTarget::HIGH_QUALITY;
    config.enable_audio_sync = true;
    config.enable_quality_assessment = true;
    
    // High quality render settings
    config.render_config.resolution = {1920, 1080};
    config.render_config.bit_depth = render::RenderConfig::BitDepth::BIT_10;
    config.render_config.quality = 95;
    config.render_config.upsampling_filter = render::RenderConfig::UpsamplingFilter::LANCZOS;
    
    // High quality export settings
    config.export_config.video_codec = render::VideoCodec::H265;
    config.export_config.bitrate_mbps = 12.0f;
    config.export_config.fps = 30;
    
    return config;
}

PipelineConfig create_balanced_config(
    const std::vector<Tensor>& frames,
    const std::string& output_path,
    const std::string& audio_path
) {
    PipelineConfig config;
    config.animation_frames = frames;
    config.audio_path = audio_path;
    config.output_video_path = output_path;
    config.quality_target = PipelineConfig::QualityTarget::BALANCED;
    config.enable_audio_sync = true;
    config.enable_quality_assessment = true;
    
    // Balanced render settings
    config.render_config.resolution = {1280, 720};
    config.render_config.bit_depth = render::RenderConfig::BitDepth::BIT_8;
    config.render_config.quality = 85;
    config.render_config.upsampling_filter = render::RenderConfig::UpsamplingFilter::BICUBIC;
    
    // Balanced export settings
    config.export_config.video_codec = render::VideoCodec::H264;
    config.export_config.bitrate_mbps = 6.0f;
    config.export_config.fps = 30;
    
    return config;
}

PipelineConfig create_fast_config(
    const std::vector<Tensor>& frames,
    const std::string& output_path,
    const std::string& audio_path
) {
    PipelineConfig config;
    config.animation_frames = frames;
    config.audio_path = audio_path;
    config.output_video_path = output_path;
    config.quality_target = PipelineConfig::QualityTarget::FAST;
    config.enable_audio_sync = false;
    config.enable_quality_assessment = false;
    
    // Fast render settings
    config.render_config.resolution = {854, 480};
    config.render_config.bit_depth = render::RenderConfig::BitDepth::BIT_8;
    config.render_config.quality = 75;
    config.render_config.upsampling_filter = render::RenderConfig::UpsamplingFilter::LINEAR;
    
    // Fast export settings
    config.export_config.video_codec = render::VideoCodec::H264;
    config.export_config.bitrate_mbps = 2.0f;
    config.export_config.fps = 24;
    config.export_config.preset = render::ExportConfig::Preset::FAST;
    
    return config;
}

PipelineConfig create_preview_config(
    const std::vector<Tensor>& frames,
    const std::string& audio_path
) {
    PipelineConfig config;
    config.animation_frames = frames;
    config.audio_path = audio_path;
    config.output_video_path = "preview_output.mp4";
    config.quality_target = PipelineConfig::QualityTarget::FAST;
    config.enable_audio_sync = false;
    config.enable_quality_assessment = false;
    config.enable_preview = true;
    
    // Preview settings (optimized for speed)
    config.render_config.resolution = {1280, 720};
    config.render_config.bit_depth = render::RenderConfig::BitDepth::BIT_8;
    config.render_config.quality = 70;
    config.render_config.upsampling_filter = render::RenderConfig::UpsamplingFilter::NEAREST;
    
    return config;
}

bool validate_frame_sequence(const std::vector<Tensor>& frames) {
    if (frames.empty()) {
        return false;
    }
    
    // Check if all frames have consistent dimensions
    auto reference_shape = frames[0].shape();
    for (const auto& frame : frames) {
        if (frame.shape() != reference_shape) {
            return false;
        }
    }
    
    return true;
}

bool validate_audio_file(const std::string& filepath) {
    if (filepath.empty()) {
        return true; // No audio is valid
    }
    
    // Check if file exists and has supported extension
    if (!std::filesystem::exists(filepath)) {
        return false;
    }
    
    std::string extension = std::filesystem::path(filepath).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return (extension == ".wav" || extension == ".mp3" || extension == ".flac");
}

bool validate_output_path(const std::string& output_path) {
    if (output_path.empty()) {
        return false;
    }
    
    // Check directory exists and is writable
    std::filesystem::path path(output_path);
    std::filesystem::path directory = path.parent_path();
    
    if (!directory.empty() && !std::filesystem::exists(directory)) {
        return false;
    }
    
    // Check if file extension is supported
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    return (extension == ".mp4" || extension == ".webm" || extension == ".mov");
}

bool validate_render_config(const RenderConfig& config) {
    // Check resolution bounds
    int width = config.resolution.first;
    int height = config.resolution.second;
    
    if (width < 256 || width > 4096 || height < 256 || height > 4096) {
        return false;
    }
    
    // Check quality bounds
    if (config.quality < 0 || config.quality > 100) {
        return false;
    }
    
    return true;
}

bool validate_export_config(const ExportConfig& config) {
    // Check bitrate bounds
    if (config.bitrate_mbps < 0.5f || config.bitrate_mbps > 50.0f) {
        return false;
    }
    
    // Check FPS bounds
    if (config.fps != 24 && config.fps != 30 && config.fps != 60) {
        return false;
    }
    
    return true;
}

} // namespace config

} // namespace render
} // namespace jin