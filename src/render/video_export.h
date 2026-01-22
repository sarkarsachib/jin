#pragma once

#include "renderer.h"
#include "../audio/processor.h"
#include "../core/tensor.h"
#include <string>
#include <vector>
#include <map>

namespace jin {
namespace render {

// Video codec enumerations
enum class VideoCodec {
    H264, H265, VP9, PRORES
};

enum class AudioCodec {
    AAC, OPUS, PCM, VORBIS
};

enum class Container {
    MP4, WEBM, MOV, MKV
};

// Video export configuration
struct ExportConfig {
    VideoCodec video_codec;
    AudioCodec audio_codec;
    Container container;
    float bitrate_mbps;              // Target bitrate (1-50)
    int fps;                         // 24, 30, 60
    enum class Preset {
        FAST, BALANCED, SLOW
    } preset;
    
    ExportConfig()
        : video_codec(VideoCodec::H264), audio_codec(AudioCodec::AAC),
          container(Container::MP4), bitrate_mbps(8.0f), fps(30),
          preset(Preset::BALANCED) {}
};

// Metadata structure for video files
struct VideoMetadata {
    std::string title;
    std::string creator;
    std::string description;
    std::pair<int, int> resolution;
    float duration_seconds;
    int fps;
    int frame_count;
    float bitrate_mbps;
    std::string creation_date;
};

// Video exporter class
class VideoExporter {
public:
    VideoExporter();
    explicit VideoExporter(const ExportConfig& config);
    
    // Export video from frames and audio
    bool export_video(
        const std::vector<Tensor>& frames,
        const Tensor& audio,
        const std::string& output_path,
        const VideoMetadata& metadata = VideoMetadata{}
    );
    
    // Export image sequence
    bool export_image_sequence(
        const std::vector<Tensor>& frames,
        const std::string& output_dir,
        const std::string& format = "png"
    );
    
    // Export animated GIF
    bool export_gif(
        const std::vector<Tensor>& frames,
        const std::string& output_path,
        int duration_per_frame_ms = 33  // ~30 FPS
    );
    
    // Metadata operations
    bool add_metadata(const std::string& video_path, const VideoMetadata& metadata);
    VideoMetadata extract_metadata(const std::string& video_path);
    
    // Utility methods
    bool is_codec_available(VideoCodec codec) const;
    bool is_audio_codec_available(AudioCodec codec) const;
    size_t compute_file_size_estimate(
        int num_frames,
        int fps,
        const ExportConfig& config
    ) const;
    
    // Configuration management
    void set_config(const ExportConfig& config) { config_ = config; }
    const ExportConfig& get_config() const { return config_; }
    
    // Progress tracking
    void set_progress_callback(std::function<void(float progress)> callback);
    void set_encode_callback(std::function<void(int frame, int total_frames)> callback);

private:
    ExportConfig config_;
    std::function<void(float)> progress_callback_;
    std::function<void(int, int)> encode_callback_;
    
    // Helper methods for different export formats
    bool export_h264_video(
        const std::vector<Tensor>& frames,
        const Tensor& audio,
        const std::string& output_path,
        const VideoMetadata& metadata
    );
    
    bool export_h265_video(
        const std::vector<Tensor>& frames,
        const Tensor& audio,
        const std::string& output_path,
        const VideoMetadata& metadata
    );
    
    bool export_webm_video(
        const std::vector<Tensor>& frames,
        const Tensor& audio,
        const std::string& output_path,
        const VideoMetadata& metadata
    );
    
    // Frame sequence export
    bool export_frame_sequence(
        const std::vector<Tensor>& frames,
        const std::string& output_dir,
        const std::string& format
    );
    
    // Audio processing for video export
    Tensor process_audio_for_video(const Tensor& audio, int target_sample_rate);
    
    // Format-specific helpers
    std::string get_container_extension() const;
    std::string get_video_codec_extension() const;
    std::string get_audio_codec_extension() const;
    
    // FFmpeg wrapper methods (simplified)
    bool run_ffmpeg_command(const std::vector<std::string>& args);
    bool check_ffmpeg_availability();
    
    // Quality and compression settings
    std::vector<std::string> get_video_codec_args() const;
    std::vector<std::string> get_audio_codec_args() const;
    std::vector<std::string> get_container_args() const;
    
    // File operations
    bool create_temp_frame_files(const std::vector<Tensor>& frames, const std::string& temp_dir);
    bool cleanup_temp_files(const std::string& temp_dir);
    std::string create_temp_directory();
    
    // Progress tracking
    void update_progress(float progress);
    void update_encode_progress(int frame, int total);
};

// Simplified image sequence exporter
class ImageSequenceExporter {
public:
    ImageSequenceExporter();
    
    bool export_sequence(
        const std::vector<Tensor>& frames,
        const std::string& output_pattern,  // e.g., "frame_%04d.png"
        const std::string& format = "png"
    );
    
    bool export_hdri_sequence(
        const std::vector<Tensor>& frames,
        const std::string& output_pattern,  // e.g., "frame_%04d.exr"
        const std::string& format = "exr"
    );

private:
    std::string format_frame_number(int frame_number, int total_digits = 4) const;
};

// GIF exporter (simplified)
class GIFExporter {
public:
    GIFExporter();
    
    bool export_gif(
        const std::vector<Tensor>& frames,
        const std::string& output_path,
        int duration_per_frame_ms = 100,
        int max_colors = 256
    );

private:
    // GIF-specific helpers
    std::vector<uint8_t> create_gif_palette(const std::vector<Tensor>& frames);
    bool quantize_frames(const std::vector<Tensor>& frames, int max_colors);
};

} // namespace render
} // namespace jin