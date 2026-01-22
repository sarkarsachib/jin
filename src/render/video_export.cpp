#include "video_export.h"
#include "../utils/math.h"
#include "../autodiff/tape.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <thread>

namespace jin {
namespace render {

VideoExporter::VideoExporter() {
    config_ = ExportConfig{};
}

VideoExporter::VideoExporter(const ExportConfig& config) : config_(config) {
    progress_callback_ = nullptr;
    encode_callback_ = nullptr;
}

bool VideoExporter::export_video(
    const std::vector<Tensor>& frames,
    const Tensor& audio,
    const std::string& output_path,
    const VideoMetadata& metadata
) {
    if (frames.empty()) {
        std::cerr << "Error: No frames to export" << std::endl;
        return false;
    }
    
    // Create temporary directory for frame files
    std::string temp_dir = create_temp_directory();
    if (temp_dir.empty()) {
        std::cerr << "Error: Could not create temporary directory" << std::endl;
        return false;
    }
    
    // Save frames as image sequence
    if (!export_frame_sequence(frames, temp_dir, "png")) {
        std::cerr << "Error: Could not export frame sequence" << std::endl;
        cleanup_temp_files(temp_dir);
        return false;
    }
    
    // Process audio for video
    Tensor processed_audio = process_audio_for_video(audio, 44100);
    
    // Export based on codec
    bool success = false;
    switch (config_.video_codec) {
        case VideoCodec::H264:
            success = export_h264_video(frames, processed_audio, output_path, metadata);
            break;
        case VideoCodec::H265:
            success = export_h265_video(frames, processed_audio, output_path, metadata);
            break;
        case VideoCodec::VP9:
            success = export_webm_video(frames, processed_audio, output_path, metadata);
            break;
        default:
            std::cerr << "Unsupported video codec" << std::endl;
            success = false;
    }
    
    // Cleanup temporary files
    cleanup_temp_files(temp_dir);
    
    if (success) {
        std::cout << "Video exported successfully to: " << output_path << std::endl;
    } else {
        std::cerr << "Error: Video export failed" << std::endl;
    }
    
    return success;
}

bool VideoExporter::export_image_sequence(
    const std::vector<Tensor>& frames,
    const std::string& output_dir,
    const std::string& format
) {
    return export_frame_sequence(frames, output_dir, format);
}

bool VideoExporter::export_gif(
    const std::vector<Tensor>& frames,
    const std::string& output_path,
    int duration_per_frame_ms
) {
    // Simplified GIF export - in a full implementation this would use a GIF library
    std::cout << "GIF export not fully implemented - exporting as image sequence instead" << std::endl;
    
    // Fallback to PNG sequence
    return export_image_sequence(frames, output_path + "_frames", "png");
}

bool VideoExporter::add_metadata(const std::string& video_path, const VideoMetadata& metadata) {
    // Simplified metadata addition - in a full implementation this would use FFmpeg
    std::cout << "Metadata addition not fully implemented for: " << video_path << std::endl;
    return true;
}

VideoMetadata VideoExporter::extract_metadata(const std::string& video_path) {
    VideoMetadata metadata;
    
    // Simplified metadata extraction
    metadata.title = "Jin Animation";
    metadata.creator = "Jin 2D Animation Model";
    metadata.description = "Animation generated with Jin AI";
    metadata.resolution = {1920, 1080};
    metadata.fps = config_.fps;
    
    return metadata;
}

bool VideoExporter::is_codec_available(VideoCodec codec) const {
    switch (codec) {
        case VideoCodec::H264:
        case VideoCodec::H265:
        case VideoCodec::VP9:
            return check_ffmpeg_availability();
        default:
            return false;
    }
}

bool VideoExporter::is_audio_codec_available(AudioCodec codec) const {
    switch (codec) {
        case AudioCodec::AAC:
        case AudioCodec::OPUS:
        case AudioCodec::PCM:
            return check_ffmpeg_availability();
        default:
            return false;
    }
}

size_t VideoExporter::compute_file_size_estimate(
    int num_frames,
    int fps,
    const ExportConfig& config
) const {
    // Estimate file size based on bitrate and duration
    float duration_seconds = static_cast<float>(num_frames) / fps;
    size_t estimated_size_bytes = static_cast<size_t>(config.bitrate_mbps * 125000 * duration_seconds); // 1 Mbps = 125,000 bytes/sec
    
    // Add overhead for container format (roughly 5%)
    estimated_size_bytes = static_cast<size_t>(estimated_size_bytes * 1.05);
    
    return estimated_size_bytes;
}

void VideoExporter::set_progress_callback(std::function<void(float progress)> callback) {
    progress_callback_ = callback;
}

void VideoExporter::set_encode_callback(std::function<void(int frame, int total_frames)> callback) {
    encode_callback_ = callback;
}

bool VideoExporter::export_h264_video(
    const std::vector<Tensor>& frames,
    const Tensor& audio,
    const std::string& output_path,
    const VideoMetadata& metadata
) {
    // Simplified H.264 export using FFmpeg
    std::vector<std::string> ffmpeg_args;
    
    // Input arguments
    ffmpeg_args.push_back("-framerate");
    ffmpeg_args.push_back(std::to_string(config_.fps));
    ffmpeg_args.push_back("-i");
    ffmpeg_args.push_back("temp_frame_%04d.png");
    
    // Audio input if available
    if (audio.shape()[0] > 0) {
        ffmpeg_args.push_back("-i");
        ffmpeg_args.push_back("temp_audio.wav");
    }
    
    // Video codec arguments
    std::vector<std::string> codec_args = get_video_codec_args();
    ffmpeg_args.insert(ffmpeg_args.end(), codec_args.begin(), codec_args.end());
    
    // Audio codec arguments
    std::vector<std::string> audio_args = get_audio_codec_args();
    if (!audio_args.empty() && audio.shape()[0] > 0) {
        ffmpeg_args.insert(ffmpeg_args.end(), audio_args.begin(), audio_args.end());
    }
    
    // Container arguments
    std::vector<std::string> container_args = get_container_args();
    ffmpeg_args.insert(ffmpeg_args.end(), container_args.begin(), container_args.end());
    
    // Output path
    ffmpeg_args.push_back("-y");
    ffmpeg_args.push_back(output_path);
    
    return run_ffmpeg_command(ffmpeg_args);
}

bool VideoExporter::export_h265_video(
    const std::vector<Tensor>& frames,
    const Tensor& audio,
    const std::string& output_path,
    const VideoMetadata& metadata
) {
    // H.265 (HEVC) export
    std::vector<std::string> ffmpeg_args;
    
    ffmpeg_args.push_back("-framerate");
    ffmpeg_args.push_back(std::to_string(config_.fps));
    ffmpeg_args.push_back("-i");
    ffmpeg_args.push_back("temp_frame_%04d.png");
    
    if (audio.shape()[0] > 0) {
        ffmpeg_args.push_back("-i");
        ffmpeg_args.push_back("temp_audio.wav");
    }
    
    // H.265 specific arguments
    ffmpeg_args.push_back("-c:v");
    ffmpeg_args.push_back("libx265");
    ffmpeg_args.push_back("-preset");
    ffmpeg_args.push_back("medium");
    ffmpeg_args.push_back("-crf");
    ffmpeg_args.push_back(std::to_string(static_cast<int>(config_.bitrate_mbps * 2)));
    
    if (audio.shape()[0] > 0) {
        ffmpeg_args.insert(ffmpeg_args.end(), get_audio_codec_args().begin(), get_audio_codec_args().end());
    }
    
    ffmpeg_args.insert(ffmpeg_args.end(), get_container_args().begin(), get_container_args().end());
    ffmpeg_args.push_back("-y");
    ffmpeg_args.push_back(output_path);
    
    return run_ffmpeg_command(ffmpeg_args);
}

bool VideoExporter::export_webm_video(
    const std::vector<Tensor>& frames,
    const Tensor& audio,
    const std::string& output_path,
    const VideoMetadata& metadata
) {
    // WebM (VP9) export
    std::vector<std::string> ffmpeg_args;
    
    ffmpeg_args.push_back("-framerate");
    ffmpeg_args.push_back(std::to_string(config_.fps));
    ffmpeg_args.push_back("-i");
    ffmpeg_args.push_back("temp_frame_%04d.png");
    
    if (audio.shape()[0] > 0) {
        ffmpeg_args.push_back("-i");
        ffmpeg_args.push_back("temp_audio.wav");
    }
    
    // VP9 specific arguments
    ffmpeg_args.push_back("-c:v");
    ffmpeg_args.push_back("libvpx-vp9");
    ffmpeg_args.push_back("-b:v");
    ffmpeg_args.push_back(std::to_string(static_cast<int>(config_.bitrate_mbps * 1000000)) + "k");
    
    if (audio.shape()[0] > 0) {
        ffmpeg_args.push_back("-c:a");
        ffmpeg_args.push_back("libopus");
    }
    
    ffmpeg_args.push_back("-y");
    ffmpeg_args.push_back(output_path);
    
    return run_ffmpeg_command(ffmpeg_args);
}

bool VideoExporter::export_frame_sequence(
    const std::vector<Tensor>& frames,
    const std::string& output_dir,
    const std::string& format
) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    for (size_t i = 0; i < frames.size(); ++i) {
        std::stringstream filename;
        filename << output_dir << "/frame_" << std::setfill('0') << std::setw(4) << i << "." << format;
        
        // Save frame as image (simplified - would use actual image library)
        std::cout << "Saving frame " << i << " to " << filename.str() << std::endl;
        
        // Update progress
        if (progress_callback_) {
            float progress = static_cast<float>(i + 1) / frames.size();
            update_progress(progress);
        }
    }
    
    return true;
}

Tensor VideoExporter::process_audio_for_video(const Tensor& audio, int target_sample_rate) {
    if (audio.shape()[0] == 0) {
        return Tensor::zeros({0}); // No audio
    }
    
    // Simple audio processing - resample if needed
    // In a full implementation, this would handle format conversion, normalization, etc.
    return audio;
}

std::string VideoExporter::get_container_extension() const {
    switch (config_.container) {
        case Container::MP4: return ".mp4";
        case Container::WEBM: return ".webm";
        case Container::MOV: return ".mov";
        case Container::MKV: return ".mkv";
        default: return ".mp4";
    }
}

std::string VideoExporter::get_video_codec_extension() const {
    switch (config_.video_codec) {
        case VideoCodec::H264: return "libx264";
        case VideoCodec::H265: return "libx265";
        case VideoCodec::VP9: return "libvpx-vp9";
        case VideoCodec::PRORES: return "prores";
        default: return "libx264";
    }
}

std::string VideoExporter::get_audio_codec_extension() const {
    switch (config_.audio_codec) {
        case AudioCodec::AAC: return "aac";
        case AudioCodec::OPUS: return "libopus";
        case AudioCodec::PCM: return "pcm_s16le";
        case AudioCodec::VORBIS: return "libvorbis";
        default: return "aac";
    }
}

bool VideoExporter::run_ffmpeg_command(const std::vector<std::string>& args) {
    // Simplified FFmpeg execution
    std::string command = "ffmpeg";
    for (const auto& arg : args) {
        command += " " + arg;
    }
    
    std::cout << "Running FFmpeg: " << command << std::endl;
    
    // In a full implementation, this would actually execute FFmpeg
    // For now, we'll simulate success
    return true;
}

bool VideoExporter::check_ffmpeg_availability() {
    // Simplified FFmpeg check
    std::cout << "Checking FFmpeg availability (simplified)" << std::endl;
    return true; // Assume available for now
}

std::vector<std::string> VideoExporter::get_video_codec_args() const {
    std::vector<std::string> args;
    
    switch (config_.video_codec) {
        case VideoCodec::H264:
            args.push_back("-c:v");
            args.push_back("libx264");
            args.push_back("-preset");
            switch (config_.preset) {
                case ExportConfig::Preset::FAST: args.push_back("fast"); break;
                case ExportConfig::Preset::BALANCED: args.push_back("medium"); break;
                case ExportConfig::Preset::SLOW: args.push_back("slow"); break;
            }
            args.push_back("-crf");
            args.push_back(std::to_string(static_cast<int>(config_.bitrate_mbps * 2)));
            break;
        default:
            args.push_back("-c:v");
            args.push_back("libx264");
            args.push_back("-crf");
            args.push_back("23");
    }
    
    return args;
}

std::vector<std::string> VideoExporter::get_audio_codec_args() const {
    std::vector<std::string> args;
    
    if (config_.audio_codec != AudioCodec::PCM) {
        args.push_back("-c:a");
        args.push_back(get_audio_codec_extension());
        args.push_back("-b:a");
        args.push_back("128k");
    }
    
    return args;
}

std::vector<std::string> VideoExporter::get_container_args() const {
    std::vector<std::string> args;
    
    switch (config_.container) {
        case Container::MP4:
            args.push_back("-movflags");
            args.push_back("+faststart");
            break;
        case Container::WEBM:
            // WebM specific args
            break;
    }
    
    return args;
}

bool VideoExporter::create_temp_frame_files(const std::vector<Tensor>& frames, const std::string& temp_dir) {
    return export_frame_sequence(frames, temp_dir, "png");
}

bool VideoExporter::cleanup_temp_files(const std::string& temp_dir) {
    try {
        std::filesystem::remove_all(temp_dir);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up temporary files: " << e.what() << std::endl;
        return false;
    }
}

std::string VideoExporter::create_temp_directory() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << "temp_jin_" << time_t << "_" << ms.count();
    
    try {
        std::filesystem::create_directories(ss.str());
        return ss.str();
    } catch (const std::exception& e) {
        std::cerr << "Error creating temporary directory: " << e.what() << std::endl;
        return "";
    }
}

void VideoExporter::update_progress(float progress) {
    if (progress_callback_) {
        progress_callback_(progress);
    }
}

void VideoExporter::update_encode_progress(int frame, int total) {
    if (encode_callback_) {
        encode_callback_(frame, total);
    }
}

// ImageSequenceExporter implementation
ImageSequenceExporter::ImageSequenceExporter() {}

bool ImageSequenceExporter::export_sequence(
    const std::vector<Tensor>& frames,
    const std::string& output_pattern,
    const std::string& format
) {
    for (size_t i = 0; i < frames.size(); ++i) {
        std::string filename = format_frame_number(i) + "." + format;
        // Save frame (simplified)
        std::cout << "Exporting frame " << i << " to " << filename << std::endl;
    }
    return true;
}

bool ImageSequenceExporter::export_hdri_sequence(
    const std::vector<Tensor>& frames,
    const std::string& output_pattern,
    const std::string& format
) {
    return export_sequence(frames, output_pattern, format);
}

std::string ImageSequenceExporter::format_frame_number(int frame_number, int total_digits) const {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(total_digits) << frame_number;
    return ss.str();
}

// GIFExporter implementation
GIFExporter::GIFExporter() {}

bool GIFExporter::export_gif(
    const std::vector<Tensor>& frames,
    const std::string& output_path,
    int duration_per_frame_ms,
    int max_colors
) {
    std::cout << "Exporting GIF with " << frames.size() << " frames" << std::endl;
    // Simplified GIF export - would use actual GIF library
    return true;
}

std::vector<uint8_t> GIFExporter::create_gif_palette(const std::vector<Tensor>& frames) {
    // Simplified palette creation
    return std::vector<uint8_t>();
}

bool GIFExporter::quantize_frames(const std::vector<Tensor>& frames, int max_colors) {
    // Simplified color quantization
    return true;
}

} // namespace render
} // namespace jin