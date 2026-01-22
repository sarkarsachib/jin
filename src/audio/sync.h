#pragma once

#include "processor.h"
#include "../temporal/diffusion_loop.h"
#include "../core/tensor.h"
#include <vector>
#include <string>
#include <map>

namespace jin {
namespace audio {

// Sync point structure representing alignment between audio time and frame
struct SyncPoint {
    float audio_time;      // Time in seconds
    int frame_index;       // Animation frame index
    float confidence;      // Alignment confidence [0, 1]
    
    SyncPoint(float time, int frame, float conf) 
        : audio_time(time), frame_index(frame), confidence(conf) {}
};

// Sync correction curve for temporal warping
struct CorrectionCurve {
    std::vector<float> original_times;   // Original time points
    std::vector<float> warped_times;      // Warped time points
    bool is_monotonic;                   // Whether curve is monotonic (no reversals)
};

// Audio-visual synchronization configuration
struct SyncConfig {
    float correlation_threshold;         // Minimum correlation for sync [0.1, 1.0]
    int max_alignment_lag;              // Maximum frames of alignment lag
    bool use_beat_sync;                 // Whether to prioritize beat alignment
    float temporal_smoothing_strength;   // Smoothing for correction curve [0.0, 1.0]
    int motion_analysis_window;          // Window size for motion analysis
    
    SyncConfig() 
        : correlation_threshold(0.5f), max_alignment_lag(5), 
          use_beat_sync(true), temporal_smoothing_strength(0.3f), 
          motion_analysis_window(8) {}
};

// Quality metrics for synchronization
struct SyncMetrics {
    float alignment_score;              // Overall alignment quality [0, 1]
    float correlation_coefficient;       // Audio-motion correlation
    float max_frame_drift;              // Maximum frame timing error (frames)
    float average_frame_drift;          // Average frame timing error (frames)
    int num_sync_points;                 // Number of detected sync points
    float sync_point_confidence;         // Average confidence of sync points
};

class AudioSync {
public:
    AudioSync(const SyncConfig& config = SyncConfig{});
    
    // Synchronize animation frames with audio
    std::vector<Tensor> sync_to_audio(
        const std::vector<Tensor>& frames,
        const Tensor& audio,
        int audio_sample_rate,
        std::vector<SyncPoint>* sync_points_out = nullptr
    );
    
    // Compute alignment score between frames and audio
    float compute_alignment_score(
        const std::vector<Tensor>& frames,
        const Tensor& audio,
        const std::vector<float>& timestamps
    );
    
    // Get interpolated frame at specific time
    Tensor get_frame_at_time(
        const std::vector<Tensor>& frames,
        float time_seconds,
        float fps
    );
    
    // Apply temporal warping correction
    std::vector<Tensor> apply_sync_correction(
        const std::vector<Tensor>& frames,
        const CorrectionCurve& correction_curve
    );
    
    // Extract synchronization points from audio and animation
    std::vector<SyncPoint> extract_sync_points(
        const Tensor& audio,
        const std::vector<Tensor>& frames,
        float fps
    );
    
    // Compute motion features for frames
    std::vector<float> compute_motion_features(const std::vector<Tensor>& frames);
    
    // Analyze audio-motion correlation
    float compute_audio_motion_correlation(
        const std::vector<float>& audio_features,
        const std::vector<float>& motion_features
    );
    
    // Get sync metrics
    SyncMetrics get_sync_metrics() const { return sync_metrics_; }
    
    // Update configuration
    void set_config(const SyncConfig& config) { config_ = config; }
    const SyncConfig& get_config() const { return config_; }
    
    // Advanced sync methods
    std::vector<Tensor> dynamic_time_warping_sync(
        const std::vector<Tensor>& frames,
        const std::vector<float>& audio_features,
        float fps
    );
    
    CorrectionCurve compute_correction_curve(
        const std::vector<SyncPoint>& sync_points,
        int num_frames
    );
    
    // Quality assessment
    std::map<std::string, float> assess_sync_quality(
        const std::vector<Tensor>& frames,
        const Tensor& audio
    );

private:
    SyncConfig config_;
    SyncMetrics sync_metrics_;
    
    // Audio processor for feature extraction
    AudioProcessor audio_processor_;
    
    // Helper methods
    std::vector<float> extract_temporal_features(const std::vector<Tensor>& frames);
    Tensor create_interpolation_matrix(const std::vector<float>& times);
    std::vector<float> smooth_correction_curve(const std::vector<float>& curve, float strength);
    
    // Motion analysis helpers
    std::vector<float> compute_optical_flow_magnitude(const std::vector<Tensor>& frames);
    std::vector<float> compute_frame_differences(const std::vector<Tensor>& frames);
    
    // Audio analysis helpers
    std::vector<float> extract_beats_as_features(const Tensor& audio);
    std::vector<float> extract_energy_envelope(const Tensor& audio);
    
    // Correlation computation
    float pearson_correlation(const std::vector<float>& x, const std::vector<float>& y);
    
    // DTW helpers
    std::vector<std::vector<float>> compute_distance_matrix(
        const std::vector<float>& sequence_a,
        const std::vector<float>& sequence_b
    );
    std::vector<std::pair<int, int>> find_optimal_alignment(
        const std::vector<std::vector<float>>& distance_matrix
    );
};

} // namespace audio
} // namespace jin