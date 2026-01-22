#include "sync.h"
#include "../utils/math.h"
#include "../autodiff/tape.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

namespace jin {
namespace audio {

AudioSync::AudioSync(const SyncConfig& config) 
    : config_(config), audio_processor_(AudioConfig{}) {
    sync_metrics_ = SyncMetrics{};
}

std::vector<Tensor> AudioSync::sync_to_audio(
    const std::vector<Tensor>& frames,
    const Tensor& audio,
    int audio_sample_rate,
    std::vector<SyncPoint>* sync_points_out
) {
    // Extract audio features for sync analysis
    Tensor audio_features = audio_processor_.extract_features(audio, FeatureType::ENERGY);
    std::vector<float> energy_envelope = audio_features.get_col(0).to_vector<float>();
    
    // Compute motion features from frames
    std::vector<float> motion_features = compute_motion_features(frames);
    
    // Extract sync points
    float fps = static_cast<float>(frames.size()) / audio_processor_.get_audio_length_seconds(audio);
    std::vector<SyncPoint> sync_points = extract_sync_points(audio, frames, fps);
    
    if (sync_points_out) {
        *sync_points_out = sync_points;
    }
    
    // Compute alignment score
    float alignment_score = compute_alignment_score(frames, audio, {});
    
    // If alignment is poor, apply corrections
    std::vector<Tensor> corrected_frames = frames;
    if (alignment_score < config_.correlation_threshold) {
        // Compute correction curve
        CorrectionCurve correction_curve = compute_correction_curve(sync_points, frames.size());
        
        // Apply temporal warping
        corrected_frames = apply_sync_correction(frames, correction_curve);
    }
    
    // Update sync metrics
    sync_metrics_.alignment_score = alignment_score;
    sync_metrics_.correlation_coefficient = compute_audio_motion_correlation(energy_envelope, motion_features);
    sync_metrics_.num_sync_points = sync_points.size();
    sync_metrics_.sync_point_confidence = sync_points.empty() ? 0.0f : 
        std::accumulate(sync_points.begin(), sync_points.end(), 0.0f, 
                       [](float sum, const SyncPoint& sp) { return sum + sp.confidence; }) / sync_points.size();
    
    return corrected_frames;
}

float AudioSync::compute_alignment_score(
    const std::vector<Tensor>& frames,
    const Tensor& audio,
    const std::vector<float>& timestamps
) {
    // Extract audio energy envelope
    Tensor energy_features = audio_processor_.extract_features(audio, FeatureType::ENERGY);
    std::vector<float> audio_energy = energy_features.get_col(0).to_vector<float>();
    
    // Compute motion features
    std::vector<float> motion_features = compute_motion_features(frames);
    
    // Compute correlation
    float correlation = compute_audio_motion_correlation(audio_energy, motion_features);
    
    return std::max(0.0f, correlation); // Ensure non-negative
}

Tensor AudioSync::get_frame_at_time(
    const std::vector<Tensor>& frames,
    float time_seconds,
    float fps
) {
    float frame_rate = 1.0f / fps;
    float frame_position = time_seconds / frame_rate;
    
    int left_frame = static_cast<int>(std::floor(frame_position));
    int right_frame = left_frame + 1;
    float alpha = frame_position - left_frame;
    
    // Handle boundary conditions
    if (left_frame < 0) {
        return frames[0];
    } else if (right_frame >= static_cast<int>(frames.size())) {
        return frames.back();
    }
    
    // Linear interpolation between frames
    Tensor left_frame_tensor = frames[left_frame];
    Tensor right_frame_tensor = frames[right_frame];
    
    return left_frame_tensor * (1.0f - alpha) + right_frame_tensor * alpha;
}

std::vector<Tensor> AudioSync::apply_sync_correction(
    const std::vector<Tensor>& frames,
    const CorrectionCurve& correction_curve
) {
    if (!correction_curve.is_monotonic) {
        std::cerr << "Warning: Correction curve is not monotonic!" << std::endl;
    }
    
    std::vector<Tensor> corrected_frames;
    corrected_frames.reserve(frames.size());
    
    // Get original frame timing
    int num_frames = frames.size();
    float original_fps = static_cast<float>(num_frames) / 
        (correction_curve.original_times.back() - correction_curve.original_times.front());
    
    for (int i = 0; i < num_frames; ++i) {
        float original_time = static_cast<float>(i) / original_fps;
        
        // Find corresponding warped time
        float warped_time = original_time;
        for (size_t j = 1; j < correction_curve.original_times.size(); ++j) {
            if (original_time <= correction_curve.original_times[j]) {
                float alpha = (original_time - correction_curve.original_times[j-1]) / 
                             (correction_curve.original_times[j] - correction_curve.original_times[j-1]);
                warped_time = correction_curve.warped_times[j-1] + alpha * 
                             (correction_curve.warped_times[j] - correction_curve.warped_times[j-1]);
                break;
            }
        }
        
        // Get interpolated frame at warped time
        Tensor corrected_frame = get_frame_at_time(frames, warped_time, original_fps);
        corrected_frames.push_back(corrected_frame);
    }
    
    return corrected_frames;
}

std::vector<SyncPoint> AudioSync::extract_sync_points(
    const Tensor& audio,
    const std::vector<Tensor>& frames,
    float fps
) {
    std::vector<SyncPoint> sync_points;
    
    // Extract beats from audio
    std::vector<float> beats = audio_processor_.detect_beats(audio);
    
    // For each beat, find corresponding frame
    for (float beat_time : beats) {
        int frame_index = static_cast<int>(std::round(beat_time * fps));
        
        if (frame_index >= 0 && frame_index < static_cast<int>(frames.size())) {
            // Compute confidence based on motion magnitude at this frame
            std::vector<float> motion_features = compute_motion_features(frames);
            float motion_magnitude = motion_features[std::min(frame_index, static_cast<int>(motion_features.size()) - 1)];
            float confidence = std::min(1.0f, motion_magnitude / 100.0f); // Normalize confidence
            
            sync_points.emplace_back(beat_time, frame_index, confidence);
        }
    }
    
    return sync_points;
}

std::vector<float> AudioSync::compute_motion_features(const std::vector<Tensor>& frames) {
    std::vector<float> motion_features;
    
    if (frames.size() < 2) {
        return std::vector<float>(frames.size(), 0.0f);
    }
    
    // Compute frame-to-frame differences as motion magnitude
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        float motion_magnitude = diff.abs().sum().item<float>();
        motion_features.push_back(motion_magnitude);
    }
    
    // Add first frame with zero motion
    motion_features.insert(motion_features.begin(), 0.0f);
    
    return motion_features;
}

float AudioSync::compute_audio_motion_correlation(
    const std::vector<float>& audio_features,
    const std::vector<float>& motion_features
) {
    if (audio_features.empty() || motion_features.empty()) {
        return 0.0f;
    }
    
    // Ensure same length by interpolation
    size_t min_length = std::min(audio_features.size(), motion_features.size());
    std::vector<float> audio_subset(audio_features.begin(), audio_features.begin() + min_length);
    std::vector<float> motion_subset(motion_features.begin(), motion_features.begin() + min_length);
    
    return pearson_correlation(audio_subset, motion_subset);
}

CorrectionCurve AudioSync::compute_correction_curve(
    const std::vector<SyncPoint>& sync_points,
    int num_frames
) {
    CorrectionCurve curve;
    
    if (sync_points.empty()) {
        // Return identity curve
        curve.original_times = {0.0f, 1.0f};
        curve.warped_times = {0.0f, 1.0f};
        curve.is_monotonic = true;
        return curve;
    }
    
    // Create smooth curve through sync points
    std::vector<float> original_times;
    std::vector<float> warped_times;
    
    float max_original_time = static_cast<float>(num_frames);
    
    for (const auto& sync_point : sync_points) {
        original_times.push_back(sync_point.frame_index);
        warped_times.push_back(sync_point.audio_time * max_original_time);
    }
    
    // Add endpoints
    original_times.insert(original_times.begin(), 0.0f);
    original_times.push_back(max_original_time);
    warped_times.insert(warped_times.begin(), 0.0f);
    warped_times.push_back(max_original_time);
    
    // Sort by original time
    std::vector<size_t> indices(original_times.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
              [&original_times](size_t a, size_t b) { return original_times[a] < original_times[b]; });
    
    std::vector<float> sorted_original;
    std::vector<float> sorted_warped;
    for (size_t idx : indices) {
        sorted_original.push_back(original_times[idx]);
        sorted_warped.push_back(warped_times[idx]);
    }
    
    // Apply temporal smoothing
    sorted_warped = smooth_correction_curve(sorted_warped, config_.temporal_smoothing_strength);
    
    curve.original_times = sorted_original;
    curve.warped_times = sorted_warped;
    
    // Check monotonicity
    curve.is_monotonic = true;
    for (size_t i = 1; i < sorted_warped.size(); ++i) {
        if (sorted_warped[i] < sorted_warped[i-1]) {
            curve.is_monotonic = false;
            break;
        }
    }
    
    return curve;
}

std::vector<Tensor> AudioSync::dynamic_time_warping_sync(
    const std::vector<Tensor>& frames,
    const std::vector<float>& audio_features,
    float fps
) {
    // Extract motion features from frames
    std::vector<float> motion_features = compute_motion_features(frames);
    
    // Compute distance matrix
    std::vector<std::vector<float>> distance_matrix = 
        compute_distance_matrix(audio_features, motion_features);
    
    // Find optimal alignment
    std::vector<std::pair<int, int>> alignment = find_optimal_alignment(distance_matrix);
    
    // Apply alignment to frames
    std::vector<Tensor> aligned_frames;
    for (const auto& pair : alignment) {
        int frame_idx = pair.second;
        if (frame_idx >= 0 && frame_idx < static_cast<int>(frames.size())) {
            aligned_frames.push_back(frames[frame_idx]);
        }
    }
    
    return aligned_frames;
}

std::map<std::string, float> AudioSync::assess_sync_quality(
    const std::vector<Tensor>& frames,
    const Tensor& audio
) {
    std::map<std::string, float> quality_metrics;
    
    // Alignment score
    quality_metrics["alignment_score"] = compute_alignment_score(frames, audio, {});
    
    // Motion smoothness
    std::vector<float> motion_features = compute_motion_features(frames);
    float motion_smoothness = 1.0f;
    if (motion_features.size() > 1) {
        float total_motion = std::accumulate(motion_features.begin(), motion_features.end(), 0.0f);
        float avg_motion = total_motion / motion_features.size();
        float motion_variance = 0.0f;
        for (float motion : motion_features) {
            float diff = motion - avg_motion;
            motion_variance += diff * diff;
        }
        motion_variance /= motion_features.size();
        motion_smoothness = 1.0f / (1.0f + motion_variance);
    }
    quality_metrics["motion_smoothness"] = motion_smoothness;
    
    // Temporal coherence
    if (frames.size() > 1) {
        float temporal_coherence = 0.0f;
        for (size_t i = 1; i < frames.size(); ++i) {
            Tensor diff = frames[i] - frames[i-1];
            temporal_coherence += diff.abs().sum().item<float>();
        }
        temporal_coherence /= frames.size();
        quality_metrics["temporal_coherence"] = 1.0f / (1.0f + temporal_coherence);
    }
    
    return quality_metrics;
}

std::vector<float> AudioSync::compute_optical_flow_magnitude(const std::vector<Tensor>& frames) {
    // Simplified optical flow computation using frame differences
    return compute_frame_differences(frames);
}

std::vector<float> AudioSync::compute_frame_differences(const std::vector<Tensor>& frames) {
    std::vector<float> frame_diffs;
    
    for (size_t i = 1; i < frames.size(); ++i) {
        Tensor diff = frames[i] - frames[i-1];
        float diff_magnitude = diff.abs().sum().item<float>();
        frame_diffs.push_back(diff_magnitude);
    }
    
    return frame_diffs;
}

std::vector<float> AudioSync::extract_beats_as_features(const Tensor& audio) {
    std::vector<float> beats = audio_processor_.detect_beats(audio);
    return beats;
}

std::vector<float> AudioSync::extract_energy_envelope(const Tensor& audio) {
    Tensor energy_features = audio_processor_.extract_features(audio, FeatureType::ENERGY);
    return energy_features.get_col(0).to_vector<float>();
}

float AudioSync::pearson_correlation(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) {
        return 0.0f;
    }
    
    float mean_x = std::accumulate(x.begin(), x.end(), 0.0f) / x.size();
    float mean_y = std::accumulate(y.begin(), y.end(), 0.0f) / y.size();
    
    float numerator = 0.0f;
    float denom_x = 0.0f;
    float denom_y = 0.0f;
    
    for (size_t i = 0; i < x.size(); ++i) {
        float diff_x = x[i] - mean_x;
        float diff_y = y[i] - mean_y;
        numerator += diff_x * diff_y;
        denom_x += diff_x * diff_x;
        denom_y += diff_y * diff_y;
    }
    
    float denominator = std::sqrt(denom_x * denom_y);
    if (denominator < 1e-10f) {
        return 0.0f;
    }
    
    return numerator / denominator;
}

std::vector<std::vector<float>> AudioSync::compute_distance_matrix(
    const std::vector<float>& sequence_a,
    const std::vector<float>& sequence_b
) {
    std::vector<std::vector<float>> distance_matrix(sequence_a.size(), 
                                                   std::vector<float>(sequence_b.size()));
    
    for (size_t i = 0; i < sequence_a.size(); ++i) {
        for (size_t j = 0; j < sequence_b.size(); ++j) {
            distance_matrix[i][j] = std::abs(sequence_a[i] - sequence_b[j]);
        }
    }
    
    return distance_matrix;
}

std::vector<std::pair<int, int>> AudioSync::find_optimal_alignment(
    const std::vector<std::vector<float>>& distance_matrix
) {
    // Simplified DTW alignment
    std::vector<std::pair<int, int>> alignment;
    
    if (distance_matrix.empty() || distance_matrix[0].empty()) {
        return alignment;
    }
    
    int rows = distance_matrix.size();
    int cols = distance_matrix[0].size();
    
    // Dynamic programming for DTW
    std::vector<std::vector<float>> dtw(rows, std::vector<float>(cols));
    dtw[0][0] = distance_matrix[0][0];
    
    // Initialize first row and column
    for (int i = 1; i < rows; ++i) {
        dtw[i][0] = dtw[i-1][0] + distance_matrix[i][0];
    }
    for (int j = 1; j < cols; ++j) {
        dtw[0][j] = dtw[0][j-1] + distance_matrix[0][j];
    }
    
    // Fill DP matrix
    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            dtw[i][j] = distance_matrix[i][j] + std::min({
                dtw[i-1][j],     // insertion
                dtw[i][j-1],     // deletion
                dtw[i-1][j-1]    // match
            });
        }
    }
    
    // Backtrack to find optimal path
    int i = rows - 1;
    int j = cols - 1;
    
    while (i > 0 || j > 0) {
        alignment.emplace_back(i, j);
        
        if (i == 0) {
            j--;
        } else if (j == 0) {
            i--;
        } else {
            float min_val = std::min({dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]});
            if (min_val == dtw[i-1][j-1]) {
                i--;
                j--;
            } else if (min_val == dtw[i-1][j]) {
                i--;
            } else {
                j--;
            }
        }
    }
    
    alignment.emplace_back(0, 0);
    std::reverse(alignment.begin(), alignment.end());
    
    return alignment;
}

std::vector<float> AudioSync::smooth_correction_curve(const std::vector<float>& curve, float strength) {
    if (curve.size() < 3) {
        return curve;
    }
    
    std::vector<float> smoothed = curve;
    
    // Simple moving average smoothing
    int window = std::max(1, static_cast<int>(curve.size() * 0.1f));
    
    for (size_t i = window; i < curve.size() - window; ++i) {
        float sum = 0.0f;
        for (int j = -window; j <= window; ++j) {
            sum += curve[i + j];
        }
        smoothed[i] = curve[i] * (1.0f - strength) + (sum / (2 * window + 1)) * strength;
    }
    
    return smoothed;
}

std::vector<float> AudioSync::extract_temporal_features(const std::vector<Tensor>& frames) {
    return compute_motion_features(frames);
}

Tensor AudioSync::create_interpolation_matrix(const std::vector<float>& times) {
    // Simplified interpolation matrix creation
    return Tensor::zeros({static_cast<int>(times.size()), static_cast<int>(times.size())});
}

} // namespace audio
} // namespace jin