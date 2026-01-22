#pragma once

#include "../core/tensor.h"
#include <string>
#include <vector>
#include <map>

namespace jin {
namespace audio {

// Forward declarations
class AudioProcessor;

// Audio feature types
enum class FeatureType {
    MFCC,           // Mel-frequency cepstral coefficients
    SPECTRAL,       // Spectral magnitude features
    CHROMA,         // Chromatic features
    ENERGY          // Energy features
};

// Audio configuration structure
struct AudioConfig {
    int sample_rate;                    // Audio sample rate (44100 or 48000)
    int frame_length;                   // Frame size for feature extraction (512, 1024, 2048)
    int hop_length;                     // Hop length between frames (frame_length / 4)
    int num_mfcc_coeffs;               // Number of MFCC coefficients (13-40)
    bool normalize_audio;               // Whether to normalize audio to [-1, 1]
    
    AudioConfig() 
        : sample_rate(44100), frame_length(1024), hop_length(256), 
          num_mfcc_coeffs(13), normalize_audio(true) {}
};

class AudioProcessor {
public:
    AudioProcessor(const AudioConfig& config = AudioConfig{});
    
    // Load and preprocess audio
    Tensor load_audio(const std::string& filepath);
    Tensor load_audio_from_buffer(const std::vector<float>& audio_data, int sample_rate);
    
    // Feature extraction
    Tensor extract_features(const Tensor& audio, FeatureType feature_type);
    Tensor extract_mfcc(const Tensor& audio);
    Tensor extract_spectral_features(const Tensor& audio);
    Tensor extract_chroma_features(const Tensor& audio);
    Tensor extract_energy_features(const Tensor& audio);
    
    // Audio analysis
    std::vector<float> detect_beats(const Tensor& audio);
    float detect_tempo(const Tensor& audio);
    
    // Audio preprocessing
    Tensor normalize_audio(const Tensor& audio);
    Tensor resample_audio(const Tensor& audio, int target_sample_rate);
    Tensor apply_window_function(const Tensor& audio);
    
    // Audio information
    int get_sample_rate() const { return config_.sample_rate; }
    int get_audio_length_samples(const Tensor& audio) const;
    float get_audio_length_seconds(const Tensor& audio) const;
    
    // Utility methods
    void set_config(const AudioConfig& config) { config_ = config; }
    const AudioConfig& get_config() const { return config_; }
    
    // Create mel filterbank for MFCC
    Tensor create_mel_filterbank(int num_filters, int fft_size);
    
    // Apply discrete cosine transform for MFCC
    Tensor apply_dct(const Tensor& input, int num_coeffs);
    
private:
    AudioConfig config_;
    
    // Precomputed mel filterbank
    Tensor mel_filterbank_;
    
    // Helper methods
    Tensor compute_stft(const Tensor& audio);
    Tensor compute_magnitude_spectrum(const Tensor& stft);
    Tensor compute_power_spectrum(const Tensor& magnitude);
    
    // Window function for STFT
    Tensor create_hann_window(int window_length);
    
    // Beat detection helpers
    std::vector<float> compute_onset_strength(const Tensor& audio);
    std::vector<float> peak_picking(const std::vector<float>& signal, float threshold);
    
    // Tempo detection helpers
    float autocorrelation(const std::vector<float>& signal, int lag);
    std::vector<float> compute_onset_envelope(const Tensor& audio);
};

} // namespace audio
} // namespace jin