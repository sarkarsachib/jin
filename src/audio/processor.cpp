#include "processor.h"
#include "../utils/math.h"
#include "../autodiff/tape.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>

namespace jin {
namespace audio {

// WAV file header structure
struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t file_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

AudioProcessor::AudioProcessor(const AudioConfig& config) : config_(config) {
    // Precompute mel filterbank
    mel_filterbank_ = create_mel_filterbank(config_.num_mfcc_coeffs, config_.frame_length);
}

Tensor AudioProcessor::load_audio(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open audio file: " + filepath);
    }
    
    // Read WAV header
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    
    // Validate WAV format
    if (strncmp(header.riff, "RIFF", 4) != 0 || strncmp(header.wave, "WAVE", 4) != 0) {
        throw std::runtime_error("Invalid WAV file format");
    }
    
    // Read audio data
    std::vector<float> audio_data(header.data_size / (header.bits_per_sample / 8));
    
    if (header.bits_per_sample == 16) {
        std::vector<int16_t> int_data(header.data_size / 2);
        file.read(reinterpret_cast<char*>(int_data.data()), header.data_size);
        
        // Convert to float and normalize
        for (size_t i = 0; i < int_data.size(); ++i) {
            audio_data[i] = static_cast<float>(int_data[i]) / 32768.0f;
        }
    } else if (header.bits_per_sample == 32) {
        std::vector<int32_t> int_data(header.data_size / 4);
        file.read(reinterpret_cast<char*>(int_data.data()), header.data_size);
        
        // Convert to float and normalize
        for (size_t i = 0; i < int_data.size(); ++i) {
            audio_data[i] = static_cast<float>(int_data[i]) / 2147483648.0f;
        }
    }
    
    file.close();
    
    // Create tensor from audio data
    Tensor audio_tensor = Tensor::from_vector(audio_data, {static_cast<int>(audio_data.size())});
    
    // Normalize if requested
    if (config_.normalize_audio) {
        audio_tensor = normalize_audio(audio_tensor);
    }
    
    return audio_tensor;
}

Tensor AudioProcessor::load_audio_from_buffer(const std::vector<float>& audio_data, int sample_rate) {
    Tensor audio_tensor = Tensor::from_vector(audio_data, {static_cast<int>(audio_data.size())});
    
    // Update sample rate in config if different
    if (sample_rate != config_.sample_rate) {
        config_.sample_rate = sample_rate;
    }
    
    // Normalize if requested
    if (config_.normalize_audio) {
        audio_tensor = normalize_audio(audio_tensor);
    }
    
    return audio_tensor;
}

Tensor AudioProcessor::extract_mfcc(const Tensor& audio) {
    // Compute STFT
    Tensor stft = compute_stft(audio);
    Tensor magnitude = compute_magnitude_spectrum(stft);
    
    // Apply mel filterbank
    Tensor mel_spectrum = magnitude.matmul(mel_filterbank_.transpose());
    
    // Take log (add small constant to avoid log(0))
    mel_spectrum = (mel_spectrum + 1e-10f).log();
    
    // Apply DCT to get MFCCs
    Tensor mfcc = apply_dct(mel_spectrum, config_.num_mfcc_coeffs);
    
    return mfcc;
}

Tensor AudioProcessor::extract_spectral_features(const Tensor& audio) {
    // Compute STFT
    Tensor stft = compute_stft(audio);
    Tensor magnitude = compute_magnitude_spectrum(stft);
    
    // Compute spectral features: spectral centroid, rolloff, bandwidth
    int num_frames = magnitude.shape()[0];
    int num_freq_bins = magnitude.shape()[1];
    
    // Frequency bins (normalized)
    std::vector<float> freq_bins(num_freq_bins);
    for (int i = 0; i < num_freq_bins; ++i) {
        freq_bins[i] = static_cast<float>(i) / num_freq_bins;
    }
    
    Tensor features = Tensor::zeros({num_frames, 3}); // centroid, rolloff, bandwidth
    
    for (int frame = 0; frame < num_frames; ++frame) {
        Tensor frame_mag = magnitude.get_row(frame);
        
        // Spectral centroid
        float centroid = 0.0f;
        float total_mag = frame_mag.sum().item<float>();
        if (total_mag > 1e-10f) {
            for (int i = 0; i < num_freq_bins; ++i) {
                centroid += freq_bins[i] * frame_mag.item<float>(i);
            }
            centroid /= total_mag;
        }
        
        // Spectral rolloff (85% of energy)
        float rolloff_threshold = 0.85f * total_mag;
        float cumulative = 0.0f;
        float rolloff = 0.0f;
        for (int i = 0; i < num_freq_bins; ++i) {
            cumulative += frame_mag.item<float>(i);
            if (cumulative >= rolloff_threshold) {
                rolloff = freq_bins[i];
                break;
            }
        }
        
        // Spectral bandwidth
        float bandwidth = 0.0f;
        if (total_mag > 1e-10f) {
            for (int i = 0; i < num_freq_bins; ++i) {
                float diff = freq_bins[i] - centroid;
                bandwidth += diff * diff * frame_mag.item<float>(i);
            }
            bandwidth = std::sqrt(bandwidth / total_mag);
        }
        
        features.set_item(centroid, {frame, 0});
        features.set_item(rolloff, {frame, 1});
        features.set_item(bandwidth, {frame, 2});
    }
    
    return features;
}

Tensor AudioProcessor::extract_chroma_features(const Tensor& audio) {
    // Compute STFT
    Tensor stft = compute_stft(audio);
    Tensor magnitude = compute_magnitude_spectrum(stft);
    
    // Chroma features (12 pitch classes)
    int num_frames = magnitude.shape()[0];
    int num_freq_bins = magnitude.shape()[1];
    
    // Map frequency bins to chroma classes
    Tensor chroma = Tensor::zeros({num_frames, 12});
    
    for (int frame = 0; frame < num_frames; ++frame) {
        Tensor frame_mag = magnitude.get_row(frame);
        
        for (int bin = 0; bin < num_freq_bins; ++bin) {
            // Convert bin to pitch class (simplified mapping)
            int pitch_class = (bin * 12 / num_freq_bins) % 12;
            chroma.set_item(
                chroma.item<float>(frame, pitch_class) + frame_mag.item<float>(bin),
                {frame, pitch_class}
            );
        }
    }
    
    // Normalize each frame
    for (int frame = 0; frame < num_frames; ++frame) {
        Tensor chroma_frame = chroma.get_row(frame);
        float norm = chroma_frame.norm().item<float>();
        if (norm > 1e-10f) {
            chroma_frame = chroma_frame / norm;
        }
        chroma.set_row(frame, chroma_frame);
    }
    
    return chroma;
}

Tensor AudioProcessor::extract_energy_features(const Tensor& audio) {
    // Compute STFT
    Tensor stft = compute_stft(audio);
    Tensor power_spectrum = compute_power_spectrum(stft);
    
    // Energy is the sum of power spectrum across frequencies for each frame
    int num_frames = power_spectrum.shape()[0];
    Tensor energy = Tensor::zeros({num_frames, 1});
    
    for (int frame = 0; frame < num_frames; ++frame) {
        Tensor frame_power = power_spectrum.get_row(frame);
        energy.set_item(frame_power.sum().item<float>(), {frame, 0});
    }
    
    return energy;
}

Tensor AudioProcessor::extract_features(const Tensor& audio, FeatureType feature_type) {
    switch (feature_type) {
        case FeatureType::MFCC:
            return extract_mfcc(audio);
        case FeatureType::SPECTRAL:
            return extract_spectral_features(audio);
        case FeatureType::CHROMA:
            return extract_chroma_features(audio);
        case FeatureType::ENERGY:
            return extract_energy_features(audio);
        default:
            throw std::runtime_error("Unknown feature type");
    }
}

std::vector<float> AudioProcessor::detect_beats(const Tensor& audio) {
    // Compute onset strength
    std::vector<float> onset_strength = compute_onset_strength(audio);
    
    // Peak picking to find beats
    float threshold = 0.3f * *std::max_element(onset_strength.begin(), onset_strength.end());
    std::vector<float> beats = peak_picking(onset_strength, threshold);
    
    // Convert to seconds
    float hop_time = static_cast<float>(config_.hop_length) / config_.sample_rate;
    for (float& beat : beats) {
        beat *= hop_time;
    }
    
    return beats;
}

float AudioProcessor::detect_tempo(const Tensor& audio) {
    // Compute onset envelope
    std::vector<float> onset_envelope = compute_onset_envelope(audio);
    
    // Compute autocorrelation for tempo detection
    int min_lag = static_cast<int>(config_.sample_rate * 0.5f / config_.hop_length); // 120 BPM
    int max_lag = static_cast<int>(config_.sample_rate * 2.0f / config_.hop_length); // 30 BPM
    
    float best_corr = -1.0f;
    int best_lag = min_lag;
    
    for (int lag = min_lag; lag <= max_lag; ++lag) {
        float corr = autocorrelation(onset_envelope, lag);
        if (corr > best_corr) {
            best_corr = corr;
            best_lag = lag;
        }
    }
    
    // Convert lag to BPM
    float beat_period = static_cast<float>(best_lag) * config_.hop_length / config_.sample_rate;
    float bpm = 60.0f / beat_period;
    
    // Ensure tempo is in reasonable range
    bpm = std::clamp(bpm, 40.0f, 240.0f);
    
    return bpm;
}

Tensor AudioProcessor::normalize_audio(const Tensor& audio) {
    float max_val = audio.abs().max().item<float>();
    if (max_val > 1e-10f) {
        return audio / max_val;
    }
    return audio;
}

Tensor AudioProcessor::resample_audio(const Tensor& audio, int target_sample_rate) {
    if (target_sample_rate == config_.sample_rate) {
        return audio;
    }
    
    // Simple linear interpolation resampling
    float ratio = static_cast<float>(target_sample_rate) / config_.sample_rate;
    int new_length = static_cast<int>(audio.shape()[0] * ratio);
    
    Tensor resampled = Tensor::zeros({new_length});
    
    for (int i = 0; i < new_length; ++i) {
        float original_pos = i / ratio;
        int left_idx = static_cast<int>(original_pos);
        int right_idx = std::min(left_idx + 1, audio.shape()[0] - 1);
        float alpha = original_pos - left_idx;
        
        float left_val = left_idx >= 0 ? audio.item<float>(left_idx) : 0.0f;
        float right_val = right_idx >= 0 ? audio.item<float>(right_idx) : 0.0f;
        
        resampled.set_item(left_val * (1 - alpha) + right_val * alpha, {i});
    }
    
    return resampled;
}

Tensor AudioProcessor::compute_stft(const Tensor& audio) {
    int window_length = config_.frame_length;
    int hop_length = config_.hop_length;
    int fft_size = window_length;
    
    // Create window
    Tensor window = create_hann_window(window_length);
    
    // Pad audio
    int num_frames = (audio.shape()[0] + window_length - 1) / hop_length + 1;
    int padded_length = (num_frames - 1) * hop_length + window_length;
    
    Tensor padded_audio = Tensor::zeros({padded_length});
    padded_audio.get_slice(0, audio.shape()[0]) = audio;
    
    // Compute STFT
    Tensor stft = Tensor::zeros({num_frames, fft_size / 2 + 1});
    
    for (int frame = 0; frame < num_frames; ++frame) {
        int start = frame * hop_length;
        Tensor windowed = padded_audio.get_slice(start, start + window_length) * window;
        
        // FFT (simplified - use magnitude only)
        for (int i = 0; i < fft_size / 2 + 1; ++i) {
            float real = 0.0f, imag = 0.0f;
            for (int n = 0; n < window_length; ++n) {
                float angle = -2.0f * M_PI * i * n / fft_size;
                real += windowed.item<float>(n) * std::cos(angle);
                imag += windowed.item<float>(n) * std::sin(angle);
            }
            stft.set_item(std::sqrt(real * real + imag * imag), {frame, i});
        }
    }
    
    return stft;
}

Tensor AudioProcessor::compute_magnitude_spectrum(const Tensor& stft) {
    return stft; // Already computed magnitude in STFT
}

Tensor AudioProcessor::compute_power_spectrum(const Tensor& magnitude) {
    return magnitude * magnitude;
}

Tensor AudioProcessor::create_mel_filterbank(int num_filters, int fft_size) {
    int num_fft_bins = fft_size / 2 + 1;
    int num_mel_bins = num_filters;
    
    Tensor filterbank = Tensor::zeros({num_fft_bins, num_mel_bins});
    
    // Convert to mel scale
    auto hz_to_mel = [](float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    };
    
    auto mel_to_hz = [](float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    };
    
    // Create mel scale points
    float nyquist = config_.sample_rate / 2.0f;
    float min_hz = 0.0f;
    float max_hz = nyquist;
    
    std::vector<float> mel_points(num_mel_bins + 2);
    for (int i = 0; i < num_mel_bins + 2; ++i) {
        float mel = hz_to_mel(min_hz) + (hz_to_mel(max_hz) - hz_to_mel(min_hz)) * i / (num_mel_bins + 1);
        mel_points[i] = mel_to_hz(mel);
    }
    
    // Convert to FFT bins
    std::vector<int> bin_points(num_mel_bins + 2);
    for (int i = 0; i < num_mel_bins + 2; ++i) {
        bin_points[i] = static_cast<int>(std::floor((fft_size + 1) * mel_points[i] / nyquist));
    }
    
    // Create triangular filters
    for (int i = 1; i <= num_mel_bins; ++i) {
        int left = bin_points[i - 1];
        int center = bin_points[i];
        int right = bin_points[i + 1];
        
        // Left slope
        for (int j = left; j < center; ++j) {
            filterbank.set_item(
                static_cast<float>(j - left) / (center - left),
                {j, i - 1}
            );
        }
        
        // Right slope
        for (int j = center; j < right; ++j) {
            filterbank.set_item(
                static_cast<float>(right - j) / (right - center),
                {j, i - 1}
            );
        }
    }
    
    return filterbank;
}

Tensor AudioProcessor::create_hann_window(int window_length) {
    Tensor window = Tensor::zeros({window_length});
    
    for (int i = 0; i < window_length; ++i) {
        window.set_item(
            0.5f * (1.0f - std::cos(2.0f * M_PI * i / (window_length - 1))),
            {i}
        );
    }
    
    return window;
}

Tensor AudioProcessor::apply_dct(const Tensor& input, int num_coeffs) {
    int num_frames = input.shape()[0];
    int num_bins = input.shape()[1];
    
    Tensor dct = Tensor::zeros({num_frames, num_coeffs});
    
    for (int frame = 0; frame < num_frames; ++frame) {
        for (int k = 0; k < num_coeffs; ++k) {
            float sum = 0.0f;
            for (int n = 0; n < num_bins; ++n) {
                sum += input.item<float>(frame, n) * 
                       std::cos(M_PI * k * (2 * n + 1) / (2 * num_bins));
            }
            dct.set_item(sum, {frame, k});
        }
    }
    
    return dct;
}

std::vector<float> AudioProcessor::compute_onset_strength(const Tensor& audio) {
    // Compute STFT
    Tensor stft = compute_stft(audio);
    Tensor magnitude = compute_magnitude_spectrum(stft);
    
    // Compute frame-to-frame differences
    int num_frames = magnitude.shape()[0];
    int num_bins = magnitude.shape()[1];
    
    std::vector<float> onset_strength(num_frames, 0.0f);
    
    for (int frame = 1; frame < num_frames; ++frame) {
        float diff_sum = 0.0f;
        for (int bin = 0; bin < num_bins; ++bin) {
            float prev = magnitude.item<float>(frame - 1, bin);
            float curr = magnitude.item<float>(frame, bin);
            diff_sum += std::max(0.0f, curr - prev);
        }
        onset_strength[frame] = diff_sum;
    }
    
    return onset_strength;
}

std::vector<float> AudioProcessor::peak_picking(const std::vector<float>& signal, float threshold) {
    std::vector<float> peaks;
    int window = 3; // Minimum distance between peaks
    
    for (size_t i = window; i < signal.size() - window; ++i) {
        bool is_peak = signal[i] > threshold;
        
        // Check local maximum
        for (size_t j = i - window; j <= i + window; ++j) {
            if (j != i && signal[j] >= signal[i]) {
                is_peak = false;
                break;
            }
        }
        
        if (is_peak) {
            peaks.push_back(static_cast<float>(i));
        }
    }
    
    return peaks;
}

std::vector<float> AudioProcessor::compute_onset_envelope(const Tensor& audio) {
    return compute_onset_strength(audio);
}

float AudioProcessor::autocorrelation(const std::vector<float>& signal, int lag) {
    float sum = 0.0f;
    for (size_t i = 0; i < signal.size() - lag; ++i) {
        sum += signal[i] * signal[i + lag];
    }
    return sum;
}

int AudioProcessor::get_audio_length_samples(const Tensor& audio) const {
    return audio.shape()[0];
}

float AudioProcessor::get_audio_length_seconds(const Tensor& audio) const {
    return static_cast<float>(audio.shape()[0]) / config_.sample_rate;
}

} // namespace audio
} // namespace jin