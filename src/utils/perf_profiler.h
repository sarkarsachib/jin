#pragma once

#include "../gpu/gpu_common.h"
#include <chrono>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace jin {
namespace utils {

// GPU performance profiler
class GpuProfiler {
public:
    struct ProfilerEntry {
        std::string kernel_name;
        std::string device_type; // "cuda", "vulkan", "metal"
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        size_t bytes_transferred = 0;
        size_t flops_executed = 0;
        double bandwidth_utilization = 0.0; // %
        double compute_utilization = 0.0; // %
        bool is_active = false;
    };
    
    struct ProfilerStats {
        size_t total_kernels_executed = 0;
        double total_execution_time = 0.0;
        size_t total_bytes_transferred = 0;
        double peak_bandwidth = 0.0;
        double average_kernel_time = 0.0;
        std::unordered_map<std::string, size_t> kernel_counts;
        std::unordered_map<std::string, double> kernel_times;
        std::vector<std::string> bottlenecks; // "compute_bound", "memory_bound", etc.
    };
    
    GpuProfiler();
    ~GpuProfiler();
    
    // Profiling control
    void start_profiling(const std::string& kernel_name, const std::string& device_type);
    void end_profiling();
    
    // Memory transfer profiling
    void start_memory_transfer(const std::string& transfer_name, size_t bytes, const std::string& device_type);
    void end_memory_transfer();
    
    // Statistics and reporting
    ProfilerStats get_stats() const;
    void print_report() const;
    void reset_stats();
    
    // Performance analysis
    std::vector<std::string> identify_bottlenecks() const;
    double get_achieved_bandwidth() const;
    double get_peak_performance() const;
    
    // Export functionality
    void export_to_csv(const std::string& filename) const;
    void export_to_json(const std::string& filename) const;
    
    // Configuration
    void enable_detailed_profiling(bool enable) { detailed_profiling_ = enable; }
    void set_profiling_output(const std::string& output_file) { output_file_ = output_file; }
    
private:
    void calculate_utilization(ProfilerEntry& entry);
    void update_stats(const ProfilerEntry& entry);
    
    std::vector<ProfilerEntry> active_entries_;
    std::vector<ProfilerEntry> completed_entries_;
    ProfilerStats stats_;
    bool detailed_profiling_ = false;
    std::string output_file_;
};

// Simple GPU timer utility
class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();
    
    void start();
    void stop();
    double elapsed_ms() const;
    double elapsed_us() const;
    
    // Context-specific timers
    void start_cuda_timer();
    void stop_cuda_timer();
    void start_vulkan_timer();
    void stop_vulkan_timer();
    void start_metal_timer();
    void stop_metal_timer();
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool is_running_ = false;
    
    // Backend-specific timing
    void* cuda_start_event_ = nullptr;
    void* cuda_stop_event_ = nullptr;
    void* vulkan_start_query_ = nullptr;
    void* vulkan_stop_query_ = nullptr;
    uint64_t metal_start_timestamp_ = 0;
    uint64_t metal_stop_timestamp_ = 0;
};

// Memory bandwidth profiler
class BandwidthProfiler {
public:
    struct BandwidthEntry {
        std::string operation_name;
        std::string device_type;
        size_t bytes_transferred;
        double transfer_time_ms;
        double bandwidth_gb_s;
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    void record_transfer(const std::string& operation, const std::string& device_type, 
                        size_t bytes, double time_ms);
    
    std::vector<BandwidthEntry> get_transfer_history() const { return transfer_history_; }
    double get_peak_bandwidth() const;
    double get_average_bandwidth() const;
    std::string get_fastest_device() const;
    
    void print_bandwidth_report() const;
    void reset_history();
    
private:
    std::vector<BandwidthEntry> transfer_history_;
};

// GPU utilization monitor
class GpuUtilizationMonitor {
public:
    struct UtilizationEntry {
        std::string device_name;
        double compute_utilization; // 0.0 - 1.0
        double memory_utilization; // 0.0 - 1.0
        double memory_used_gb;
        double memory_total_gb;
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    void record_utilization(const std::string& device_name, double compute_util, 
                           double memory_util, size_t memory_used, size_t memory_total);
    
    std::vector<UtilizationEntry> get_utilization_history() const { return utilization_history_; }
    double get_average_compute_utilization() const;
    double get_average_memory_utilization() const;
    
    void print_utilization_report() const;
    void reset_history();
    
private:
    std::vector<UtilizationEntry> utilization_history_;
};

// Benchmarking utilities
class GpuBenchmark {
public:
    struct BenchmarkResult {
        std::string operation_name;
        std::string device_type;
        double execution_time_ms;
        double throughput; // ops/sec or GB/s
        double bandwidth_gb_s;
        std::vector<double> execution_times; // for statistical analysis
    };
    
    // Element-wise operations benchmark
    BenchmarkResult benchmark_elementwise_ops(const std::string& device_type, size_t data_size);
    
    // Matrix multiplication benchmark
    BenchmarkResult benchmark_matmul(const std::string& device_type, int m, int n, int k);
    
    // Memory bandwidth benchmark
    BenchmarkResult benchmark_memory_bandwidth(const std::string& device_type, size_t data_size);
    
    // Reduction operations benchmark
    BenchmarkResult benchmark_reduction(const std::string& device_type, size_t data_size);
    
    // Comprehensive benchmark suite
    void run_full_benchmark_suite();
    std::vector<BenchmarkResult> get_benchmark_results() const { return benchmark_results_; }
    
    void print_benchmark_report() const;
    void export_benchmark_results(const std::string& filename) const;
    
private:
    std::vector<BenchmarkResult> benchmark_results_;
    
    // Helper methods
    double measure_execution_time(std::function<void()> operation, int num_runs = 10);
    void warmup_gpu();
};

// Performance optimization suggestions
class PerformanceOptimizer {
public:
    struct OptimizationSuggestion {
        std::string category; // "memory", "compute", "kernel_fusion"
        std::string suggestion;
        double estimated_improvement; // percentage
        std::string implementation_effort; // "low", "medium", "high"
        bool is_applicable;
    };
    
    std::vector<OptimizationSuggestion> analyze_performance(const GpuProfiler::ProfilerStats& stats);
    std::vector<OptimizationSuggestion> suggest_memory_optimizations(const GpuProfiler::ProfilerStats& stats);
    std::vector<OptimizationSuggestion> suggest_compute_optimizations(const GpuProfiler::ProfilerStats& stats);
    std::vector<OptimizationSuggestion> suggest_fusion_opportunities(const GpuProfiler::ProfilerStats& stats);
    
    void print_optimization_report(const std::vector<OptimizationSuggestion>& suggestions) const;
    
private:
    // Analysis heuristics
    bool is_memory_bound(const GpuProfiler::ProfilerStats& stats) const;
    bool is_compute_bound(const GpuProfiler::ProfilerStats& stats) const;
    double calculate_potential_speedup(const std::string& optimization_type) const;
};

// Global profiler instance
GpuProfiler& get_global_gpu_profiler();

} // namespace utils
} // namespace jin