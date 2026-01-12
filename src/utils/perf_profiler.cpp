#include "perf_profiler.h"
#include "../gpu/gpu_common.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <chrono>

namespace jin {
namespace utils {

// Global GPU profiler instance
GpuProfiler& get_global_gpu_profiler() {
    static GpuProfiler instance;
    return instance;
}

// GpuProfiler implementation
GpuProfiler::GpuProfiler() = default;

GpuProfiler::~GpuProfiler() = default;

void GpuProfiler::start_profiling(const std::string& kernel_name, const std::string& device_type) {
    ProfilerEntry entry;
    entry.kernel_name = kernel_name;
    entry.device_type = device_type;
    entry.start_time = std::chrono::high_resolution_clock::now();
    entry.is_active = true;
    
    active_entries_.push_back(entry);
}

void GpuProfiler::end_profiling() {
    if (active_entries_.empty()) {
        return;
    }
    
    auto& entry = active_entries_.back();
    entry.end_time = std::chrono::high_resolution_clock::now();
    entry.is_active = false;
    
    // Calculate metrics
    calculate_utilization(entry);
    update_stats(entry);
    
    completed_entries_.push_back(entry);
    active_entries_.pop_back();
}

void GpuProfiler::start_memory_transfer(const std::string& transfer_name, size_t bytes, const std::string& device_type) {
    start_profiling("memory_transfer_" + transfer_name, device_type);
    if (!active_entries_.empty()) {
        active_entries_.back().bytes_transferred = bytes;
    }
}

void GpuProfiler::end_memory_transfer() {
    end_profiling();
}

GpuProfiler::ProfilerStats GpuProfiler::get_stats() const {
    return stats_;
}

void GpuProfiler::print_report() const {
    std::cout << "\n=== GPU Performance Report ===\n";
    std::cout << "Total kernels executed: " << stats_.total_kernels_executed << "\n";
    std::cout << "Total execution time: " << stats_.total_execution_time << " ms\n";
    std::cout << "Total bytes transferred: " << stats_.total_bytes_transferred << " bytes\n";
    std::cout << "Peak bandwidth: " << stats_.peak_bandwidth << " GB/s\n";
    std::cout << "Average kernel time: " << stats_.average_kernel_time << " ms\n";
    
    if (!stats_.kernel_counts.empty()) {
        std::cout << "\nKernel execution counts:\n";
        for (const auto& pair : stats_.kernel_counts) {
            std::cout << "  " << pair.first << ": " << pair.second << " times\n";
        }
    }
    
    if (!stats_.bottlenecks.empty()) {
        std::cout << "\nIdentified bottlenecks:\n";
        for (const auto& bottleneck : stats_.bottlenecks) {
            std::cout << "  - " << bottleneck << "\n";
        }
    }
    
    std::cout << "\n";
}

void GpuProfiler::reset_stats() {
    stats_ = ProfilerStats{};
    completed_entries_.clear();
    active_entries_.clear();
}

std::vector<std::string> GpuProfiler::identify_bottlenecks() const {
    std::vector<std::string> bottlenecks;
    
    if (stats_.total_kernels_executed == 0) {
        return bottlenecks;
    }
    
    // Check if memory-bound
    double avg_bandwidth_utilization = 0.0;
    if (!completed_entries_.empty()) {
        double total_bandwidth = 0.0;
        for (const auto& entry : completed_entries_) {
            total_bandwidth += entry.bandwidth_utilization;
        }
        avg_bandwidth_utilization = total_bandwidth / completed_entries_.size();
    }
    
    if (avg_bandwidth_utilization > 0.8) {
        bottlenecks.push_back("memory_bound");
    }
    
    // Check if compute-bound
    double avg_compute_utilization = 0.0;
    if (!completed_entries_.empty()) {
        double total_compute = 0.0;
        for (const auto& entry : completed_entries_) {
            total_compute += entry.compute_utilization;
        }
        avg_compute_utilization = total_compute / completed_entries_.size();
    }
    
    if (avg_compute_utilization > 0.8) {
        bottlenecks.push_back("compute_bound");
    }
    
    // Check for small kernels (launch overhead dominant)
    if (stats_.average_kernel_time < 0.1) { // Less than 0.1ms
        bottlenecks.push_back("kernel_launch_overhead");
    }
    
    return bottlenecks;
}

double GpuProfiler::get_achieved_bandwidth() const {
    if (stats_.total_execution_time <= 0) {
        return 0.0;
    }
    
    double total_bytes = static_cast<double>(stats_.total_bytes_transferred);
    double total_time_s = stats_.total_execution_time / 1000.0; // Convert ms to s
    double bandwidth_gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / total_time_s;
    
    return bandwidth_gb_s;
}

double GpuProfiler::get_peak_performance() const {
    // Simplified: assume peak performance based on kernel counts and times
    // In practice, this would be based on device specifications
    return 1.0; // Placeholder
}

void GpuProfiler::export_to_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for CSV export: " << filename << "\n";
        return;
    }
    
    // Write header
    file << "kernel_name,device_type,start_time,end_time,execution_time_ms,bytes_transferred,bandwidth_utilization,compute_utilization\n";
    
    // Write data
    for (const auto& entry : completed_entries_) {
        auto start_time_t = std::chrono::duration_cast<std::chrono::microseconds>(
            entry.start_time.time_since_epoch()).count();
        auto end_time_t = std::chrono::duration_cast<std::chrono::microseconds>(
            entry.end_time.time_since_epoch()).count();
        
        auto duration_ms = std::chrono::duration<double, std::milli>(entry.end_time - entry.start_time).count();
        
        file << entry.kernel_name << ","
             << entry.device_type << ","
             << start_time_t << ","
             << end_time_t << ","
             << duration_ms << ","
             << entry.bytes_transferred << ","
             << entry.bandwidth_utilization << ","
             << entry.compute_utilization << "\n";
    }
    
    file.close();
}

void GpuProfiler::export_to_json(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for JSON export: " << filename << "\n";
        return;
    }
    
    file << "{\n";
    file << "  \"total_kernels_executed\": " << stats_.total_kernels_executed << ",\n";
    file << "  \"total_execution_time\": " << stats_.total_execution_time << ",\n";
    file << "  \"total_bytes_transferred\": " << stats_.total_bytes_transferred << ",\n";
    file << "  \"peak_bandwidth\": " << stats_.peak_bandwidth << ",\n";
    file << "  \"average_kernel_time\": " << stats_.average_kernel_time << ",\n";
    
    file << "  \"kernel_counts\": {";
    bool first = true;
    for (const auto& pair : stats_.kernel_counts) {
        if (!first) file << ",";
        file << "\"" << pair.first << "\": " << pair.second;
        first = false;
    }
    file << "},\n";
    
    file << "  \"bottlenecks\": [";
    first = true;
    for (const auto& bottleneck : stats_.bottlenecks) {
        if (!first) file << ",";
        file << "\"" << bottleneck << "\"";
        first = false;
    }
    file << "]\n";
    
    file << "}\n";
    file.close();
}

void GpuProfiler::calculate_utilization(ProfilerEntry& entry) {
    auto duration_ms = std::chrono::duration<double, std::milli>(entry.end_time - entry.start_time).count();
    
    if (entry.bytes_transferred > 0 && duration_ms > 0) {
        // Calculate bandwidth utilization (simplified)
        double bandwidth_gb_s = (entry.bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (duration_ms / 1000.0);
        
        // Assume theoretical peak bandwidth (this should come from device capabilities)
        double peak_bandwidth = 500.0; // GB/s (typical for modern GPUs)
        entry.bandwidth_utilization = std::min(1.0, bandwidth_gb_s / peak_bandwidth);
    }
    
    // Simplified compute utilization calculation
    entry.compute_utilization = 0.8; // Placeholder
}

void GpuProfiler::update_stats(const ProfilerEntry& entry) {
    stats_.total_kernels_executed++;
    
    auto duration_ms = std::chrono::duration<double, std::milli>(entry.end_time - entry.start_time).count();
    stats_.total_execution_time += duration_ms;
    stats_.total_bytes_transferred += entry.bytes_transferred;
    
    stats_.average_kernel_time = stats_.total_execution_time / stats_.total_kernels_executed;
    
    // Update peak bandwidth
    if (entry.bytes_transferred > 0 && duration_ms > 0) {
        double bandwidth_gb_s = (entry.bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / (duration_ms / 1000.0);
        stats_.peak_bandwidth = std::max(stats_.peak_bandwidth, bandwidth_gb_s);
    }
    
    // Update kernel counts and times
    stats_.kernel_counts[entry.kernel_name]++;
    stats_.kernel_times[entry.kernel_name] += duration_ms;
    
    // Update bottlenecks
    auto bottlenecks = identify_bottlenecks();
    for (const auto& bottleneck : bottlenecks) {
        if (std::find(stats_.bottlenecks.begin(), stats_.bottlenecks.end(), bottleneck) == stats_.bottlenecks.end()) {
            stats_.bottlenecks.push_back(bottleneck);
        }
    }
}

// GpuTimer implementation
GpuTimer::GpuTimer() = default;

GpuTimer::~GpuTimer() = default;

void GpuTimer::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = true;
}

void GpuTimer::stop() {
    if (!is_running_) return;
    
    end_time_ = std::chrono::high_resolution_clock::now();
    is_running_ = false;
}

double GpuTimer::elapsed_ms() const {
    if (!is_running_) {
        return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_time_).count();
}

double GpuTimer::elapsed_us() const {
    if (!is_running_) {
        return std::chrono::duration<double, std::micro>(end_time_ - start_time_).count();
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(now - start_time_).count();
}

void GpuTimer::start_cuda_timer() {
    // CUDA event-based timing would be implemented here
    start();
}

void GpuTimer::stop_cuda_timer() {
    // CUDA event-based timing would be implemented here
    stop();
}

void GpuTimer::start_vulkan_timer() {
    // Vulkan query-based timing would be implemented here
    start();
}

void GpuTimer::stop_vulkan_timer() {
    // Vulkan query-based timing would be implemented here
    stop();
}

void GpuTimer::start_metal_timer() {
    // Metal timestamp-based timing would be implemented here
    start();
}

void GpuTimer::stop_metal_timer() {
    // Metal timestamp-based timing would be implemented here
    stop();
}

// BandwidthProfiler implementation
void BandwidthProfiler::record_transfer(const std::string& operation, const std::string& device_type,
                                       size_t bytes, double time_ms) {
    BandwidthEntry entry;
    entry.operation_name = operation;
    entry.device_type = device_type;
    entry.bytes_transferred = bytes;
    entry.transfer_time_ms = time_ms;
    entry.bandwidth_gb_s = (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
    entry.timestamp = std::chrono::high_resolution_clock::now();
    
    transfer_history_.push_back(entry);
}

double BandwidthProfiler::get_peak_bandwidth() const {
    if (transfer_history_.empty()) {
        return 0.0;
    }
    
    double peak = 0.0;
    for (const auto& entry : transfer_history_) {
        peak = std::max(peak, entry.bandwidth_gb_s);
    }
    
    return peak;
}

double BandwidthProfiler::get_average_bandwidth() const {
    if (transfer_history_.empty()) {
        return 0.0;
    }
    
    double total_bandwidth = 0.0;
    for (const auto& entry : transfer_history_) {
        total_bandwidth += entry.bandwidth_gb_s;
    }
    
    return total_bandwidth / transfer_history_.size();
}

std::string BandwidthProfiler::get_fastest_device() const {
    if (transfer_history_.empty()) {
        return "None";
    }
    
    std::unordered_map<std::string, double> device_bandwidths;
    std::unordered_map<std::string, size_t> device_counts;
    
    for (const auto& entry : transfer_history_) {
        device_bandwidths[entry.device_type] += entry.bandwidth_gb_s;
        device_counts[entry.device_type]++;
    }
    
    std::string fastest_device = "None";
    double fastest_bandwidth = 0.0;
    
    for (const auto& pair : device_bandwidths) {
        double avg_bandwidth = pair.second / device_counts[pair.first];
        if (avg_bandwidth > fastest_bandwidth) {
            fastest_bandwidth = avg_bandwidth;
            fastest_device = pair.first;
        }
    }
    
    return fastest_device;
}

void BandwidthProfiler::print_bandwidth_report() const {
    std::cout << "\n=== Bandwidth Report ===\n";
    std::cout << "Total transfers recorded: " << transfer_history_.size() << "\n";
    std::cout << "Peak bandwidth: " << get_peak_bandwidth() << " GB/s\n";
    std::cout << "Average bandwidth: " << get_average_bandwidth() << " GB/s\n";
    std::cout << "Fastest device: " << get_fastest_device() << "\n";
    std::cout << "\n";
}

void BandwidthProfiler::reset_history() {
    transfer_history_.clear();
}

// GpuUtilizationMonitor implementation
void GpuUtilizationMonitor::record_utilization(const std::string& device_name, double compute_util,
                                              double memory_util, size_t memory_used, size_t memory_total) {
    UtilizationEntry entry;
    entry.device_name = device_name;
    entry.compute_utilization = compute_util;
    entry.memory_utilization = memory_util;
    entry.memory_used_gb = memory_used / (1024.0 * 1024.0 * 1024.0);
    entry.memory_total_gb = memory_total / (1024.0 * 1024.0 * 1024.0);
    entry.timestamp = std::chrono::high_resolution_clock::now();
    
    utilization_history_.push_back(entry);
}

double GpuUtilizationMonitor::get_average_compute_utilization() const {
    if (utilization_history_.empty()) {
        return 0.0;
    }
    
    double total = 0.0;
    for (const auto& entry : utilization_history_) {
        total += entry.compute_utilization;
    }
    
    return total / utilization_history_.size();
}

double GpuUtilizationMonitor::get_average_memory_utilization() const {
    if (utilization_history_.empty()) {
        return 0.0;
    }
    
    double total = 0.0;
    for (const auto& entry : utilization_history_) {
        total += entry.memory_utilization;
    }
    
    return total / utilization_history_.size();
}

void GpuUtilizationMonitor::print_utilization_report() const {
    std::cout << "\n=== GPU Utilization Report ===\n";
    std::cout << "Total measurements: " << utilization_history_.size() << "\n";
    std::cout << "Average compute utilization: " << (get_average_compute_utilization() * 100.0) << "%\n";
    std::cout << "Average memory utilization: " << (get_average_memory_utilization() * 100.0) << "%\n";
    std::cout << "\n";
}

void GpuUtilizationMonitor::reset_history() {
    utilization_history_.clear();
}

// GpuBenchmark implementation
GpuBenchmark::BenchmarkResult GpuBenchmark::benchmark_elementwise_ops(const std::string& device_type, size_t data_size) {
    BenchmarkResult result;
    result.operation_name = "elementwise_ops";
    result.device_type = device_type;
    
    // Simulate benchmarking
    result.execution_time_ms = measure_execution_time([&]() {
        // This would run actual element-wise operations
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    
    // Calculate throughput
    result.throughput = data_size / (result.execution_time_ms / 1000.0);
    result.bandwidth_gb_s = (data_size * sizeof(float)) / (1024.0 * 1024.0 * 1024.0) / (result.execution_time_ms / 1000.0);
    
    benchmark_results_.push_back(result);
    return result;
}

GpuBenchmark::BenchmarkResult GpuBenchmark::benchmark_matmul(const std::string& device_type, int m, int n, int k) {
    BenchmarkResult result;
    result.operation_name = "matmul_" + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    result.device_type = device_type;
    
    // Simulate benchmarking
    result.execution_time_ms = measure_execution_time([&]() {
        // This would run actual matrix multiplication
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
    
    // Calculate FLOPS
    double flops = 2.0 * m * n * k; // Approximate FLOPs for matrix multiplication
    result.throughput = flops / (result.execution_time_ms / 1000.0) / 1e9; // GFLOPS
    
    benchmark_results_.push_back(result);
    return result;
}

GpuBenchmark::BenchmarkResult GpuBenchmark::benchmark_memory_bandwidth(const std::string& device_type, size_t data_size) {
    BenchmarkResult result;
    result.operation_name = "memory_bandwidth";
    result.device_type = device_type;
    
    // Simulate benchmarking
    result.execution_time_ms = measure_execution_time([&]() {
        // This would run actual memory transfers
        std::vector<float> data(data_size / sizeof(float));
        std::fill(data.begin(), data.end(), 1.0f);
        volatile float sum = 0.0f;
        for (const auto& val : data) {
            sum += val;
        }
    });
    
    result.bandwidth_gb_s = data_size / (1024.0 * 1024.0 * 1024.0) / (result.execution_time_ms / 1000.0);
    result.throughput = result.bandwidth_gb_s;
    
    benchmark_results_.push_back(result);
    return result;
}

GpuBenchmark::BenchmarkResult GpuBenchmark::benchmark_reduction(const std::string& device_type, size_t data_size) {
    BenchmarkResult result;
    result.operation_name = "reduction";
    result.device_type = device_type;
    
    // Simulate benchmarking
    result.execution_time_ms = measure_execution_time([&]() {
        // This would run actual reduction operations
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    
    // Calculate throughput
    result.throughput = data_size / (result.execution_time_ms / 1000.0);
    
    benchmark_results_.push_back(result);
    return result;
}

void GpuBenchmark::run_full_benchmark_suite() {
    warmup_gpu();
    
    // Run comprehensive benchmarks
    benchmark_elementwise_ops("cuda", 1024 * 1024 * 100); // 100MB
    benchmark_matmul("cuda", 1024, 1024, 1024);
    benchmark_memory_bandwidth("cuda", 1024 * 1024 * 100); // 100MB
    benchmark_reduction("cuda", 1024 * 1024 * 100); // 100MB
}

void GpuBenchmark::print_benchmark_report() const {
    std::cout << "\n=== GPU Benchmark Report ===\n";
    
    for (const auto& result : benchmark_results_) {
        std::cout << "\n" << result.operation_name << " (" << result.device_type << "):\n";
        std::cout << "  Execution time: " << result.execution_time_ms << " ms\n";
        std::cout << "  Throughput: " << result.throughput;
        
        if (result.operation_name.find("matmul") != std::string::npos) {
            std::cout << " GFLOPS\n";
        } else if (result.operation_name.find("memory") != std::string::npos) {
            std::cout << " GB/s\n";
        } else {
            std::cout << " ops/sec\n";
        }
        
        std::cout << "  Bandwidth: " << result.bandwidth_gb_s << " GB/s\n";
    }
    
    std::cout << "\n";
}

void GpuBenchmark::export_benchmark_results(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for benchmark export: " << filename << "\n";
        return;
    }
    
    file << "operation_name,device_type,execution_time_ms,throughput,bandwidth_gb_s\n";
    
    for (const auto& result : benchmark_results_) {
        file << result.operation_name << ","
             << result.device_type << ","
             << result.execution_time_ms << ","
             << result.throughput << ","
             << result.bandwidth_gb_s << "\n";
    }
    
    file.close();
}

double GpuBenchmark::measure_execution_time(std::function<void()> operation, int num_runs) {
    std::vector<double> times;
    
    for (int i = 0; i < num_runs; i++) {
        GpuTimer timer;
        timer.start();
        operation();
        timer.stop();
        times.push_back(timer.elapsed_ms());
    }
    
    // Return median time (more robust than average)
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

void GpuBenchmark::warmup_gpu() {
    // Warm up GPU with dummy operations
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// PerformanceOptimizer implementation
std::vector<PerformanceOptimizer::OptimizationSuggestion> PerformanceOptimizer::analyze_performance(const GpuProfiler::ProfilerStats& stats) {
    std::vector<OptimizationSuggestion> suggestions;
    
    auto memory_suggestions = suggest_memory_optimizations(stats);
    auto compute_suggestions = suggest_compute_optimizations(stats);
    auto fusion_suggestions = suggest_fusion_opportunities(stats);
    
    suggestions.insert(suggestions.end(), memory_suggestions.begin(), memory_suggestions.end());
    suggestions.insert(suggestions.end(), compute_suggestions.begin(), compute_suggestions.end());
    suggestions.insert(suggestions.end(), fusion_suggestions.begin(), fusion_suggestions.end());
    
    return suggestions;
}

std::vector<PerformanceOptimizer::OptimizationSuggestion> PerformanceOptimizer::suggest_memory_optimizations(const GpuProfiler::ProfilerStats& stats) {
    std::vector<OptimizationSuggestion> suggestions;
    
    if (is_memory_bound(stats)) {
        OptimizationSuggestion suggestion1;
        suggestion1.category = "memory";
        suggestion1.suggestion = "Consider using memory coalescing to improve global memory access patterns";
        suggestion1.estimated_improvement = 15.0;
        suggestion1.implementation_effort = "medium";
        suggestion1.is_applicable = true;
        suggestions.push_back(suggestion1);
        
        OptimizationSuggestion suggestion2;
        suggestion2.category = "memory";
        suggestion2.suggestion = "Use shared memory to cache frequently accessed data";
        suggestion2.estimated_improvement = 25.0;
        suggestion2.implementation_effort = "high";
        suggestion2.is_applicable = true;
        suggestions.push_back(suggestion2);
        
        OptimizationSuggestion suggestion3;
        suggestion3.category = "memory";
        suggestion3.suggestion = "Optimize memory access patterns to reduce bank conflicts";
        suggestion3.estimated_improvement = 10.0;
        suggestion3.implementation_effort = "medium";
        suggestion3.is_applicable = true;
        suggestions.push_back(suggestion3);
    }
    
    return suggestions;
}

std::vector<PerformanceOptimizer::OptimizationSuggestion> PerformanceOptimizer::suggest_compute_optimizations(const GpuProfiler::ProfilerStats& stats) {
    std::vector<OptimizationSuggestion> suggestions;
    
    if (is_compute_bound(stats)) {
        OptimizationSuggestion suggestion1;
        suggestion1.category = "compute";
        suggestion1.suggestion = "Use mixed precision arithmetic to reduce computation time";
        suggestion1.estimated_improvement = 30.0;
        suggestion1.implementation_effort = "low";
        suggestion1.is_applicable = true;
        suggestions.push_back(suggestion1);
        
        OptimizationSuggestion suggestion2;
        suggestion2.category = "compute";
        suggestion2.suggestion = "Implement warp-level primitives for reduction operations";
        suggestion2.estimated_improvement = 20.0;
        suggestion2.implementation_effort = "medium";
        suggestion2.is_applicable = true;
        suggestions.push_back(suggestion2);
        
        OptimizationSuggestion suggestion3;
        suggestion3.category = "compute";
        suggestion3.suggestion = "Consider using tensor cores or specialized hardware units";
        suggestion3.estimated_improvement = 50.0;
        suggestion3.implementation_effort = "high";
        suggestion3.is_applicable = true;
        suggestions.push_back(suggestion3);
    }
    
    return suggestions;
}

std::vector<PerformanceOptimizer::OptimizationSuggestion> PerformanceOptimizer::suggest_fusion_opportunities(const GpuProfiler::ProfilerStats& stats) {
    std::vector<OptimizationSuggestion> suggestions;
    
    // Suggest kernel fusion for frequently executed kernels
    for (const auto& pair : stats.kernel_counts) {
        if (pair.second > 100) { // Frequently executed kernel
            OptimizationSuggestion suggestion;
            suggestion.category = "kernel_fusion";
            suggestion.suggestion = "Consider fusing kernel " + pair.first + " with adjacent operations";
            suggestion.estimated_improvement = 20.0;
            suggestion.implementation_effort = "medium";
            suggestion.is_applicable = true;
            suggestions.push_back(suggestion);
        }
    }
    
    return suggestions;
}

void PerformanceOptimizer::print_optimization_report(const std::vector<OptimizationSuggestion>& suggestions) const {
    std::cout << "\n=== Performance Optimization Report ===\n";
    
    for (const auto& suggestion : suggestions) {
        if (suggestion.is_applicable) {
            std::cout << "\n" << suggestion.category << " optimization:\n";
            std::cout << "  Suggestion: " << suggestion.suggestion << "\n";
            std::cout << "  Estimated improvement: " << suggestion.estimated_improvement << "%\n";
            std::cout << "  Implementation effort: " << suggestion.implementation_effort << "\n";
        }
    }
    
    std::cout << "\n";
}

bool PerformanceOptimizer::is_memory_bound(const GpuProfiler::ProfilerStats& stats) const {
    return stats.peak_bandwidth > 0 && stats.total_execution_time > 0;
}

bool PerformanceOptimizer::is_compute_bound(const GpuProfiler::ProfilerStats& stats) const {
    return stats.total_kernels_executed > 0 && stats.average_kernel_time > 0;
}

double PerformanceOptimizer::calculate_potential_speedup(const std::string& optimization_type) const {
    // Simplified potential speedup calculation
    if (optimization_type == "memory") {
        return 1.2; // 20% potential improvement
    } else if (optimization_type == "compute") {
        return 1.5; // 50% potential improvement
    } else if (optimization_type == "fusion") {
        return 2.0; // 100% potential improvement
    }
    
    return 1.1; // 10% default
}

} // namespace utils
} // namespace jin