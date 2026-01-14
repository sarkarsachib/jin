#pragma once

#include "../gpu/gpu_common.h"
#include "../core/tensor.h"
#include "../core/math_ops.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace jin {
namespace gpu {

// Forward declaration
struct FusionCandidate;

// Kernel fusion optimization
class KernelFusion {
public:
    struct FusionPattern {
        std::vector<std::string> operation_sequence;
        std::string fused_kernel_name;
        int estimated_speedup; // 2x, 3x, 5x, etc.
        size_t memory_saved; // bytes saved by fusion
    };
    
    struct FusionCandidate {
        std::vector<Tensor> input_tensors;
        std::vector<std::string> operations;
        std::vector<Tensor> output_tensors;
        FusionPattern* pattern;
        double estimated_performance_gain;
    };
    
    KernelFusion();
    ~KernelFusion();
    
    // Pattern detection and registration
    void register_fusion_pattern(const FusionPattern& pattern);
    std::vector<FusionCandidate> find_fusion_candidates(const std::vector<Tensor>& tensors,
                                                       const std::vector<std::string>& operations);
    
    // Fusion analysis
    bool can_fuse(const std::vector<std::string>& operations) const;
    double estimate_memory_savings(const std::vector<std::string>& operations) const;
    double estimate_performance_gain(const std::vector<std::string>& operations) const;
    
    // Code generation for fused kernels
    std::string generate_fused_kernel_code(const FusionCandidate& candidate, 
                                         const std::string& backend);
    
    // Fusion execution
    std::vector<Tensor> execute_fused_operations(const FusionCandidate& candidate,
                                             const std::vector<Tensor>& inputs);
    
    // Auto-tuning for fusion decisions
    void enable_auto_tuning(bool enable) { auto_tuning_enabled_ = enable; }
    void update_fusion_costs(const std::string& pattern_name, double actual_time);
    
    // Statistics
    struct FusionStats {
        size_t total_fusions_attempted = 0;
        size_t total_fusions_successful = 0;
        double total_time_saved = 0.0;
        size_t total_memory_saved = 0;
        std::unordered_map<std::string, size_t> pattern_usage;
    };
    
    FusionStats get_stats() const { return stats_; }
    void reset_stats();
    
private:
    void initialize_patterns();
    
    std::vector<FusionPattern> fusion_patterns_;
    bool auto_tuning_enabled_ = false;
    FusionStats stats_;
    
    // Backend-specific generators
    std::string generate_cuda_fused_kernel(const FusionCandidate& candidate);
    std::string generate_vulkan_fused_kernel(const FusionCandidate& candidate);
    std::string generate_metal_fused_kernel(const FusionCandidate& candidate);
    
    // Pattern analysis helpers
    std::vector<size_t> find_contiguous_ops(const std::vector<std::string>& operations, 
                                          const std::vector<std::string>& pattern);
    bool is_memory_compatible(const std::vector<Tensor>& tensors) const;
    
    // Performance modeling
    double estimate_op_cost(const std::string& operation, const Tensor& tensor) const;
    double estimate_memory_bandwidth_usage(const std::vector<Tensor>& tensors) const;
};

// Kernel fusion registry for different backends
class FusionRegistry {
public:
    static FusionRegistry& instance();
    
    void register_fusion_backend(const std::string& backend_name, 
                               std::function<std::string(const FusionCandidate&)> generator);
    
    std::string generate_fused_code(const std::string& backend, const FusionCandidate& candidate);
    
    std::vector<std::string> get_available_backends() const;
    
private:
    FusionRegistry() = default;
    
    std::unordered_map<std::string, std::function<std::string(const FusionCandidate&)>> generators_;
};

// Common fusion patterns for tensor operations
namespace patterns {

// Pattern: elementwise -> activation -> elementwise
inline KernelFusion::FusionPattern make_relu_fusion_pattern() {
    return {
        {"multiply", "relu", "add"},  // (a * b) -> relu -> + c
        "cuda_relu_fused_kernel",
        3, // 3x speedup
        0  // No memory saved, just computation fused
    };
}

// Pattern: matmul -> bias -> activation
inline KernelFusion::FusionPattern make_bias_activation_fusion_pattern() {
    return {
        {"matmul", "add", "relu"},  // (a @ b + bias) -> relu
        "cuda_bias_relu_fused_kernel",
        4, // 4x speedup
        sizeof(float) // One less intermediate tensor
    };
}

// Pattern: normalization operations
inline KernelFusion::FusionPattern make_normalization_fusion_pattern() {
    return {
        {"subtract", "multiply", "multiply", "add"},  // ((x - mean) * inv_std) * scale + bias
        "cuda_normalization_fused_kernel",
        5, // 5x speedup due to memory locality
        sizeof(float) * 2 // mean and variance tensors avoided
    };
}

// Pattern: elementwise chain
inline KernelFusion::FusionPattern make_elementwise_chain_pattern() {
    return {
        {"multiply", "add", "multiply", "add"},  // ((a * b) + c) * d + e
        "cuda_elementwise_chain_fused_kernel",
        2, // 2x speedup
        sizeof(float) * 3 // Three intermediate tensors avoided
    };
}

} // namespace patterns

} // namespace gpu
} // namespace jin