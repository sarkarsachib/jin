#include "kernel_fusion.h"
#include "../core/tensor.h"
#include "../gpu/gpu_common.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace jin {
namespace gpu {

// KernelFusion implementation
KernelFusion::KernelFusion() {
    initialize_patterns();
}

KernelFusion::~KernelFusion() = default;

void KernelFusion::initialize_patterns() {
    // Register common fusion patterns
    fusion_patterns_.push_back(patterns::make_relu_fusion_pattern());
    fusion_patterns_.push_back(patterns::make_bias_activation_fusion_pattern());
    fusion_patterns_.push_back(patterns::make_normalization_fusion_pattern());
    fusion_patterns_.push_back(patterns::make_elementwise_chain_pattern());
}

void KernelFusion::register_fusion_pattern(const FusionPattern& pattern) {
    fusion_patterns_.push_back(pattern);
}

std::vector<KernelFusion::FusionCandidate> KernelFusion::find_fusion_candidates(
    const std::vector<Tensor>& tensors,
    const std::vector<std::string>& operations) {
    
    std::vector<FusionCandidate> candidates;
    
    // For each pattern, check if it matches a subsequence of operations
    for (const auto& pattern : fusion_patterns_) {
        auto matches = find_contiguous_ops(operations, pattern.operation_sequence);
        
        for (size_t start_idx : matches) {
            FusionCandidate candidate;
            
            // Extract input tensors for this operation sequence
            // This is a simplified version - in practice you'd track tensor dependencies
            candidate.input_tensors = tensors; // Simplified
            candidate.operations = std::vector<std::string>(
                operations.begin() + start_idx,
                operations.begin() + start_idx + pattern.operation_sequence.size()
            );
            candidate.pattern = const_cast<FusionPattern*>(&pattern);
            candidate.estimated_performance_gain = estimate_performance_gain(candidate.operations);
            
            candidates.push_back(candidate);
        }
    }
    
    return candidates;
}

std::vector<size_t> KernelFusion::find_contiguous_ops(const std::vector<std::string>& operations,
                                                    const std::vector<std::string>& pattern) {
    std::vector<size_t> matches;
    
    if (pattern.size() > operations.size()) {
        return matches;
    }
    
    for (size_t i = 0; i <= operations.size() - pattern.size(); i++) {
        bool match = true;
        for (size_t j = 0; j < pattern.size(); j++) {
            if (operations[i + j] != pattern[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            matches.push_back(i);
        }
    }
    
    return matches;
}

bool KernelFusion::can_fuse(const std::vector<std::string>& operations) const {
    for (const auto& pattern : fusion_patterns_) {
        auto matches = find_contiguous_ops(operations, pattern.operation_sequence);
        if (!matches.empty()) {
            return true;
        }
    }
    return false;
}

double KernelFusion::estimate_memory_savings(const std::vector<std::string>& operations) const {
    double savings = 0.0;
    
    for (const auto& pattern : fusion_patterns_) {
        auto matches = find_contiguous_ops(operations, pattern.operation_sequence);
        for (size_t match : matches) {
            savings += pattern.memory_saved;
        }
    }
    
    return savings;
}

double KernelFusion::estimate_performance_gain(const std::vector<std::string>& operations) const {
    double max_gain = 1.0;
    
    for (const auto& pattern : fusion_patterns_) {
        auto matches = find_contiguous_ops(operations, pattern.operation_sequence);
        for (size_t match : matches) {
            max_gain = std::max(max_gain, static_cast<double>(pattern.estimated_speedup));
        }
    }
    
    return max_gain;
}

std::string KernelFusion::generate_fused_kernel_code(const FusionCandidate& candidate,
                                                    const std::string& backend) {
    if (backend == "cuda") {
        return generate_cuda_fused_kernel(candidate);
    } else if (backend == "vulkan") {
        return generate_vulkan_fused_kernel(candidate);
    } else if (backend == "metal") {
        return generate_metal_fused_kernel(candidate);
    } else {
        throw std::runtime_error("Unsupported backend for kernel fusion: " + backend);
    }
}

std::string KernelFusion::generate_cuda_fused_kernel(const FusionCandidate& candidate) {
    std::ostringstream code;
    
    // Generate CUDA kernel header
    code << "__global__ void " << candidate.pattern->fused_kernel_name << "(\n";
    
    // Generate parameter list based on operations
    int param_count = 0;
    for (const auto& op : candidate.operations) {
        code << "    float* " << op << "_param_" << param_count;
        if (param_count < static_cast<int>(candidate.operations.size()) - 1) {
            code << ",\n";
        } else {
            code << "\n";
        }
        param_count++;
    }
    
    code << ") {\n";
    code << "    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
    code << "    if (idx < data_size) {\n";
    
    // Generate fused operation code
    if (candidate.operations.size() >= 3) {
        // Example for a simple fusion pattern
        code << "        float temp = " << candidate.operations[0] << "_param_0[idx] * " 
             << candidate.operations[0] << "_param_1[idx];\n";
        
        for (size_t i = 1; i < candidate.operations.size() - 1; i++) {
            code << "        temp = ";
            if (candidate.operations[i] == "relu") {
                code << "max(0.0f, temp);\n";
            } else if (candidate.operations[i] == "add") {
                code << "temp + " << candidate.operations[i] << "_param_" << i << "[idx];\n";
            } else if (candidate.operations[i] == "multiply") {
                code << "temp * " << candidate.operations[i] << "_param_" << i << "[idx];\n";
            }
        }
        
        // Write final result
        code << "        " << candidate.operations.back() << "_param_" << (candidate.operations.size() - 1) 
             << "[idx] = temp;\n";
    }
    
    code << "    }\n";
    code << "}\n";
    
    return code.str();
}

std::string KernelFusion::generate_vulkan_fused_kernel(const FusionCandidate& candidate) {
    std::ostringstream code;
    
    // Generate GLSL compute shader
    code << "#version 450\n";
    code << "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n\n";
    
    // Declare storage buffers
    for (size_t i = 0; i < candidate.operations.size(); i++) {
        code << "layout(binding = " << i << ") buffer " << candidate.operations[i] << "_buffer {\n";
        code << "    float " << candidate.operations[i] << "_data[];\n";
        code << "};\n\n";
    }
    
    code << "void main() {\n";
    code << "    uint idx = gl_GlobalInvocationID.x;\n";
    code << "    if (idx >= data_size) return;\n\n";
    
    // Generate fused operation code (similar to CUDA but in GLSL)
    if (candidate.operations.size() >= 3) {
        code << "    float temp = " << candidate.operations[0] << "_data[idx] * " 
             << candidate.operations[0] << "_data[idx];\n";
        
        for (size_t i = 1; i < candidate.operations.size() - 1; i++) {
            code << "    temp = ";
            if (candidate.operations[i] == "relu") {
                code << "max(0.0, temp);\n";
            } else if (candidate.operations[i] == "add") {
                code << "temp + " << candidate.operations[i] << "_data[idx];\n";
            } else if (candidate.operations[i] == "multiply") {
                code << "temp * " << candidate.operations[i] << "_data[idx];\n";
            }
        }
        
        code << "    " << candidate.operations.back() << "_data[idx] = temp;\n";
    }
    
    code << "}\n";
    
    return code.str();
}

std::string KernelFusion::generate_metal_fused_kernel(const FusionCandidate& candidate) {
    std::ostringstream code;
    
    // Generate Metal kernel
    code << "kernel void " << candidate.pattern->fused_kernel_name << "(\n";
    
    for (size_t i = 0; i < candidate.operations.size(); i++) {
        code << "    device float* " << candidate.operations[i] << "_param [[buffer(" << i << ")]]";
        if (i < candidate.operations.size() - 1) {
            code << ",\n";
        } else {
            code << "\n";
        }
    }
    
    code << ") {\n";
    code << "    uint idx = thread_position_in_grid.x;\n";
    code << "    if (idx >= data_size) return;\n\n";
    
    // Generate fused operation code
    if (candidate.operations.size() >= 3) {
        code << "    float temp = " << candidate.operations[0] << "_param[idx] * " 
             << candidate.operations[0] << "_param[idx];\n";
        
        for (size_t i = 1; i < candidate.operations.size() - 1; i++) {
            code << "    temp = ";
            if (candidate.operations[i] == "relu") {
                code << "max(0.0, temp);\n";
            } else if (candidate.operations[i] == "add") {
                code << "temp + " << candidate.operations[i] << "_param[idx];\n";
            } else if (candidate.operations[i] == "multiply") {
                code << "temp * " << candidate.operations[i] << "_param[idx];\n";
            }
        }
        
        code << "    " << candidate.operations.back() << "_param[idx] = temp;\n";
    }
    
    code << "}\n";
    
    return code.str();
}

std::vector<Tensor> KernelFusion::execute_fused_operations(const FusionCandidate& candidate,
                                                       const std::vector<Tensor>& inputs) {
    // This would execute the fused kernel on the appropriate backend
    // For now, return a placeholder result
    
    std::vector<Tensor> outputs;
    
    // Simplified: just return empty tensors for now
    // In practice, you'd:
    // 1. Compile the fused kernel for the appropriate backend
    // 2. Execute it with the input tensors
    // 3. Return the output tensors
    
    std::cout << "Executing fused kernel: " << candidate.pattern->fused_kernel_name 
              << " with estimated speedup: " << candidate.pattern->estimated_speedup << "x\n";
    
    return outputs;
}

void KernelFusion::update_fusion_costs(const std::string& pattern_name, double actual_time) {
    // Update performance model based on actual execution times
    // This enables auto-tuning of fusion decisions
    for (auto& pattern : fusion_patterns_) {
        if (pattern.fused_kernel_name == pattern_name) {
            // Update estimated performance based on actual results
            // This is a simplified version
            break;
        }
    }
}

void KernelFusion::reset_stats() {
    stats_ = FusionStats{};
}

// Performance modeling helpers
bool KernelFusion::is_memory_compatible(const std::vector<Tensor>& tensors) const {
    // Check if all tensors are on the same device and have compatible memory layout
    if (tensors.empty()) return true;
    
    const Device& first_device = tensors[0].device();
    
    for (const auto& tensor : tensors) {
        if (tensor.device() != first_device) {
            return false;
        }
        if (!tensor.is_contiguous()) {
            return false;
        }
    }
    
    return true;
}

double KernelFusion::estimate_op_cost(const std::string& operation, const Tensor& tensor) const {
    size_t element_count = tensor.size();
    
    // Simple cost model based on operation complexity
    switch (hash_operation(operation)) {
        case hash_operation("add"):
        case hash_operation("multiply"):
        case hash_operation("subtract"):
        case hash_operation("divide"):
            return element_count * 1.0; // O(n)
            
        case hash_operation("matmul"):
            // Assuming matrix multiplication
            return element_count * std::log(element_count); // O(n^2 log n) approximation
            
        case hash_operation("relu"):
        case hash_operation("exp"):
        case hash_operation("log"):
        case hash_operation("sigmoid"):
        case hash_operation("tanh"):
            return element_count * 2.0; // O(n) but more expensive
            
        case hash_operation("sum"):
        case hash_operation("mean"):
            return element_count * 0.5; // O(n) but reduces data
            
        default:
            return element_count * 1.0;
    }
}

// Simple hash function for operations
size_t KernelFusion::hash_operation(const std::string& operation) const {
    std::hash<std::string> hasher;
    return hasher(operation);
}

double KernelFusion::estimate_memory_bandwidth_usage(const std::vector<Tensor>& tensors) const {
    double bandwidth = 0.0;
    
    for (const auto& tensor : tensors) {
        bandwidth += tensor.size() * sizeof(float); // FP32
    }
    
    return bandwidth;
}

// FusionRegistry implementation
FusionRegistry& FusionRegistry::instance() {
    static FusionRegistry instance;
    return instance;
}

void FusionRegistry::register_fusion_backend(const std::string& backend_name,
                                          std::function<std::string(const FusionCandidate&)> generator) {
    generators_[backend_name] = generator;
}

std::string FusionRegistry::generate_fused_code(const std::string& backend,
                                              const FusionCandidate& candidate) {
    auto it = generators_.find(backend);
    if (it == generators_.end()) {
        throw std::runtime_error("No fusion generator found for backend: " + backend);
    }
    
    return it->second(candidate);
}

std::vector<std::string> FusionRegistry::get_available_backends() const {
    std::vector<std::string> backends;
    for (const auto& pair : generators_) {
        backends.push_back(pair.first);
    }
    return backends;
}

} // namespace gpu
} // namespace jin