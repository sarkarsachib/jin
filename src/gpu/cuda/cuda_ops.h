#pragma once

#include "../cuda_runtime.h"
#include "../../core/tensor.h"
#include "../../core/math_ops.h"
#include <memory>
#include <functional>

namespace jin {
namespace gpu {
namespace cuda {

// CUDA Operations class - wraps kernel calls
class CudaOps {
public:
    explicit CudaOps(int device_id = 0);
    ~CudaOps();
    
    // Binary operations
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor multiply(const Tensor& a, const Tensor& b);
    Tensor subtract(const Tensor& a, const Tensor& b);
    Tensor divide(const Tensor& a, const Tensor& b);
    
    // Matrix operations
    Tensor matmul(const Tensor& a, const Tensor& b);
    
    // Unary operations
    Tensor exp(const Tensor& a);
    Tensor log(const Tensor& a);
    Tensor relu(const Tensor& a);
    Tensor tanh(const Tensor& a);
    Tensor sigmoid(const Tensor& a);
    Tensor neg(const Tensor& a);
    
    // Reduction operations
    Tensor sum(const Tensor& a, int axis = -1, bool keepdims = false);
    Tensor mean(const Tensor& a, int axis = -1, bool keepdims = false);
    
    // Shape operations
    Tensor transpose(const Tensor& a, const std::vector<int>& axes = {});
    
    // Gradient operations (for autodiff)
    Tensor relu_backward(const Tensor& input, const Tensor& grad_output);
    Tensor tanh_backward(const Tensor& input, const Tensor& grad_output);
    Tensor sigmoid_backward(const Tensor& input, const Tensor& grad_output);
    Tensor exp_backward(const Tensor& input, const Tensor& grad_output);
    
    // Utility methods
    void synchronize();
    bool is_compatible(const Tensor& a, const Tensor& b);
    Device get_device() const { return Device::gpu(device_id_); }
    
private:
    int device_id_;
    std::unique_ptr<CudaRuntime> runtime_;
    std::unique_ptr<CudaMemoryPool> memory_pool_;
    
    // Helper methods
    void ensure_device();
    void ensure_tensor_on_device(Tensor& tensor);
    void ensure_compatible(const Tensor& a, const Tensor& b);
    
    // Memory transfer helpers
    void host_to_device(Tensor& tensor, const void* host_data, size_t size);
    void device_to_host(void* host_data, const Tensor& tensor);
    
    // Kernel launch helpers
    template<typename KernelFunc>
    void launch_elementwise_kernel(KernelFunc kernel, const Tensor& input, Tensor& output);
    
    template<typename KernelFunc>
    void launch_binary_kernel(KernelFunc kernel, const Tensor& a, const Tensor& b, Tensor& output);
    
    template<typename KernelFunc>
    void launch_reduction_kernel(KernelFunc kernel, const Tensor& input, Tensor& output, size_t reduced_size);
    
    // CUDA kernel function pointers (would be loaded from compiled PTX)
    // For now, these are function pointers to the actual kernels
    void* cuda_add_kernel_func_ = nullptr;
    void* cuda_multiply_kernel_func_ = nullptr;
    void* cuda_subtract_kernel_func_ = nullptr;
    void* cuda_divide_kernel_func_ = nullptr;
    void* cuda_exp_kernel_func_ = nullptr;
    void* cuda_log_kernel_func_ = nullptr;
    void* cuda_relu_kernel_func_ = nullptr;
    void* cuda_tanh_kernel_func_ = nullptr;
    void* cuda_sigmoid_kernel_func_ = nullptr;
    void* cuda_neg_kernel_func_ = nullptr;
    void* cuda_matmul_kernel_func_ = nullptr;
    void* cuda_sum_kernel_func_ = nullptr;
    void* cuda_mean_kernel_func_ = nullptr;
    void* cuda_transpose_kernel_func_ = nullptr;
    void* cuda_relu_backward_kernel_func_ = nullptr;
    void* cuda_tanh_backward_kernel_func_ = nullptr;
    void* cuda_sigmoid_backward_kernel_func_ = nullptr;
    void* cuda_exp_backward_kernel_func_ = nullptr;
};

// CUDA memory manager for tensor operations
class CudaMemoryManager {
public:
    static CudaMemoryManager& instance();
    
    // Memory allocation/deallocation
    void* allocate_device(size_t size, DType dtype);
    void deallocate_device(void* ptr);
    
    // Memory transfer
    void copy_to_device(void* device_ptr, const void* host_ptr, size_t size);
    void copy_from_device(void* host_ptr, const void* device_ptr, size_t size);
    void copy_device_to_device(void* dst, const void* src, size_t size);
    
    // Async operations
    void copy_to_device_async(void* device_ptr, const void* host_ptr, size_t size, void* stream);
    void copy_from_device_async(void* host_ptr, const void* device_ptr, size_t size, void* stream);
    
    // Synchronization
    void synchronize();
    
    // Memory info
    size_t get_free_memory();
    size_t get_total_memory();
    
private:
    CudaMemoryManager() = default;
    
    int device_id_ = 0;
    std::unique_ptr<CudaRuntime> runtime_;
};

// Global CUDA operations instance
CudaOps& get_global_cuda_ops(int device_id = 0);

} // namespace cuda
} // namespace gpu
} // namespace jin