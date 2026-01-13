#pragma once

// Conditionally include CUDA headers
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "../../gpu/gpu_common.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <stdexcept>

namespace jin {
namespace gpu {
namespace cuda {

// CUDA-specific implementations
#ifdef __CUDACC__
class CudaMemoryPool : public GpuMemoryPool {
public:
    explicit CudaMemoryPool(int device_id);
    ~CudaMemoryPool() override;
    
    void* allocate(size_t size, GpuMemoryType type = GpuMemoryType::DEVICE) override;
    void deallocate(void* ptr) override;
    void reset() override;
    
    // CUDA-specific allocation
    void* allocate_device_memory(size_t size);
    void* allocate_unified_memory(size_t size);
    void* allocate_pinned_memory(size_t size);
    
private:
    struct Allocation {
        void* ptr;
        size_t size;
        GpuMemoryType type;
        bool is_free;
    };
    
    int device_id_;
    std::vector<Allocation> allocations_;
    std::vector<void*> free_blocks_;
    size_t total_pool_size_;
    static constexpr size_t MAX_POOL_SIZE = 1ULL << 30; // 1GB max pool
};
#else
// Stub class for non-CUDA builds
class CudaMemoryPool : public GpuMemoryPool {
public:
    explicit CudaMemoryPool(int device_id) {}
    ~CudaMemoryPool() override {}
    
    void* allocate(size_t size, GpuMemoryType type = GpuMemoryType::DEVICE) override {
        throw std::runtime_error("CUDA not available - recompile with CUDA support");
    }
    void deallocate(void* ptr) override {}
    void reset() override {}
};
#endif

#ifdef __CUDACC__
class CudaStream : public GpuStream {
public:
    explicit CudaStream(int device_id);
    ~CudaStream() override;
    
    void synchronize() override;
    bool is_synchronized() const override;
    void wait_event(void* cuda_event) override;
    
    cudaStream_t get_cuda_stream() const { return stream_; }
    
private:
    cudaStream_t stream_;
    int device_id_;
};
#else
// Stub class for non-CUDA builds
class CudaStream : public GpuStream {
public:
    explicit CudaStream(int device_id) {}
    ~CudaStream() override {}
    
    void synchronize() override {}
    bool is_synchronized() const override { return true; }
    void wait_event(void* cuda_event) override {}
};
#endif

#ifdef __CUDACC__
class CudaEvent : public GpuEvent {
public:
    explicit CudaEvent(int device_id, bool uses_timing = true);
    ~CudaEvent() override;
    
    void record() override;
    void wait() override;
    
    cudaEvent_t get_cuda_event() const { return event_; }
    
private:
    cudaEvent_t event_;
    int device_id_;
};
#else
// Stub class for non-CUDA builds
class CudaEvent : public GpuEvent {
public:
    explicit CudaEvent(int device_id, bool uses_timing = true) {}
    ~CudaEvent() override {}
    
    void record() override {}
    void wait() override {}
};
#endif

#ifdef __CUDACC__
class CudaKernel : public GpuKernel {
public:
    CudaKernel(cudaFunction_t function, const std::string& name);
    ~CudaKernel() override = default;
    
    void launch(GpuStream& stream, const Shape& grid, const Shape& block, 
               GpuKernelParams& params) override;
    
    std::string name() const override { return name_; }
    
private:
    cudaFunction_t function_;
    std::string name_;
};
#else
// Stub class for non-CUDA builds
class CudaKernel : public GpuKernel {
public:
    CudaKernel(cudaFunction_t function, const std::string& name) : name_(name) {}
    ~CudaKernel() override = default;
    
    void launch(GpuStream& stream, const Shape& grid, const Shape& block, 
               GpuKernelParams& params) override {
        throw std::runtime_error("CUDA not available");
    }
    
    std::string name() const override { return name_; }
    
private:
    std::string name_;
};
#endif

#ifdef __CUDACC__
class CudaRuntime : public GpuRuntime {
public:
    CudaRuntime();
    ~CudaRuntime() override;
    
    // Device management
    int get_device_count() override;
    std::string get_device_name(int device_id) override;
    GpuCapabilities get_device_capabilities(int device_id) override;
    void set_device(int device_id) override;
    int get_current_device() override;
    
    // Memory management
    void* allocate_device(size_t size) override;
    void deallocate_device(void* ptr) override;
    void* allocate_pinned(size_t size) override;
    void deallocate_pinned(void* ptr) override;
    
    // Memory copy operations
    void copy_device_to_device(void* dst, const void* src, size_t size) override;
    void copy_host_to_device(void* dst, const void* src, size_t size) override;
    void copy_device_to_host(void* dst, const void* src, size_t size) override;
    void copy_host_to_device_async(void* dst, const void* src, size_t size, GpuStream& stream) override;
    void copy_device_to_host_async(void* dst, const void* src, size_t size, GpuStream& stream) override;
    
    // Stream management
    std::unique_ptr<GpuStream> create_stream() override;
    void destroy_stream(GpuStream* stream) override;
    
    // Event management
    std::unique_ptr<GpuEvent> create_event() override;
    void destroy_event(GpuEvent* event) override;
    
    // Synchronization
    void synchronize() override;
    bool is_available(int device_id) override;
    
    // Memory pools
    std::unique_ptr<GpuMemoryPool> create_memory_pool() override;
    
    // Kernel compilation
    std::unique_ptr<GpuKernel> compile_kernel(const std::string& source, 
                                            const std::string& kernel_name) override;
    
    // Runtime info
    std::string name() const override { return "CUDA"; }
    std::string version() const override;
    
    // CUDA-specific helper methods
    static void check_cuda_error(cudaError_t error, const std::string& operation);
    static bool is_cuda_available();
    
private:
    void initialize_cuda();
    void cleanup_cuda();
    
    int device_count_;
    int current_device_;
    std::unordered_map<int, GpuCapabilities> device_caps_;
    std::vector<std::unique_ptr<CudaMemoryPool>> memory_pools_;
    
    static bool initialized_;
    static int init_count_;
};
#else
// Stub class for non-CUDA builds
class CudaRuntime : public GpuRuntime {
public:
    CudaRuntime() {}
    ~CudaRuntime() override {}
    
    int get_device_count() override { return 0; }
    std::string get_device_name(int device_id) override { return "CUDA not available"; }
    GpuCapabilities get_device_capabilities(int device_id) override { return GpuCapabilities{}; }
    void set_device(int device_id) override {}
    int get_current_device() override { return 0; }
    
    void* allocate_device(size_t size) override { throw std::runtime_error("CUDA not available"); }
    void deallocate_device(void* ptr) override {}
    void* allocate_pinned(size_t size) override { throw std::runtime_error("CUDA not available"); }
    void deallocate_pinned(void* ptr) override {}
    
    void copy_device_to_device(void* dst, const void* src, size_t size) override { throw std::runtime_error("CUDA not available"); }
    void copy_host_to_device(void* dst, const void* src, size_t size) override { throw std::runtime_error("CUDA not available"); }
    void copy_device_to_host(void* dst, const void* src, size_t size) override { throw std::runtime_error("CUDA not available"); }
    void copy_host_to_device_async(void* dst, const void* src, size_t size, GpuStream& stream) override { throw std::runtime_error("CUDA not available"); }
    void copy_device_to_host_async(void* dst, const void* src, size_t size, GpuStream& stream) override { throw std::runtime_error("CUDA not available"); }
    
    std::unique_ptr<GpuStream> create_stream() override { throw std::runtime_error("CUDA not available"); }
    void destroy_stream(GpuStream* stream) override {}
    std::unique_ptr<GpuEvent> create_event() override { throw std::runtime_error("CUDA not available"); }
    void destroy_event(GpuEvent* event) override {}
    void synchronize() override {}
    bool is_available(int device_id) override { return false; }
    std::unique_ptr<GpuMemoryPool> create_memory_pool() override { throw std::runtime_error("CUDA not available"); }
    std::unique_ptr<GpuKernel> compile_kernel(const std::string& source, const std::string& kernel_name) override { throw std::runtime_error("CUDA not available"); }
    std::string version() const override { return "CUDA not available"; }
};
#endif

// Global CUDA runtime instance
std::unique_ptr<CudaRuntime>& get_global_cuda_runtime();

} // namespace cuda
} // namespace gpu
} // namespace jin