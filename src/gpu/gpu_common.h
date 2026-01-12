#pragma once

#include "../core/device.h"
#include "../core/dtype.h"
#include "../utils/shapes.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace jin {
namespace gpu {

// GPU Device capabilities
struct GpuCapabilities {
    bool compute_capability_supported = false;
    size_t max_threads_per_block = 0;
    size_t max_shared_memory_per_block = 0;
    size_t warp_size = 32;
    size_t max_blocks_per_dimension = 0;
    
    // Memory info
    size_t total_memory = 0;
    size_t available_memory = 0;
    
    // Performance metrics
    float compute_capability = 0.0f;
};

// GPU Memory allocation types
enum class GpuMemoryType {
    DEVICE,        // Regular GPU memory
    UNIFIED,      // Unified memory (Metal)
    PINNED,       // Pinned host memory for fast transfers
    SHARED        // Shared memory (scratch space)
};

// Base GPU memory pool
class GpuMemoryPool {
public:
    virtual ~GpuMemoryPool() = default;
    
    virtual void* allocate(size_t size, GpuMemoryType type = GpuMemoryType::DEVICE) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void reset() = 0;
    
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_peak_allocated() const { return peak_allocated_; }
    
protected:
    size_t total_allocated_ = 0;
    size_t peak_allocated_ = 0;
};

// GPU Stream for asynchronous operations
class GpuStream {
public:
    virtual ~GpuStream() = default;
    virtual void synchronize() = 0;
    virtual bool is_synchronized() const = 0;
    virtual void wait_event(void* event) = 0;
};

// GPU Event for synchronization
class GpuEvent {
public:
    virtual ~GpuEvent() = default;
    virtual void record() = 0;
    virtual void wait() = 0;
};

// GPU Kernel parameter structure
struct GpuKernelParams {
    void* arguments[32];  // Support up to 32 parameters
    int num_args = 0;
    
    template<typename T>
    void add_arg(T* arg) {
        if (num_args < 32) {
            arguments[num_args++] = static_cast<void*>(arg);
        }
    }
};

// GPU Kernel interface
class GpuKernel {
public:
    virtual ~GpuKernel() = default;
    virtual void launch(GpuStream& stream, const Shape& grid, const Shape& block, 
                       GpuKernelParams& params) = 0;
    virtual std::string name() const = 0;
};

// Base GPU runtime interface
class GpuRuntime {
public:
    virtual ~GpuRuntime() = default;
    
    // Device management
    virtual int get_device_count() = 0;
    virtual std::string get_device_name(int device_id) = 0;
    virtual GpuCapabilities get_device_capabilities(int device_id) = 0;
    virtual void set_device(int device_id) = 0;
    virtual int get_current_device() = 0;
    
    // Memory management
    virtual void* allocate_device(size_t size) = 0;
    virtual void deallocate_device(void* ptr) = 0;
    virtual void* allocate_pinned(size_t size) = 0;
    virtual void deallocate_pinned(void* ptr) = 0;
    
    // Memory copy operations
    virtual void copy_device_to_device(void* dst, const void* src, size_t size) = 0;
    virtual void copy_host_to_device(void* dst, const void* src, size_t size) = 0;
    virtual void copy_device_to_host(void* dst, const void* src, size_t size) = 0;
    virtual void copy_host_to_device_async(void* dst, const void* src, size_t size, GpuStream& stream) = 0;
    virtual void copy_device_to_host_async(void* dst, const void* src, size_t size, GpuStream& stream) = 0;
    
    // Stream management
    virtual std::unique_ptr<GpuStream> create_stream() = 0;
    virtual void destroy_stream(GpuStream* stream) = 0;
    
    // Event management
    virtual std::unique_ptr<GpuEvent> create_event() = 0;
    virtual void destroy_event(GpuEvent* event) = 0;
    
    // Synchronization
    virtual void synchronize() = 0;
    virtual bool is_available(int device_id) = 0;
    
    // Memory pools
    virtual std::unique_ptr<GpuMemoryPool> create_memory_pool() = 0;
    
    // Kernel compilation
    virtual std::unique_ptr<GpuKernel> compile_kernel(const std::string& source, 
                                                    const std::string& kernel_name) = 0;
    
    // Runtime info
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
};

// Global GPU runtime registry
class GpuRuntimeRegistry {
public:
    using RuntimeFactory = std::function<std::unique_ptr<GpuRuntime>()>;
    
    static GpuRuntimeRegistry& instance();
    
    void register_runtime(const std::string& name, RuntimeFactory factory);
    std::unique_ptr<GpuRuntime> create_runtime(const std::string& name);
    std::vector<std::string> get_available_runtimes() const;
    bool is_runtime_available(const std::string& name) const;
    
private:
    GpuRuntimeRegistry() = default;
    
    std::unordered_map<std::string, RuntimeFactory> factories_;
    std::vector<std::string> available_runtimes_;
};

// Utility functions
namespace utils {

// Compute optimal grid and block dimensions
struct Dim3 {
    size_t x, y, z;
    
    Dim3(size_t x = 1, size_t y = 1, size_t z = 1) : x(x), y(y), z(z) {}
};

Dim3 calculate_grid_dimensions(const Shape& tensor_shape, size_t threads_per_block = 256);
Dim3 calculate_block_dimensions(size_t threads_per_block = 256);

// Memory alignment utilities
size_t get_optimal_memory_alignment(size_t size);
void* align_memory(void* ptr, size_t alignment);

// Profiling utilities
class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();
    
    void start();
    void stop();
    double elapsed_ms() const;
    
private:
    void* start_event_;
    void* stop_event_;
    bool is_running_;
};

// Check GPU operation success
void check_gpu_error(const std::string& operation);
void check_gpu_error_code(int error_code, const std::string& operation);

} // namespace utils

} // namespace gpu
} // namespace jin