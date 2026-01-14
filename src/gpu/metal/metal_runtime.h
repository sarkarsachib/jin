#pragma once

#include "../../gpu/gpu_common.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace jin {
namespace gpu {
namespace metal {

// Metal device capabilities (simplified for non-Metal builds)
struct MetalCapabilities : public GpuCapabilities {
    // No Metal-specific fields when Metal is not available
};

// Metal-specific implementations (stub versions)
class MetalMemoryPool : public GpuMemoryPool {
public:
    MetalMemoryPool(void* device) {}
    ~MetalMemoryPool() override {}
    
    void* allocate(size_t size, GpuMemoryType type = GpuMemoryType::DEVICE) override {
        throw std::runtime_error("Metal not available - only available on macOS");
    }
    void deallocate(void* ptr) override {}
    void reset() override {}
};

class MetalStream : public GpuStream {
public:
    MetalStream(void* command_queue) {}
    ~MetalStream() override {}
    
    void synchronize() override {}
    bool is_synchronized() const override { return true; }
    void wait_event(void* metal_event) override {}
};

class MetalEvent : public GpuEvent {
public:
    MetalEvent(void* device) {}
    ~MetalEvent() override {}
    
    void record() override {}
    void wait() override {}
};

class MetalKernel : public GpuKernel {
public:
    MetalKernel(void* device, void* function, const std::string& name) : name_(name) {}
    ~MetalKernel() override {}
    
    void launch(GpuStream& stream, const Shape& grid, const Shape& block, 
               GpuKernelParams& params) override {
        throw std::runtime_error("Metal not available");
    }
    
    std::string name() const override { return name_; }
    
private:
    std::string name_;
};

class MetalRuntime : public GpuRuntime {
public:
    MetalRuntime() {}
    ~MetalRuntime() override {}
    
    int get_device_count() override { return 0; }
    std::string get_device_name(int device_id) override { return "Metal not available"; }
    GpuCapabilities get_device_capabilities(int device_id) override { return GpuCapabilities{}; }
    void set_device(int device_id) override {}
    int get_current_device() override { return 0; }
    
    void* allocate_device(size_t size) override { 
        throw std::runtime_error("Metal not available"); 
    }
    void deallocate_device(void* ptr) override {}
    void* allocate_pinned(size_t size) override { 
        throw std::runtime_error("Metal not available"); 
    }
    void deallocate_pinned(void* ptr) override {}
    
    void copy_device_to_device(void* dst, const void* src, size_t size) override { 
        throw std::runtime_error("Metal not available"); 
    }
    void copy_host_to_device(void* dst, const void* src, size_t size) override { 
        throw std::runtime_error("Metal not available"); 
    }
    void copy_device_to_host(void* dst, const void* src, size_t size) override { 
        throw std::runtime_error("Metal not available"); 
    }
    void copy_host_to_device_async(void* dst, const void* src, size_t size, GpuStream& stream) override { 
        throw std::runtime_error("Metal not available"); 
    }
    void copy_device_to_host_async(void* dst, const void* src, size_t size, GpuStream& stream) override { 
        throw std::runtime_error("Metal not available"); 
    }
    
    std::unique_ptr<GpuStream> create_stream() override { 
        throw std::runtime_error("Metal not available"); 
    }
    void destroy_stream(GpuStream* stream) override {}
    std::unique_ptr<GpuEvent> create_event() override { 
        throw std::runtime_error("Metal not available"); 
    }
    void destroy_event(GpuEvent* event) override {}
    void synchronize() override {}
    bool is_available(int device_id) override { return false; }
    std::unique_ptr<GpuMemoryPool> create_memory_pool() override { 
        throw std::runtime_error("Metal not available"); 
    }
    std::unique_ptr<GpuKernel> compile_kernel(const std::string& source, const std::string& kernel_name) override { 
        throw std::runtime_error("Metal not available"); 
    }
    std::string version() const override { return "Metal not available"; }
};

// Global Metal runtime instance
std::unique_ptr<MetalRuntime>& get_global_metal_runtime();

} // namespace metal
} // namespace gpu
} // namespace jin