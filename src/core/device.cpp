#include "device.h"
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <algorithm>

namespace jin {

Device::Device(DeviceType type, int id) : type_(type), id_(id) {
    if ((type == DeviceType::CUDA || type == DeviceType::VULKAN || type == DeviceType::METAL) && id < 0) {
        throw std::invalid_argument("GPU device id must be non-negative");
    }
}

std::string Device::to_string() const {
    std::ostringstream oss;
    switch (type_) {
        case DeviceType::CPU:
            oss << "cpu";
            break;
        case DeviceType::CUDA:
            oss << "cuda:" << id_;
            break;
        case DeviceType::VULKAN:
            oss << "vulkan:" << id_;
            break;
        case DeviceType::METAL:
            oss << "metal:" << id_;
            break;
    }
    return oss.str();
}

Device Device::select(const std::string& backend, int id) {
    std::string backend_lower = backend;
    std::transform(backend_lower.begin(), backend_lower.end(), backend_lower.begin(), ::tolower);
    
    if (backend_lower == "cuda" || backend_lower == "gpu" || backend_lower == "gpu:0") {
        return cuda(id);
    } else if (backend_lower == "vulkan") {
        return vulkan(id);
    } else if (backend_lower == "metal") {
        return metal(id);
    } else if (backend_lower == "cpu" || backend_lower == "cpu:0") {
        return cpu();
    } else {
        throw std::runtime_error("Unknown device backend: " + backend);
    }
}

std::vector<Device> Device::get_available_devices() {
    std::vector<Device> devices;
    
    // Always add CPU
    devices.push_back(cpu());
    
    // Check for CUDA availability
    #ifdef __CUDACC__
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        if (cuda_runtime && cuda_runtime->is_available(0)) {
            int cuda_device_count = cuda_runtime->get_device_count();
            for (int i = 0; i < cuda_device_count; i++) {
                devices.push_back(cuda(i));
            }
        }
    } catch (...) {
        // CUDA not available
    }
    #endif
    
    // Check for Vulkan availability
    #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        if (vulkan_runtime && vulkan_runtime->is_available(0)) {
            int vulkan_device_count = vulkan_runtime->get_device_count();
            for (int i = 0; i < vulkan_device_count; i++) {
                devices.push_back(vulkan(i));
            }
        }
    } catch (...) {
        // Vulkan not available
    }
    #endif
    
    // Check for Metal availability (only on macOS)
    #ifdef __APPLE__
    try {
        auto& metal_runtime = gpu::metal::get_global_metal_runtime();
        if (metal_runtime && metal_runtime->is_available(0)) {
            int metal_device_count = metal_runtime->get_device_count();
            for (int i = 0; i < metal_device_count; i++) {
                devices.push_back(metal(i));
            }
        }
    } catch (...) {
        // Metal not available
    }
    #endif
    
    return devices;
}

DeviceMemoryManager& DeviceMemoryManager::instance() {
    static DeviceMemoryManager instance;
    return instance;
}

void* DeviceMemoryManager::allocate(Device device, size_t size) {
    switch (device.type()) {
        case DeviceType::CPU:
            return std::malloc(size);
        case DeviceType::CUDA:
            return allocate_cuda(size);
        case DeviceType::VULKAN:
            return allocate_vulkan(size);
        case DeviceType::METAL:
            return allocate_metal(size);
        default:
            throw std::runtime_error("Unknown device type for allocation");
    }
}

void DeviceMemoryManager::deallocate(Device device, void* ptr) {
    if (ptr == nullptr) return;
    
    switch (device.type()) {
        case DeviceType::CPU:
            std::free(ptr);
            break;
        case DeviceType::CUDA:
            deallocate_cuda(ptr);
            break;
        case DeviceType::VULKAN:
            deallocate_vulkan(ptr);
            break;
        case DeviceType::METAL:
            deallocate_metal(ptr);
            break;
        default:
            throw std::runtime_error("Unknown device type for deallocation");
    }
}

void DeviceMemoryManager::copy(Device dst_device, void* dst, Device src_device, const void* src, size_t size) {
    // Handle same-device copies
    if (dst_device == src_device) {
        if (dst_device.is_cpu()) {
            std::memcpy(dst, src, size);
        } else {
            // GPU to GPU copy
            copy_async(dst_device, dst, src_device, src, size, nullptr);
        }
        return;
    }
    
    // Handle cross-device copies
    if (src_device.is_cpu() && dst_device.is_cuda()) {
        copy_cpu_to_cuda(dst, const_cast<void*>(src), size);
    } else if (src_device.is_cuda() && dst_device.is_cpu()) {
        copy_cuda_to_cpu(dst, const_cast<void*>(src), size);
    } else if (src_device.is_cpu() && dst_device.is_vulkan()) {
        copy_cpu_to_vulkan(dst, const_cast<void*>(src), size);
    } else if (src_device.is_vulkan() && dst_device.is_cpu()) {
        copy_vulkan_to_cpu(dst, const_cast<void*>(src), size);
    } else if (src_device.is_cpu() && dst_device.is_metal()) {
        copy_cpu_to_metal(dst, const_cast<void*>(src), size);
    } else if (src_device.is_metal() && dst_device.is_cpu()) {
        copy_metal_to_cpu(dst, const_cast<void*>(src), size);
    } else if (src_device.is_cuda() && dst_device.is_vulkan()) {
        // CUDA to Vulkan: CPU intermediate
        void* temp = std::malloc(size);
        copy_cuda_to_cpu(temp, const_cast<void*>(src), size);
        copy_cpu_to_vulkan(dst, temp, size);
        std::free(temp);
    } else if (src_device.is_vulkan() && dst_device.is_cuda()) {
        // Vulkan to CUDA: CPU intermediate
        void* temp = std::malloc(size);
        copy_vulkan_to_cpu(temp, const_cast<void*>(src), size);
        copy_cpu_to_cuda(dst, temp, size);
        std::free(temp);
    } else {
        throw std::runtime_error("Unsupported device copy operation: " + 
                               src_device.to_string() + " -> " + dst_device.to_string());
    }
}

void DeviceMemoryManager::synchronize(Device device) {
    switch (device.type()) {
        case DeviceType::CPU:
            // CPU is always synchronized
            break;
        case DeviceType::CUDA:
            #ifdef __CUDACC__
            try {
                auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
                if (cuda_runtime) {
                    cuda_runtime->synchronize();
                }
            } catch (...) {
                // CUDA not available
            }
            #endif
            break;
        case DeviceType::VULKAN:
            #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
            try {
                auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
                if (vulkan_runtime) {
                    vulkan_runtime->synchronize();
                }
            } catch (...) {
                // Vulkan not available
            }
            #endif
            break;
        case DeviceType::METAL:
            #ifdef __APPLE__
            try {
                auto& metal_runtime = gpu::metal::get_global_metal_runtime();
                if (metal_runtime) {
                    metal_runtime->synchronize();
                }
            } catch (...) {
                // Metal not available
            }
            #endif
            break;
        default:
            throw std::runtime_error("Unknown device type for synchronization");
    }
}

// CUDA implementations
#ifdef __CUDACC__
void* DeviceMemoryManager::allocate_cuda(size_t size) {
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        if (cuda_runtime) {
            return cuda_runtime->allocate_device(size);
        }
    } catch (...) {
        // Fallback to basic allocation
    }
    throw std::runtime_error("CUDA not available for memory allocation");
}

void* DeviceMemoryManager::allocate_vulkan(size_t size) {
    #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        if (vulkan_runtime) {
            return vulkan_runtime->allocate_device(size);
        }
    } catch (...) {
        // Fallback to basic allocation
    }
    #endif
    throw std::runtime_error("Vulkan not available for memory allocation");
}

void* DeviceMemoryManager::allocate_metal(size_t size) {
    #ifdef __APPLE__
    try {
        auto& metal_runtime = gpu::metal::get_global_metal_runtime();
        if (metal_runtime) {
            return metal_runtime->allocate_device(size);
        }
    } catch (...) {
        // Fallback to basic allocation
    }
    #endif
    throw std::runtime_error("Metal not available for memory allocation");
}
#endif

// For non-CUDA builds, provide stub implementations that always throw
#ifndef __CUDACC__
void* DeviceMemoryManager::allocate_cuda(size_t size) {
    throw std::runtime_error("CUDA not available - recompile with CUDA support");
}

void* DeviceMemoryManager::allocate_vulkan(size_t size) {
    throw std::runtime_error("Vulkan not available - recompile with Vulkan support");
}

void* DeviceMemoryManager::allocate_metal(size_t size) {
    throw std::runtime_error("Metal not available - only available on macOS");
}
#endif

void DeviceMemoryManager::deallocate_cuda(void* ptr) {
    #ifdef __CUDACC__
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        if (cuda_runtime) {
            cuda_runtime->deallocate_device(ptr);
            return;
        }
    } catch (...) {
        // Fallback
    }
    #endif
}

void DeviceMemoryManager::deallocate_vulkan(void* ptr) {
    #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        if (vulkan_runtime) {
            vulkan_runtime->deallocate_device(ptr);
            return;
        }
    } catch (...) {
        // Fallback
    }
    #endif
}

void DeviceMemoryManager::deallocate_metal(void* ptr) {
    #ifdef __APPLE__
    try {
        auto& metal_runtime = gpu::metal::get_global_metal_runtime();
        if (metal_runtime) {
            metal_runtime->deallocate_device(ptr);
            return;
        }
    } catch (...) {
        // Fallback
    }
    #endif
}

// Memory transfer implementations
#ifdef __CUDACC__
void DeviceMemoryManager::copy_cuda_to_cpu(void* cpu_ptr, void* cuda_ptr, size_t size) {
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        if (cuda_runtime) {
            cuda_runtime->copy_device_to_host(cpu_ptr, cuda_ptr, size);
            return;
        }
    } catch (...) {
        // Fallback to manual copy if runtime not available
    }
    
    // Manual copy (slower, for testing/debugging)
    std::memcpy(cpu_ptr, cuda_ptr, size);
}
#endif

#ifndef __CUDACC__
void DeviceMemoryManager::copy_cuda_to_cpu(void* cpu_ptr, void* cuda_ptr, size_t size) {
    throw std::runtime_error("CUDA not available");
}
#endif

void DeviceMemoryManager::copy_cpu_to_cuda(void* cuda_ptr, void* cpu_ptr, size_t size) {
    #ifdef __CUDACC__
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        if (cuda_runtime) {
            cuda_runtime->copy_host_to_device(cuda_ptr, cpu_ptr, size);
            return;
        }
    } catch (...) {
        // Fallback to manual copy
    }
    #endif
    // Manual copy
    std::memcpy(cuda_ptr, cpu_ptr, size);
}

void DeviceMemoryManager::copy_cuda_to_cuda(void* dst_cuda, void* src_cuda, size_t size) {
    #ifdef __CUDACC__
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        if (cuda_runtime) {
            cuda_runtime->copy_device_to_device(dst_cuda, src_cuda, size);
            return;
        }
    } catch (...) {
        // Fallback to CPU intermediate
    }
    #endif
    // CPU intermediate copy
    void* temp = std::malloc(size);
    copy_cuda_to_cpu(temp, src_cuda, size);
    copy_cpu_to_cuda(dst_cuda, temp, size);
    std::free(temp);
}

void DeviceMemoryManager::copy_vulkan_to_cpu(void* cpu_ptr, void* vulkan_ptr, size_t size) {
    #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        if (vulkan_runtime) {
            vulkan_runtime->copy_device_to_host(cpu_ptr, vulkan_ptr, size);
            return;
        }
    } catch (...) {
        // Fallback
    }
    #endif
    throw std::runtime_error("Vulkan runtime not available");
}

void DeviceMemoryManager::copy_cpu_to_vulkan(void* vulkan_ptr, void* cpu_ptr, size_t size) {
    #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        if (vulkan_runtime) {
            vulkan_runtime->copy_host_to_device(vulkan_ptr, cpu_ptr, size);
            return;
        }
    } catch (...) {
        // Fallback
    }
    #endif
    throw std::runtime_error("Vulkan runtime not available");
}

void DeviceMemoryManager::copy_vulkan_to_vulkan(void* dst_vulkan, void* src_vulkan, size_t size) {
    #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        if (vulkan_runtime) {
            vulkan_runtime->copy_device_to_device(dst_vulkan, src_vulkan, size);
            return;
        }
    } catch (...) {
        // Fallback to CPU intermediate
    }
    #endif
    // CPU intermediate copy
    void* temp = std::malloc(size);
    copy_vulkan_to_cpu(temp, src_vulkan, size);
    copy_cpu_to_vulkan(dst_vulkan, temp, size);
    std::free(temp);
}

void DeviceMemoryManager::copy_metal_to_cpu(void* cpu_ptr, void* metal_ptr, size_t size) {
    #ifdef __APPLE__
    try {
        auto& metal_runtime = gpu::metal::get_global_metal_runtime();
        if (metal_runtime) {
            metal_runtime->copy_device_to_host(cpu_ptr, metal_ptr, size);
            return;
        }
    } catch (...) {
        // Fallback (Metal unified memory)
        std::memcpy(cpu_ptr, metal_ptr, size);
        return;
    }
    #endif
    throw std::runtime_error("Metal runtime not available");
}

void DeviceMemoryManager::copy_cpu_to_metal(void* metal_ptr, void* cpu_ptr, size_t size) {
    #ifdef __APPLE__
    try {
        auto& metal_runtime = gpu::metal::get_global_metal_runtime();
        if (metal_runtime) {
            metal_runtime->copy_host_to_device(metal_ptr, cpu_ptr, size);
            return;
        }
    } catch (...) {
        // Fallback (Metal unified memory)
        std::memcpy(metal_ptr, cpu_ptr, size);
        return;
    }
    #endif
    throw std::runtime_error("Metal runtime not available");
}

void DeviceMemoryManager::copy_metal_to_metal(void* dst_metal, void* src_metal, size_t size) {
    // Metal unified memory makes this automatic
    std::memcpy(dst_metal, src_metal, size);
}

void DeviceMemoryManager::copy_async(Device dst_device, void* dst, Device src_device, const void* src, size_t size, void* stream) {
    if (stream == nullptr) {
        // Synchronous copy
        copy(dst_device, dst, src_device, src, size);
        return;
    }
    
    // Async copy implementation would go here
    // For now, fallback to synchronous
    copy(dst_device, dst, src_device, src, size);
}

// Device capability queries
bool DeviceMemoryManager::is_cuda_available() {
    #ifdef __CUDACC__
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        return cuda_runtime && cuda_runtime->is_available(0);
    } catch (...) {
        return false;
    }
    #else
    return false;
    #endif
}

bool DeviceMemoryManager::is_vulkan_available() {
    #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        return vulkan_runtime && vulkan_runtime->is_available(0);
    } catch (...) {
        return false;
    }
    #else
    return false;
    #endif
}

bool DeviceMemoryManager::is_metal_available() {
    #ifdef __APPLE__
    try {
        auto& metal_runtime = gpu::metal::get_global_metal_runtime();
        return metal_runtime && metal_runtime->is_available(0);
    } catch (...) {
        return false;
    }
    #else
    return false;
    #endif
}

size_t DeviceMemoryManager::get_cuda_memory(int device_id) {
    #ifdef __CUDACC__
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        if (cuda_runtime) {
            auto caps = cuda_runtime->get_device_capabilities(device_id);
            return caps.available_memory;
        }
    } catch (...) {
        // Fallback
    }
    #endif
    return 0;
}

size_t DeviceMemoryManager::get_vulkan_memory(int device_id) {
    #if defined(__has_include) && __has_include(<vulkan/vulkan.h>)
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        if (vulkan_runtime) {
            auto caps = vulkan_runtime->get_device_capabilities(device_id);
            return caps.available_memory;
        }
    } catch (...) {
        // Fallback
    }
    #endif
    return 0;
}

size_t DeviceMemoryManager::get_metal_memory(int device_id) {
    #ifdef __APPLE__
    try {
        auto& metal_runtime = gpu::metal::get_global_metal_runtime();
        if (metal_runtime) {
            auto caps = metal_runtime->get_device_capabilities(device_id);
            return caps.available_memory;
        }
    } catch (...) {
        // Fallback
    }
    #endif
    return 0;
}

} // namespace jin
