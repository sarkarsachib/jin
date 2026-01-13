#pragma once

#include <string>
#include <vector>
#include <memory>

namespace jin {

enum class DeviceType {
    CPU,
    CUDA,
    VULKAN,
    METAL
};

class Device {
public:
    Device(DeviceType type, int id = 0);
    
    DeviceType type() const { return type_; }
    int id() const { return id_; }
    std::string to_string() const;
    
    bool is_cpu() const { return type_ == DeviceType::CPU; }
    bool is_gpu() const { return type_ == DeviceType::CUDA || type_ == DeviceType::VULKAN || type_ == DeviceType::METAL; }
    bool is_cuda() const { return type_ == DeviceType::CUDA; }
    bool is_vulkan() const { return type_ == DeviceType::VULKAN; }
    bool is_metal() const { return type_ == DeviceType::METAL; }
    
    bool operator==(const Device& other) const {
        return type_ == other.type_ && id_ == other.id_;
    }
    
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }
    
    static Device cpu() { return Device(DeviceType::CPU, 0); }
    static Device cuda(int id = 0) { return Device(DeviceType::CUDA, id); }
    static Device vulkan(int id = 0) { return Device(DeviceType::VULKAN, id); }
    static Device metal(int id = 0) { return Device(DeviceType::METAL, id); }
    
    static Device select(const std::string& backend, int id = 0);
    static std::vector<Device> get_available_devices();
    
private:
    DeviceType type_;
    int id_;
};

class DeviceMemoryManager {
public:
    static DeviceMemoryManager& instance();
    
    void* allocate(Device device, size_t size);
    void deallocate(Device device, void* ptr);
    void copy(Device dst_device, void* dst, Device src_device, const void* src, size_t size);
    void synchronize(Device device);
    
    // GPU-specific memory operations
    void* allocate_cuda(size_t size);
    void* allocate_vulkan(size_t size);
    void* allocate_metal(size_t size);
    
    void deallocate_cuda(void* ptr);
    void deallocate_vulkan(void* ptr);
    void deallocate_metal(void* ptr);
    
    // Memory transfer between devices
    void copy_cuda_to_cpu(void* cpu_ptr, void* cuda_ptr, size_t size);
    void copy_cpu_to_cuda(void* cuda_ptr, void* cpu_ptr, size_t size);
    void copy_cuda_to_cuda(void* dst_cuda, void* src_cuda, size_t size);
    
    void copy_vulkan_to_cpu(void* cpu_ptr, void* vulkan_ptr, size_t size);
    void copy_cpu_to_vulkan(void* vulkan_ptr, void* cpu_ptr, size_t size);
    void copy_vulkan_to_vulkan(void* dst_vulkan, void* src_vulkan, size_t size);
    
    void copy_metal_to_cpu(void* cpu_ptr, void* metal_ptr, size_t size);
    void copy_cpu_to_metal(void* metal_ptr, void* cpu_ptr, size_t size);
    void copy_metal_to_metal(void* dst_metal, void* src_metal, size_t size);
    
    // Async operations
    void copy_async(Device dst_device, void* dst, Device src_device, const void* src, size_t size, void* stream);
    
    // Device capability queries
    bool is_cuda_available();
    bool is_vulkan_available();
    bool is_metal_available();
    
    size_t get_cuda_memory(int device_id = 0);
    size_t get_vulkan_memory(int device_id = 0);
    size_t get_metal_memory(int device_id = 0);
    
private:
    DeviceMemoryManager() = default;
};

} // namespace jin
