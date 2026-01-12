#pragma once

#include <string>
#include <memory>

namespace jin {

enum class DeviceType {
    CPU,
    GPU
};

class Device {
public:
    Device(DeviceType type, int id = 0);
    
    DeviceType type() const { return type_; }
    int id() const { return id_; }
    std::string to_string() const;
    
    bool is_cpu() const { return type_ == DeviceType::CPU; }
    bool is_gpu() const { return type_ == DeviceType::GPU; }
    
    bool operator==(const Device& other) const {
        return type_ == other.type_ && id_ == other.id_;
    }
    
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }
    
    static Device cpu() { return Device(DeviceType::CPU, 0); }
    static Device gpu(int id = 0) { return Device(DeviceType::GPU, id); }
    
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
    
private:
    DeviceMemoryManager() = default;
};

} // namespace jin
