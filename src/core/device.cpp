#include "device.h"
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace jin {

Device::Device(DeviceType type, int id) : type_(type), id_(id) {
    if (type == DeviceType::GPU && id < 0) {
        throw std::invalid_argument("GPU device id must be non-negative");
    }
}

std::string Device::to_string() const {
    std::ostringstream oss;
    if (type_ == DeviceType::CPU) {
        oss << "cpu";
    } else {
        oss << "gpu:" << id_;
    }
    return oss.str();
}

DeviceMemoryManager& DeviceMemoryManager::instance() {
    static DeviceMemoryManager instance;
    return instance;
}

void* DeviceMemoryManager::allocate(Device device, size_t size) {
    if (device.is_cpu()) {
        return std::malloc(size);
    } else {
        // GPU allocation stub - will be implemented in Layer 2
        throw std::runtime_error("GPU allocation not yet implemented");
    }
}

void DeviceMemoryManager::deallocate(Device device, void* ptr) {
    if (ptr == nullptr) return;
    
    if (device.is_cpu()) {
        std::free(ptr);
    } else {
        // GPU deallocation stub
        throw std::runtime_error("GPU deallocation not yet implemented");
    }
}

void DeviceMemoryManager::copy(Device dst_device, void* dst, 
                                Device src_device, const void* src, 
                                size_t size) {
    if (dst_device.is_cpu() && src_device.is_cpu()) {
        std::memcpy(dst, src, size);
    } else {
        // GPU copy stub
        throw std::runtime_error("GPU memory copy not yet implemented");
    }
}

void DeviceMemoryManager::synchronize(Device device) {
    if (device.is_cpu()) {
        // CPU is always synchronized
        return;
    } else {
        // GPU synchronization stub
        throw std::runtime_error("GPU synchronization not yet implemented");
    }
}

} // namespace jin
