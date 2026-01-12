// Demonstration of Layer 2 GPU Acceleration Features
#include "../src/core/tensor.h"
#include "../src/core/device.h"
#include "../src/gpu/gpu_common.h"
#include <iostream>

using namespace jin;

int main() {
    std::cout << "=== Jin Layer 2 GPU Acceleration Demo ===\n\n";
    
    // 1. Device Selection
    std::cout << "1. Device Selection:\n";
    Device cpu_device = Device::cpu();
    Device cuda_device = Device::cuda(0);
    Device vulkan_device = Device::vulkan(0);
    Device metal_device = Device::metal(0);
    
    std::cout << "   CPU device: " << cpu_device.to_string() << "\n";
    std::cout << "   CUDA device: " << cuda_device.to_string() << "\n";
    std::cout << "   Vulkan device: " << vulkan_device.to_string() << "\n";
    std::cout << "   Metal device: " << metal_device.to_string() << "\n\n";
    
    // 2. Available Devices
    std::cout << "2. Available Devices:\n";
    auto available_devices = Device::get_available_devices();
    for (const auto& device : available_devices) {
        std::cout << "   - " << device.to_string() << "\n";
    }
    std::cout << "\n";
    
    // 3. Device Memory Manager
    std::cout << "3. Device Memory Manager:\n";
    DeviceMemoryManager& memory_manager = DeviceMemoryManager::instance();
    
    std::cout << "   CUDA available: " << (memory_manager.is_cuda_available() ? "Yes" : "No") << "\n";
    std::cout << "   Vulkan available: " << (memory_manager.is_vulkan_available() ? "Yes" : "No") << "\n";
    std::cout << "   Metal available: " << (memory_manager.is_metal_available() ? "Yes" : "No") << "\n\n";
    
    // 4. CPU Memory Allocation
    std::cout << "4. Memory Allocation Test:\n";
    void* cpu_ptr = memory_manager.allocate(cpu_device, 1024);
    std::cout << "   CPU allocation: " << (cpu_ptr ? "Success" : "Failed") << "\n";
    
    if (cpu_ptr) {
        memory_manager.deallocate(cpu_device, cpu_ptr);
        std::cout << "   CPU deallocation: Success\n";
    }
    
    // Try GPU allocation (will throw if not available)
    try {
        void* gpu_ptr = memory_manager.allocate(cuda_device, 1024);
        std::cout << "   GPU allocation: Success\n";
        memory_manager.deallocate(cuda_device, gpu_ptr);
        std::cout << "   GPU deallocation: Success\n";
    } catch (const std::exception& e) {
        std::cout << "   GPU allocation: " << e.what() << "\n";
    }
    std::cout << "\n";
    
    // 5. Tensor Device Creation
    std::cout << "5. Tensor Device Creation:\n";
    Shape shape = {2, 3};
    
    Tensor cpu_tensor(shape, DType::FP32, cpu_device);
    std::cout << "   CPU tensor device: " << cpu_tensor.device().to_string() << "\n";
    
    try {
        Tensor gpu_tensor(shape, DType::FP32, cuda_device);
        std::cout << "   GPU tensor device: " << gpu_tensor.device().to_string() << "\n";
    } catch (const std::exception& e) {
        std::cout << "   GPU tensor creation: " << e.what() << "\n";
    }
    std::cout << "\n";
    
    std::cout << "=== Layer 2 GPU Acceleration Demo Complete ===\n";
    std::cout << "\nFeatures implemented:\n";
    std::cout << "✓ Multi-backend device abstraction (CPU, CUDA, Vulkan, Metal)\n";
    std::cout << "✓ Memory management across devices\n";
    std::cout << "✓ Device capability detection\n";
    std::cout << "✓ Graceful fallback when GPU unavailable\n";
    std::cout << "✓ Stream-based async operations\n";
    std::cout << "✓ Kernel fusion optimization framework\n";
    std::cout << "✓ Performance profiling infrastructure\n";
    std::cout << "\nTo enable GPU acceleration:\n";
    std::cout << "1. Install CUDA Toolkit (for CUDA backend)\n";
    std::cout << "2. Install Vulkan SDK (for Vulkan backend)\n";
    std::cout << "3. Build with: python build.py release all --backend cuda\n";
    
    return 0;
}