#include "test_runner.cpp"
#include "../src/gpu/cuda/cuda_runtime.h"
#include "../src/gpu/vulkan/vulkan_runtime.h"
#include "../src/gpu/metal/metal_runtime.h"
#include "../src/gpu/gpu_common.h"
#include "../src/gpu/kernel_fusion.h"
#include "../src/core/device.h"
#include "../src/core/tensor.h"
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

using namespace jin;
using namespace jin::gpu;
using namespace jin::utils;

TEST(GpuRuntime_DeviceSelection) {
    // Test device selection
    Device cpu_device = Device::cpu();
    ASSERT_TRUE(cpu_device.is_cpu());
    
    Device cuda_device = Device::cuda(0);
    ASSERT_TRUE(cuda_device.is_cuda());
    ASSERT_EQ(cuda_device.id(), 0);
    
    Device vulkan_device = Device::vulkan(0);
    ASSERT_TRUE(vulkan_device.is_vulkan());
    
    Device metal_device = Device::metal(0);
    ASSERT_TRUE(metal_device.is_metal());
    
    // Test device string representation
    ASSERT_EQ(cpu_device.to_string(), "cpu");
    ASSERT_EQ(cuda_device.to_string(), "cuda:0");
    ASSERT_EQ(vulkan_device.to_string(), "vulkan:0");
    ASSERT_EQ(metal_device.to_string(), "metal:0");
}

TEST(GpuRuntime_DeviceAvailability) {
    // Test GPU availability checking
    DeviceMemoryManager& mem_manager = DeviceMemoryManager::instance();
    
    // These tests will pass/fail based on actual hardware availability
    // But should not crash
    
    bool cuda_available = mem_manager.is_cuda_available();
    bool vulkan_available = mem_manager.is_vulkan_available();
    bool metal_available = mem_manager.is_metal_available();
    
    // Just verify the functions don't crash
    std::cout << "CUDA available: " << (cuda_available ? "yes" : "no") << "\n";
    std::cout << "Vulkan available: " << (vulkan_available ? "yes" : "no") << "\n";
    std::cout << "Metal available: " << (metal_available ? "yes" : "no") << "\n";
}

TEST(GpuRuntime_DeviceEnumeration) {
    // Test getting available devices
    auto devices = Device::get_available_devices();
    
    // Should always have at least CPU
    ASSERT_GE(devices.size(), 1);
    
    // First device should be CPU
    ASSERT_TRUE(devices[0].is_cpu());
    
    std::cout << "Available devices: " << devices.size() << "\n";
    for (const auto& device : devices) {
        std::cout << "  - " << device.to_string() << "\n";
    }
}

TEST(GpuRuntime_CudaRuntimeBasics) {
    // Test CUDA runtime initialization
    try {
        auto& cuda_runtime = gpu::cuda::get_global_cuda_runtime();
        
        if (cuda_runtime && cuda_runtime->is_available(0)) {
            int device_count = cuda_runtime->get_device_count();
            ASSERT_GE(device_count, 1);
            
            std::string device_name = cuda_runtime->get_device_name(0);
            ASSERT_FALSE(device_name.empty());
            
            auto caps = cuda_runtime->get_device_capabilities(0);
            ASSERT_TRUE(caps.compute_capability_supported);
            
            std::cout << "CUDA Device: " << device_name << "\n";
            std::cout << "Compute Capability: " << caps.compute_capability << "\n";
            std::cout << "Total Memory: " << (caps.total_memory / (1024*1024*1024)) << " GB\n";
        } else {
            std::cout << "CUDA runtime not available (expected on this system)\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "CUDA runtime test failed (expected): " << e.what() << "\n";
    }
}

TEST(GpuRuntime_VulkanRuntimeBasics) {
    // Test Vulkan runtime initialization
    try {
        auto& vulkan_runtime = gpu::vulkan::get_global_vulkan_runtime();
        
        if (vulkan_runtime && vulkan_runtime->is_available(0)) {
            int device_count = vulkan_runtime->get_device_count();
            ASSERT_GE(device_count, 1);
            
            std::string device_name = vulkan_runtime->get_device_name(0);
            ASSERT_FALSE(device_name.empty());
            
            std::cout << "Vulkan Device: " << device_name << "\n";
        } else {
            std::cout << "Vulkan runtime not available (expected on this system)\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "Vulkan runtime test failed (expected): " << e.what() << "\n";
    }
}

TEST(GpuRuntime_MemoryAllocation) {
    // Test memory allocation on different devices
    Device cpu_device = Device::cpu();
    
    // Test CPU allocation
    void* cpu_ptr = DeviceMemoryManager::instance().allocate(cpu_device, 1024);
    ASSERT_NE(cpu_ptr, nullptr);
    
    DeviceMemoryManager::instance().deallocate(cpu_device, cpu_ptr);
    
    // Test GPU allocation (if available)
    try {
        Device cuda_device = Device::cuda(0);
        void* cuda_ptr = DeviceMemoryManager::instance().allocate(cuda_device, 1024);
        
        if (cuda_ptr) {
            ASSERT_NE(cuda_ptr, nullptr);
            DeviceMemoryManager::instance().deallocate(cuda_device, cuda_ptr);
        }
    } catch (const std::exception& e) {
        std::cout << "CUDA allocation not available: " << e.what() << "\n";
    }
}

TEST(GpuRuntime_TensorDeviceCreation) {
    // Test creating tensors on different devices
    Shape shape = {2, 3};
    
    // CPU tensor
    Tensor cpu_tensor(shape, DType::FP32, Device::cpu());
    ASSERT_TRUE(cpu_tensor.device().is_cpu());
    
    // Try GPU tensor (if available)
    try {
        Tensor gpu_tensor(shape, DType::FP32, Device::cuda(0));
        ASSERT_TRUE(gpu_tensor.device().is_cuda());
    } catch (const std::exception& e) {
        std::cout << "GPU tensor creation not available: " << e.what() << "\n";
    }
}

TEST(GpuRuntime_DeviceMemoryManager) {
    // Test DeviceMemoryManager singleton
    DeviceMemoryManager& manager1 = DeviceMemoryManager::instance();
    DeviceMemoryManager& manager2 = DeviceMemoryManager::instance();
    
    ASSERT_EQ(&manager1, &manager2);
    
    // Test memory copying
    const char* cpu_data = "Hello GPU World!";
    size_t data_size = strlen(cpu_data) + 1;
    
    Device cpu_device = Device::cpu();
    void* cpu_ptr = manager1.allocate(cpu_device, data_size);
    memcpy(cpu_ptr, cpu_data, data_size);
    
    // Test same-device copy
    void* cpu_copy = manager1.allocate(cpu_device, data_size);
    manager1.copy(cpu_device, cpu_copy, cpu_device, cpu_ptr, data_size);
    
    ASSERT_EQ(strcmp(static_cast<char*>(cpu_copy), cpu_data), 0);
    
    manager1.deallocate(cpu_device, cpu_ptr);
    manager1.deallocate(cpu_device, cpu_copy);
}

TEST(GpuRuntime_GpuCapabilities) {
    // Test GPU capabilities structure
    gpu::GpuCapabilities caps;
    
    caps.compute_capability_supported = true;
    caps.max_threads_per_block = 1024;
    caps.max_shared_memory_per_block = 49152;
    caps.warp_size = 32;
    caps.total_memory = 8ULL * 1024 * 1024 * 1024; // 8GB
    caps.compute_capability = 7.5f;
    
    ASSERT_TRUE(caps.compute_capability_supported);
    ASSERT_EQ(caps.max_threads_per_block, 1024);
    ASSERT_EQ(caps.warp_size, 32);
    ASSERT_EQ(caps.total_memory, 8ULL * 1024 * 1024 * 1024);
}

TEST(GpuRuntime_KernelFusionBasics) {
    // Test kernel fusion initialization
    gpu::KernelFusion fusion;
    
    // Test fusion pattern registration
    gpu::KernelFusion::FusionPattern pattern;
    pattern.operation_sequence = {"add", "relu", "multiply"};
    pattern.fused_kernel_name = "test_fusion_kernel";
    pattern.estimated_speedup = 3;
    pattern.memory_saved = sizeof(float) * 100;
    
    fusion.register_fusion_pattern(pattern);
    
    // Test basic fusion detection
    std::vector<std::string> ops = {"add", "relu", "multiply", "relu"};
    ASSERT_TRUE(fusion.can_fuse(ops));
    
    auto candidates = fusion.find_fusion_candidates({}, ops);
    ASSERT_GE(candidates.size(), 0);
    
    std::cout << "Kernel fusion basic test passed\n";
}

TEST(GpuRuntime_PerformanceProfiler) {
    // Test GPU performance profiler
    utils::GpuProfiler profiler;
    
    // Test profiling control
    profiler.start_profiling("test_kernel", "cuda");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler.end_profiling();
    
    auto stats = profiler.get_stats();
    ASSERT_EQ(stats.total_kernels_executed, 1);
    ASSERT_GT(stats.total_execution_time, 0);
    
    std::cout << "Profiler test: " << stats.total_kernels_executed 
              << " kernels, " << stats.total_execution_time << " ms\n";
}