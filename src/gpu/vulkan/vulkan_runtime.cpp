#include "vulkan_runtime.h"
#include "../../gpu/gpu_common.h"
#include <iostream>
#include <stdexcept>

namespace jin {
namespace gpu {
namespace vulkan {

// Global Vulkan runtime instance
std::unique_ptr<VulkanRuntime>& get_global_vulkan_runtime() {
    static std::unique_ptr<VulkanRuntime> instance;
    return instance;
}

} // namespace vulkan
} // namespace gpu
} // namespace jin