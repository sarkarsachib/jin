#include "metal_runtime.h"
#include "../../gpu/gpu_common.h"
#include <iostream>
#include <stdexcept>

namespace jin {
namespace gpu {
namespace metal {

// Global Metal runtime instance
std::unique_ptr<MetalRuntime>& get_global_metal_runtime() {
    static std::unique_ptr<MetalRuntime> instance;
    return instance;
}

} // namespace metal
} // namespace gpu
} // namespace jin