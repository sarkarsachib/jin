#include "frame_state.h"
#include "../core/tensor.h"
#include "../core/math_ops.h"
#include <cmath>

namespace jin {
namespace temporal {

Tensor concatenate(const std::vector<Tensor>& tensors, int dim) {
    // Simple concatenation implementation
    if (tensors.empty()) return Tensor();
    
    size_t total_size = 0;
    for (const auto& tensor : tensors) {
        total_size += tensor.size();
    }
    
    Tensor result({static_cast<int>(total_size)});
    size_t offset = 0;
    
    for (const auto& tensor : tensors) {
        size_t tensor_size = tensor.size() * sizeof(float);
        std::memcpy(
            static_cast<char*>(result.data()) + offset,
            tensor.data(),
            tensor_size
        );
        offset += tensor.size();
    }
    
    return result;
}

} // namespace temporal
} // namespace jin