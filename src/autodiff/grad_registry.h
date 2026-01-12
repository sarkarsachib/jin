#pragma once

#include "../core/tensor.h"
#include <unordered_map>
#include <memory>
#include <mutex>

namespace jin {
namespace autodiff {

class GradientRegistry {
public:
    static GradientRegistry& instance();
    
    void set_grad(const Tensor& tensor, const Tensor& grad);
    Tensor get_grad(const Tensor& tensor) const;
    bool has_grad(const Tensor& tensor) const;
    void clear_grad(const Tensor& tensor);
    void clear_all();
    
private:
    GradientRegistry() = default;
    std::unordered_map<void*, Tensor> gradients_;
    mutable std::mutex mutex_;
};

} // namespace autodiff
} // namespace jin
