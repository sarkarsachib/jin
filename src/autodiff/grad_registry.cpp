#include "grad_registry.h"

namespace jin {
namespace autodiff {

GradientRegistry& GradientRegistry::instance() {
    static GradientRegistry registry;
    return registry;
}

void GradientRegistry::set_grad(const Tensor& tensor, const Tensor& grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tensor.storage()) {
        gradients_[tensor.storage().get()] = grad;
    }
}

Tensor GradientRegistry::get_grad(const Tensor& tensor) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tensor.storage()) {
        auto it = gradients_.find(tensor.storage().get());
        if (it != gradients_.end()) {
            return it->second;
        }
    }
    return Tensor();
}

bool GradientRegistry::has_grad(const Tensor& tensor) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tensor.storage()) {
        return gradients_.find(tensor.storage().get()) != gradients_.end();
    }
    return false;
}

void GradientRegistry::clear_grad(const Tensor& tensor) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tensor.storage()) {
        gradients_.erase(tensor.storage().get());
    }
}

void GradientRegistry::clear_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    gradients_.clear();
}

} // namespace autodiff
} // namespace jin
