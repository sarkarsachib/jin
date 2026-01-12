#include "tape.h"

namespace jin {
namespace autodiff {

GradientTape& GradientTape::instance() {
    static GradientTape tape;
    return tape;
}

void GradientTape::record(const std::vector<Tensor>& inputs, const Tensor& output,
                          GraphNode::BackwardFunc backward_fn, const std::string& op_name) {
    if (!enabled_) return;
    
    // Only record if at least one input requires grad
    bool any_requires_grad = false;
    for (const auto& input : inputs) {
        if (input.requires_grad()) {
            any_requires_grad = true;
            break;
        }
    }
    
    if (any_requires_grad) {
        nodes_.push_back(std::make_shared<GraphNode>(inputs, output, backward_fn, op_name));
    }
}

void GradientTape::clear() {
    nodes_.clear();
}

} // namespace autodiff
} // namespace jin
