#include "graph.h"

namespace jin {
namespace autodiff {

GraphNode::GraphNode(const std::vector<Tensor>& inputs, const Tensor& output,
                     BackwardFunc backward_fn, const std::string& op_name)
    : inputs_(inputs), output_(output), backward_fn_(backward_fn), op_name_(op_name) {
}

void GraphNode::compute_gradients(const Tensor& grad_output) {
    if (backward_fn_) {
        std::vector<Tensor> input_grads = backward_fn_(grad_output);
        
        // Accumulate gradients for each input
        for (size_t i = 0; i < inputs_.size() && i < input_grads.size(); ++i) {
            if (inputs_[i].requires_grad()) {
                if (inputs_[i].has_grad()) {
                    // Accumulate gradient
                    const float* new_grad = input_grads[i].data_ptr<float>();
                    float* existing_grad = inputs_[i].grad().data_ptr<float>();
                    
                    for (size_t j = 0; j < input_grads[i].size(); ++j) {
                        existing_grad[j] += new_grad[j];
                    }
                } else {
                    // Set initial gradient
                    inputs_[i].set_grad(input_grads[i]);
                }
            }
        }
    }
}

} // namespace autodiff
} // namespace jin
