#include "backward.h"
#include "tape.h"
#include "grad_registry.h"
#include "../utils/logger.h"
#include <unordered_map>

namespace jin {
namespace autodiff {

void backward(Tensor& output, const Tensor* grad_output) {
    auto& tape = GradientTape::instance();
    auto& nodes = tape.nodes();
    auto& grad_registry = GradientRegistry::instance();
    
    if (nodes.empty()) {
        warn("No operations recorded on tape");
        return;
    }
    
    // Use a map to track gradients by storage pointer
    std::unordered_map<void*, Tensor> grad_map;
    
    // Initialize gradient for output
    Tensor grad;
    if (grad_output) {
        grad = *grad_output;
    } else {
        // Default gradient is ones with same shape as output
        if (output.size() == 1) {
            grad = ones({1}, output.dtype(), output.device());
        } else {
            grad = ones(output.shape(), output.dtype(), output.device());
        }
    }
    
    // Store gradient for output
    if (output.storage()) {
        grad_map[output.storage().get()] = grad;
    }
    
    // Traverse computation graph in reverse order
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        auto& node = *it;
        
        // Get gradient of this node's output from the map
        auto node_storage = node->output().storage().get();
        if (grad_map.find(node_storage) != grad_map.end()) {
            Tensor output_grad = grad_map[node_storage];
            
            // Compute gradients for inputs
            std::vector<Tensor> input_grads = node->backward_fn()(output_grad);
            
            // Accumulate gradients for each input
            auto& inputs = node->inputs();
            for (size_t i = 0; i < inputs.size() && i < input_grads.size(); ++i) {
                if (inputs[i].requires_grad() && inputs[i].storage()) {
                    auto input_storage = inputs[i].storage().get();
                    
                    if (grad_map.find(input_storage) != grad_map.end()) {
                        // Accumulate gradient
                        Tensor& existing_grad = grad_map[input_storage];
                        const float* new_grad_data = input_grads[i].data_ptr<float>();
                        float* existing_grad_data = existing_grad.data_ptr<float>();
                        
                        for (size_t j = 0; j < existing_grad.size(); ++j) {
                            existing_grad_data[j] += new_grad_data[j];
                        }
                    } else {
                        // Set initial gradient
                        grad_map[input_storage] = input_grads[i];
                    }
                    
                    // Store gradient in registry (accessible by all tensor copies with same storage)
                    grad_registry.set_grad(inputs[i], grad_map[input_storage]);
                }
            }
        }
    }
    
    grad_registry.set_grad(output, grad);
}

void zero_grad(Tensor& tensor) {
    if (tensor.has_grad()) {
        tensor.set_grad(zeros(tensor.shape(), tensor.dtype(), tensor.device()));
    }
}

} // namespace autodiff
} // namespace jin
