#pragma once

#include "../core/tensor.h"
#include <vector>
#include <memory>
#include <functional>
#include <string>

namespace jin {
namespace autodiff {

class GraphNode {
public:
    using BackwardFunc = std::function<std::vector<Tensor>(const Tensor&)>;
    
    GraphNode(const std::vector<Tensor>& inputs, const Tensor& output, 
              BackwardFunc backward_fn, const std::string& op_name);
    
    std::vector<Tensor>& inputs() { return inputs_; }
    const std::vector<Tensor>& inputs() const { return inputs_; }
    const Tensor& output() const { return output_; }
    const BackwardFunc& backward_fn() const { return backward_fn_; }
    const std::string& op_name() const { return op_name_; }
    
    void compute_gradients(const Tensor& grad_output);
    
private:
    std::vector<Tensor> inputs_;
    Tensor output_;
    BackwardFunc backward_fn_;
    std::string op_name_;
};

} // namespace autodiff
} // namespace jin
