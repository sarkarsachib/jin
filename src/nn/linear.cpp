#include "linear.h"
#include "../autodiff/ops.h"
#include "../autodiff/tape.h"
#include "../core/math_ops.h"

namespace jin {
namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias,
               Initializer weight_init, Initializer bias_init)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias),
      mode_(ForwardMode::Train) {
    
    weight_ = initialize_tensor({out_features, in_features}, weight_init);
    weight_.set_requires_grad(true);
    
    if (has_bias_) {
        bias_ = initialize_tensor({out_features}, bias_init);
        bias_.set_requires_grad(true);
    }
}

Tensor Linear::forward(const Tensor& input) {
    autodiff::TapeGuard tape_guard;
    
    // input shape: [..., in_features]
    // weight shape: [out_features, in_features]
    
    // Reshape input to 2D: [batch, ..., in_features] -> [N, in_features]
    Shape input_shape = input.shape();
    size_t batch_size = 1;
    for (size_t i = 0; i < input.ndim() - 1; ++i) {
        batch_size *= input_shape[i];
    }
    
    Shape flat_shape = {batch_size, in_features_};
    Tensor input_flat = input.reshape(flat_shape);
    
    // matmul: [N, in_features] x [in_features, out_features] -> [N, out_features]
    Tensor output_flat = autodiff::matmul(input_flat, weight_);
    
    if (has_bias_) {
        output_flat = autodiff::add(output_flat, bias_);
    }
    
    // Reshape back to [..., out_features]
    Shape output_shape = input_shape;
    output_shape[output_shape.size() - 1] = out_features_;
    
    return output_flat.reshape(output_shape);
}

} // namespace nn
} // namespace jin
