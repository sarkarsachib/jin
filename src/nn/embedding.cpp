#include "embedding.h"
#include "../autodiff/ops.h"
#include "../autodiff/tape.h"
#include <algorithm>
#include <cmath>

namespace jin {
namespace nn {

Embedding::Embedding(size_t num_embeddings, size_t embedding_dim, Initializer init)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      normalize_(false) {
    
    weight_ = initialize_tensor({num_embeddings, embedding_dim}, init);
    weight_.set_requires_grad(true);
}

Tensor Embedding::forward(const Tensor& indices) {
    autodiff::TapeGuard tape_guard;
    
    Shape indices_shape = indices.shape();
    size_t batch_size = 1;
    for (size_t i = 0; i < indices_shape.size(); ++i) {
        batch_size *= indices_shape[i];
    }
    
    // Output shape: [..., embedding_dim]
    Shape output_shape = indices_shape;
    output_shape.push_back(embedding_dim_);
    
    Tensor output(output_shape, DType::FP32, Device::cpu());
    const int64_t* idx_data = indices.data_ptr<int64_t>();
    const float* weight_data = weight_.data_ptr<float>();
    float* out_data = output.data_ptr<float>();
    
    for (size_t i = 0; i < batch_size; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0 || idx >= static_cast<int64_t>(num_embeddings_)) {
            throw std::out_of_range("Embedding index out of range");
        }
        
        for (size_t d = 0; d < embedding_dim_; ++d) {
            out_data[i * embedding_dim_ + d] = weight_data[idx * embedding_dim_ + d];
        }
    }
    
    if (normalize_) {
        // L2 normalize each embedding vector
        for (size_t i = 0; i < batch_size; ++i) {
            float norm = 0.0f;
            for (size_t d = 0; d < embedding_dim_; ++d) {
                norm += out_data[i * embedding_dim_ + d] * out_data[i * embedding_dim_ + d];
            }
            norm = std::sqrt(norm) + 1e-8f;
            
            for (size_t d = 0; d < embedding_dim_; ++d) {
                out_data[i * embedding_dim_ + d] /= norm;
            }
        }
    }
    
    return output;
}

} // namespace nn
} // namespace jin
