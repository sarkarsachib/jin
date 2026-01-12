#pragma once

#include "../core/tensor.h"
#include "common.h"

namespace jin {
namespace nn {

class Embedding {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim,
              Initializer init = Initializer::normal(0.0f, 1.0f));
    
    Tensor forward(const Tensor& indices);
    
    // Properties
    Tensor weight() const { return weight_; }
    size_t num_embeddings() const { return num_embeddings_; }
    size_t embedding_dim() const { return embedding_dim_; }
    
    // L2 normalization
    void set_normalize(bool normalize) { normalize_ = normalize; }
    bool normalize() const { return normalize_; }

private:
    size_t num_embeddings_;
    size_t embedding_dim_;
    Tensor weight_;
    bool normalize_;
};

} // namespace nn
} // namespace jin
