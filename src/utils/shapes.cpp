#include "shapes.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace jin {

size_t compute_size(const Shape& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
}

Shape compute_strides(const Shape& shape) {
    if (shape.empty()) return {};
    
    Shape strides(shape.size());
    strides[shape.size() - 1] = 1;
    
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    return strides;
}

bool can_broadcast(const Shape& a, const Shape& b) {
    size_t max_ndim = std::max(a.size(), b.size());
    
    for (size_t i = 0; i < max_ndim; ++i) {
        size_t idx_a = a.size() - 1 - i;
        size_t idx_b = b.size() - 1 - i;
        
        size_t dim_a = (i < a.size()) ? a[idx_a] : 1;
        size_t dim_b = (i < b.size()) ? b[idx_b] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    
    return true;
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
    if (!can_broadcast(a, b)) {
        throw std::invalid_argument("Shapes cannot be broadcast together");
    }
    
    size_t max_ndim = std::max(a.size(), b.size());
    Shape result(max_ndim);
    
    for (size_t i = 0; i < max_ndim; ++i) {
        size_t idx_a = a.size() - 1 - i;
        size_t idx_b = b.size() - 1 - i;
        
        size_t dim_a = (i < a.size()) ? a[idx_a] : 1;
        size_t dim_b = (i < b.size()) ? b[idx_b] : 1;
        
        result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
    }
    
    return result;
}

std::string shape_to_string(const Shape& shape) {
    if (shape.empty()) return "()";
    
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << ")";
    return oss.str();
}

Shape infer_shape_after_transpose(const Shape& shape, const std::vector<int>& axes) {
    if (axes.empty()) {
        // Default: reverse all dimensions
        Shape result(shape.rbegin(), shape.rend());
        return result;
    }
    
    if (axes.size() != shape.size()) {
        throw std::invalid_argument("Number of axes must match number of dimensions");
    }
    
    Shape result(shape.size());
    for (size_t i = 0; i < axes.size(); ++i) {
        int axis = axes[i];
        if (axis < 0) axis += shape.size();
        if (axis < 0 || axis >= static_cast<int>(shape.size())) {
            throw std::out_of_range("Axis out of range");
        }
        result[i] = shape[axis];
    }
    
    return result;
}

Shape infer_shape_after_sum(const Shape& shape, int axis, bool keepdims) {
    if (axis < 0) axis += shape.size();
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
        throw std::out_of_range("Axis out of range");
    }
    
    if (keepdims) {
        Shape result = shape;
        result[axis] = 1;
        return result;
    } else {
        Shape result;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int>(i) != axis) {
                result.push_back(shape[i]);
            }
        }
        return result;
    }
}

} // namespace jin
