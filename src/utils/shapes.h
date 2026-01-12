#pragma once

#include <vector>
#include <cstddef>
#include <string>

namespace jin {

using Shape = std::vector<size_t>;

size_t compute_size(const Shape& shape);

Shape compute_strides(const Shape& shape);

Shape broadcast_shapes(const Shape& a, const Shape& b);

bool can_broadcast(const Shape& a, const Shape& b);

std::string shape_to_string(const Shape& shape);

Shape infer_shape_after_transpose(const Shape& shape, const std::vector<int>& axes);

Shape infer_shape_after_sum(const Shape& shape, int axis, bool keepdims);

} // namespace jin
