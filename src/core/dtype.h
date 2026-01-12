#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>
#include <ostream>

namespace jin {

enum class DType {
    FP32,
    FP16,
    BF16,
    INT8,
    INT32,
    BOOL
};

struct DTypeInfo {
    DType dtype;
    size_t size;
    size_t alignment;
    bool is_float;
    std::string name;
};

inline DTypeInfo get_dtype_info(DType dtype) {
    switch (dtype) {
        case DType::FP32:
            return {DType::FP32, 4, 4, true, "fp32"};
        case DType::FP16:
            return {DType::FP16, 2, 2, true, "fp16"};
        case DType::BF16:
            return {DType::BF16, 2, 2, true, "bf16"};
        case DType::INT8:
            return {DType::INT8, 1, 1, false, "int8"};
        case DType::INT32:
            return {DType::INT32, 4, 4, false, "int32"};
        case DType::BOOL:
            return {DType::BOOL, 1, 1, false, "bool"};
        default:
            throw std::runtime_error("Unknown dtype");
    }
}

inline size_t dtype_size(DType dtype) {
    return get_dtype_info(dtype).size;
}

inline bool dtype_is_float(DType dtype) {
    return get_dtype_info(dtype).is_float;
}

inline std::string dtype_name(DType dtype) {
    return get_dtype_info(dtype).name;
}

inline bool can_cast(DType from, DType to) {
    if (from == to) return true;
    
    // Allow upward casting for integers
    if (from == DType::INT8 && to == DType::INT32) return true;
    if (from == DType::BOOL && to == DType::INT32) return true;
    
    // Allow casting to float types
    if (dtype_is_float(to)) return true;
    
    return false;
}

inline std::ostream& operator<<(std::ostream& os, DType dtype) {
    os << dtype_name(dtype);
    return os;
}

} // namespace jin
