#include "tensor.h"
#include "math_ops.h"
#include "../utils/logger.h"
#include <cstring>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>

namespace jin {

// TensorStorage implementation
TensorStorage::TensorStorage(size_t size, DType dtype, Device device, Allocator* allocator)
    : size_(size), dtype_(dtype), device_(device), 
      allocator_(allocator ? allocator : get_default_allocator()) {
    if (size > 0) {
        data_ = allocator_->allocate(size * dtype_size(dtype));
    } else {
        data_ = nullptr;
    }
}

TensorStorage::~TensorStorage() {
    if (data_) {
        allocator_->deallocate(data_);
    }
}

std::shared_ptr<TensorStorage> TensorStorage::clone() const {
    auto new_storage = std::make_shared<TensorStorage>(size_, dtype_, device_, allocator_);
    if (data_ && new_storage->data_) {
        std::memcpy(new_storage->data_, data_, size_ * dtype_size(dtype_));
    }
    return new_storage;
}

// Tensor implementation
Tensor::Tensor() : offset_(0), requires_grad_(false) {}

Tensor::Tensor(const Shape& shape, DType dtype, Device device)
    : shape_(shape), 
      strides_(compute_strides(shape)),
      offset_(0),
      requires_grad_(false) {
    size_t total_size = compute_size(shape);
    if (total_size > 0) {
        storage_ = std::make_shared<TensorStorage>(total_size, dtype, device);
    }
}

Tensor::Tensor(const Shape& shape, std::shared_ptr<TensorStorage> storage,
               const Shape& strides, size_t offset)
    : shape_(shape),
      strides_(strides),
      storage_(storage),
      offset_(offset),
      requires_grad_(false) {}

void* Tensor::data() {
    if (!storage_) return nullptr;
    uint8_t* base = static_cast<uint8_t*>(storage_->data());
    return base + offset_ * dtype_size(storage_->dtype());
}

const void* Tensor::data() const {
    if (!storage_) return nullptr;
    const uint8_t* base = static_cast<const uint8_t*>(storage_->data());
    return base + offset_ * dtype_size(storage_->dtype());
}

float Tensor::item() const {
    if (size() != 1) {
        throw std::runtime_error("item() can only be called on scalar tensors");
    }
    
    const float* ptr = data_ptr<float>();
    return ptr ? *ptr : 0.0f;
}

void Tensor::fill(float value) {
    if (!storage_) return;
    
    float* ptr = data_ptr<float>();
    if (ptr) {
        for (size_t i = 0; i < size(); ++i) {
            ptr[i] = value;
        }
    }
}

void Tensor::fill_random(float min, float max) {
    if (!storage_) return;
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    
    float* ptr = data_ptr<float>();
    if (ptr) {
        for (size_t i = 0; i < size(); ++i) {
            ptr[i] = dist(gen);
        }
    }
}

Tensor Tensor::grad() const {
    if (grad_) {
        return *grad_;
    }
    return Tensor();
}

void Tensor::set_grad(const Tensor& grad) {
    grad_ = std::make_shared<Tensor>(grad);
}

bool Tensor::has_grad() const {
    return grad_ != nullptr && grad_->storage_ != nullptr;
}

bool Tensor::is_contiguous() const {
    if (shape_.empty()) return true;
    
    Shape expected_strides = compute_strides(shape_);
    return strides_ == expected_strides;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }
    
    Tensor result(shape_, dtype(), device());
    
    // Copy data in contiguous order
    const float* src = data_ptr<float>();
    float* dst = result.data_ptr<float>();
    
    // For now, only support simple cases
    // Full strided copy would need recursive implementation
    if (src && dst) {
        std::memcpy(dst, src, size() * dtype_size(dtype()));
    }
    
    return result;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    size_t new_size = compute_size(new_shape);
    if (new_size != size()) {
        throw std::invalid_argument("New shape must have same number of elements");
    }
    
    if (!is_contiguous()) {
        return contiguous().reshape(new_shape);
    }
    
    Shape new_strides = compute_strides(new_shape);
    return Tensor(new_shape, storage_, new_strides, offset_);
}

Tensor Tensor::transpose(const std::vector<int>& axes) const {
    Shape new_shape = infer_shape_after_transpose(shape_, axes);
    Shape new_strides(strides_.size());
    
    if (axes.empty()) {
        // Default: reverse all dimensions
        for (size_t i = 0; i < strides_.size(); ++i) {
            new_strides[i] = strides_[strides_.size() - 1 - i];
        }
    } else {
        for (size_t i = 0; i < axes.size(); ++i) {
            int axis = axes[i];
            if (axis < 0) axis += shape_.size();
            new_strides[i] = strides_[axis];
        }
    }
    
    return Tensor(new_shape, storage_, new_strides, offset_);
}

Tensor Tensor::view(const Shape& new_shape) const {
    return reshape(new_shape);
}

// Arithmetic operations
Tensor Tensor::operator+(const Tensor& other) const {
    return jin::ops::add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return jin::ops::subtract(*this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return jin::ops::multiply(*this, other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return jin::ops::divide(*this, other);
}

Tensor Tensor::operator+(float scalar) const {
    Tensor scalar_tensor({1}, dtype(), device());
    scalar_tensor.fill(scalar);
    return jin::ops::add(*this, scalar_tensor);
}

Tensor Tensor::operator-(float scalar) const {
    Tensor scalar_tensor({1}, dtype(), device());
    scalar_tensor.fill(scalar);
    return jin::ops::subtract(*this, scalar_tensor);
}

Tensor Tensor::operator*(float scalar) const {
    Tensor scalar_tensor({1}, dtype(), device());
    scalar_tensor.fill(scalar);
    return jin::ops::multiply(*this, scalar_tensor);
}

Tensor Tensor::operator/(float scalar) const {
    Tensor scalar_tensor({1}, dtype(), device());
    scalar_tensor.fill(scalar);
    return jin::ops::divide(*this, scalar_tensor);
}

Tensor Tensor::pow(float exponent) const {
    return jin::ops::pow(*this, exponent);
}

Tensor Tensor::sqrt() const {
    return jin::ops::sqrt(*this);
}

Tensor Tensor::abs() const {
    return jin::ops::abs(*this);
}

Tensor Tensor::slice(size_t start, size_t end) const {
    if (start >= size() || end > size() || start >= end) {
        return Tensor();  // Return empty tensor for invalid slice
    }
    
    size_t slice_size = end - start;
    Shape new_shape({slice_size});
    
    // Create a new tensor view
    return Tensor(new_shape, storage_, compute_strides(new_shape), offset_ + start);
}

std::string Tensor::to_string() const {
    std::ostringstream oss;
    oss << "Tensor(shape=" << shape_to_string(shape_)
        << ", dtype=" << dtype_name(dtype())
        << ", device=" << device().to_string()
        << ", requires_grad=" << (requires_grad_ ? "true" : "false")
        << ")";
    return oss.str();
}

Tensor Tensor::clone() const {
    // Create a new tensor with the same data
    Tensor cloned(shape_, dtype(), device());
    
    // Copy the data
    if (storage_ && cloned.storage_) {
        std::memcpy(cloned.data(), data(), size() * dtype_size(dtype()));
    }
    
    // Copy gradient if it exists
    if (grad_) {
        cloned.grad_ = std::make_shared<Tensor>(grad_->clone());
    }
    
    // Copy other properties
    cloned.requires_grad_ = requires_grad_;
    
    return cloned;
}

// Factory functions
Tensor zeros(const Shape& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    t.fill(0.0f);
    return t;
}

Tensor ones(const Shape& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    t.fill(1.0f);
    return t;
}

Tensor randn(const Shape& shape, DType dtype, Device device) {
    Tensor t(shape, dtype, device);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    float* ptr = t.data_ptr<float>();
    if (ptr) {
        for (size_t i = 0; i < t.size(); ++i) {
            ptr[i] = dist(gen);
        }
    }
    
    return t;
}

} // namespace jin
