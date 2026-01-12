#pragma once

#include "dtype.h"
#include "device.h"
#include "allocator.h"
#include "../utils/shapes.h"
#include <memory>
#include <vector>
#include <string>

namespace jin {

class TensorStorage {
public:
    TensorStorage(size_t size, DType dtype, Device device, Allocator* allocator = nullptr);
    ~TensorStorage();
    
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    
    std::shared_ptr<TensorStorage> clone() const;
    
private:
    void* data_;
    size_t size_;
    DType dtype_;
    Device device_;
    Allocator* allocator_;
};

class Tensor {
public:
    Tensor();
    Tensor(const Shape& shape, DType dtype = DType::FP32, Device device = Device::cpu());
    Tensor(const Shape& shape, std::shared_ptr<TensorStorage> storage, 
           const Shape& strides, size_t offset = 0);
    
    // Properties
    const Shape& shape() const { return shape_; }
    const Shape& strides() const { return strides_; }
    DType dtype() const { return storage_ ? storage_->dtype() : DType::FP32; }
    Device device() const { return storage_ ? storage_->device() : Device::cpu(); }
    size_t ndim() const { return shape_.size(); }
    size_t size() const { return compute_size(shape_); }
    bool requires_grad() const { return requires_grad_; }
    size_t offset() const { return offset_; }
    
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }
    
    // Data access
    void* data();
    const void* data() const;
    
    template<typename T>
    T* data_ptr() { return static_cast<T*>(data()); }
    
    template<typename T>
    const T* data_ptr() const { return static_cast<const T*>(data()); }
    
    // Get/set scalar values (for testing)
    float item() const;
    void fill(float value);
    void fill_random(float min = 0.0f, float max = 1.0f);
    
    // Views and operations
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(const std::vector<int>& axes = {}) const;
    Tensor view(const Shape& new_shape) const;
    
    // Gradient
    Tensor grad() const;
    void set_grad(const Tensor& grad);
    bool has_grad() const;
    
    // Storage management
    bool is_contiguous() const;
    Tensor contiguous() const;
    std::shared_ptr<TensorStorage> storage() const { return storage_; }
    
    // Utility
    std::string to_string() const;
    
private:
    Shape shape_;
    Shape strides_;
    std::shared_ptr<TensorStorage> storage_;
    size_t offset_;
    bool requires_grad_;
    std::shared_ptr<Tensor> grad_;
    
    void allocate_storage();
};

// Factory functions
Tensor zeros(const Shape& shape, DType dtype = DType::FP32, Device device = Device::cpu());
Tensor ones(const Shape& shape, DType dtype = DType::FP32, Device device = Device::cpu());
Tensor randn(const Shape& shape, DType dtype = DType::FP32, Device device = Device::cpu());

} // namespace jin
