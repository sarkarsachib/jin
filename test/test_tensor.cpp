#include "test_runner.cpp"
#include "../src/core/tensor.h"

using namespace jin;
using namespace jin::test;

TEST(tensor_creation) {
    Tensor t({2, 3}, DType::FP32);
    ASSERT_EQ(t.shape()[0], 2);
    ASSERT_EQ(t.shape()[1], 3);
    ASSERT_EQ(t.size(), 6);
    ASSERT_EQ(t.ndim(), 2);
}

TEST(tensor_zeros) {
    Tensor t = zeros({2, 2});
    t.fill(0.0f);
    float* data = t.data_ptr<float>();
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(data[i], 0.0f);
    }
}

TEST(tensor_ones) {
    Tensor t = ones({3, 3});
    float* data = t.data_ptr<float>();
    for (size_t i = 0; i < 9; ++i) {
        ASSERT_EQ(data[i], 1.0f);
    }
}

TEST(tensor_fill) {
    Tensor t({2, 2});
    t.fill(5.0f);
    float* data = t.data_ptr<float>();
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(data[i], 5.0f);
    }
}

TEST(tensor_reshape) {
    Tensor t({2, 3});
    Tensor reshaped = t.reshape({3, 2});
    ASSERT_EQ(reshaped.shape()[0], 3);
    ASSERT_EQ(reshaped.shape()[1], 2);
    ASSERT_EQ(reshaped.size(), 6);
}

TEST(tensor_transpose) {
    Tensor t({2, 3});
    t.fill(0.0f);
    float* data = t.data_ptr<float>();
    for (size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    Tensor transposed = t.transpose({1, 0});
    ASSERT_EQ(transposed.shape()[0], 3);
    ASSERT_EQ(transposed.shape()[1], 2);
}

TEST(tensor_requires_grad) {
    Tensor t({2, 2});
    ASSERT_FALSE(t.requires_grad());
    t.set_requires_grad(true);
    ASSERT_TRUE(t.requires_grad());
}

TEST(tensor_device) {
    Tensor t({2, 2}, DType::FP32, Device::cpu());
    ASSERT_TRUE(t.device().is_cpu());
    ASSERT_FALSE(t.device().is_gpu());
}

TEST(tensor_dtype) {
    Tensor t({2, 2}, DType::FP32);
    ASSERT_EQ(t.dtype(), DType::FP32);
}

TEST(tensor_contiguous) {
    Tensor t({2, 3});
    ASSERT_TRUE(t.is_contiguous());
    
    Tensor transposed = t.transpose();
    Tensor cont = transposed.contiguous();
    ASSERT_TRUE(cont.is_contiguous());
}
