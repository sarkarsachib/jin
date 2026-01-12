#include "test_runner.cpp"
#include "../src/core/tensor.h"
#include "../src/core/math_ops.h"

using namespace jin;
using namespace jin::test;

TEST(ops_add) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(2.0f);
    b.fill(3.0f);
    
    Tensor c = ops::add(a, b);
    float* data = c.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(data[i], 5.0f);
    }
}

TEST(ops_multiply) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(2.0f);
    b.fill(3.0f);
    
    Tensor c = ops::multiply(a, b);
    float* data = c.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(data[i], 6.0f);
    }
}

TEST(ops_subtract) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(5.0f);
    b.fill(3.0f);
    
    Tensor c = ops::subtract(a, b);
    float* data = c.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(data[i], 2.0f);
    }
}

TEST(ops_divide) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(6.0f);
    b.fill(2.0f);
    
    Tensor c = ops::divide(a, b);
    float* data = c.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(data[i], 3.0f);
    }
}

TEST(ops_exp) {
    Tensor a({2, 2});
    a.fill(0.0f);
    
    Tensor b = ops::exp(a);
    float* data = b.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(data[i], 1.0f, 1e-5f);
    }
}

TEST(ops_log) {
    Tensor a({2, 2});
    a.fill(1.0f);
    
    Tensor b = ops::log(a);
    float* data = b.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(data[i], 0.0f, 1e-5f);
    }
}

TEST(ops_relu) {
    Tensor a({4});
    float* a_data = a.data_ptr<float>();
    a_data[0] = -2.0f;
    a_data[1] = -1.0f;
    a_data[2] = 1.0f;
    a_data[3] = 2.0f;
    
    Tensor b = ops::relu(a);
    float* b_data = b.data_ptr<float>();
    
    ASSERT_EQ(b_data[0], 0.0f);
    ASSERT_EQ(b_data[1], 0.0f);
    ASSERT_EQ(b_data[2], 1.0f);
    ASSERT_EQ(b_data[3], 2.0f);
}

TEST(ops_tanh) {
    Tensor a({2, 2});
    a.fill(0.0f);
    
    Tensor b = ops::tanh(a);
    float* data = b.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(data[i], 0.0f, 1e-5f);
    }
}

TEST(ops_sigmoid) {
    Tensor a({2, 2});
    a.fill(0.0f);
    
    Tensor b = ops::sigmoid(a);
    float* data = b.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(data[i], 0.5f, 1e-5f);
    }
}

TEST(ops_matmul) {
    Tensor a({2, 3});
    Tensor b({3, 2});
    
    float* a_data = a.data_ptr<float>();
    float* b_data = b.data_ptr<float>();
    
    for (size_t i = 0; i < 6; ++i) {
        a_data[i] = 1.0f;
        b_data[i] = 1.0f;
    }
    
    Tensor c = ops::matmul(a, b);
    ASSERT_EQ(c.shape()[0], 2);
    ASSERT_EQ(c.shape()[1], 2);
    
    float* c_data = c.data_ptr<float>();
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(c_data[i], 3.0f);
    }
}

TEST(ops_sum_all) {
    Tensor a({2, 3});
    a.fill(2.0f);
    
    Tensor b = ops::sum(a);
    ASSERT_EQ(b.size(), 1);
    ASSERT_EQ(b.data_ptr<float>()[0], 12.0f);
}

TEST(ops_sum_axis) {
    Tensor a({2, 3});
    a.fill(2.0f);
    
    Tensor b = ops::sum(a, 0, true);
    ASSERT_EQ(b.shape()[0], 1);
    ASSERT_EQ(b.shape()[1], 3);
}

TEST(ops_mean) {
    Tensor a({2, 3});
    a.fill(6.0f);
    
    Tensor b = ops::mean(a);
    ASSERT_EQ(b.size(), 1);
    ASSERT_EQ(b.data_ptr<float>()[0], 6.0f);
}
