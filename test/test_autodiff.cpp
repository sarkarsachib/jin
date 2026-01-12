#include "test_runner.cpp"
#include "../src/core/tensor.h"
#include "../src/autodiff/ops.h"
#include "../src/autodiff/tape.h"
#include "../src/autodiff/backward.h"
#include "../src/autodiff/grad_registry.h"

using namespace jin;
using namespace jin::test;

TEST(autodiff_add_backward) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
    autodiff::GradientTape::instance().enable();
    
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(2.0f);
    b.fill(3.0f);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    
    Tensor c = autodiff::add(a, b);
    
    Tensor grad_out = ones({2, 2});
    autodiff::backward(c, &grad_out);
    
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(a));
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(b));
    
    autodiff::GradientTape::instance().disable();
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
}

TEST(autodiff_multiply_backward) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
    autodiff::GradientTape::instance().enable();
    
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(2.0f);
    b.fill(3.0f);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    
    Tensor c = autodiff::multiply(a, b);
    
    Tensor grad_out = ones({2, 2});
    autodiff::backward(c, &grad_out);
    
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(a));
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(b));
    
    Tensor grad_a = autodiff::GradientRegistry::instance().get_grad(a);
    Tensor grad_b = autodiff::GradientRegistry::instance().get_grad(b);
    float* grad_a_data = grad_a.data_ptr<float>();
    float* grad_b_data = grad_b.data_ptr<float>();
    
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(grad_a_data[i], 3.0f);
        ASSERT_EQ(grad_b_data[i], 2.0f);
    }
    
    autodiff::GradientTape::instance().disable();
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
}

TEST(autodiff_exp_backward) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
    autodiff::GradientTape::instance().enable();
    
    Tensor a({2, 2});
    a.fill(0.0f);
    a.set_requires_grad(true);
    
    Tensor b = autodiff::exp(a);
    
    Tensor grad_out = ones({2, 2});
    autodiff::backward(b, &grad_out);
    
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(a));
    
    Tensor grad_a = autodiff::GradientRegistry::instance().get_grad(a);
    float* grad_a_data = grad_a.data_ptr<float>();
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(grad_a_data[i], 1.0f, 1e-5f);
    }
    
    autodiff::GradientTape::instance().disable();
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
}

TEST(autodiff_relu_backward) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
    autodiff::GradientTape::instance().enable();
    
    Tensor a({4});
    float* a_data = a.data_ptr<float>();
    a_data[0] = -2.0f;
    a_data[1] = -1.0f;
    a_data[2] = 1.0f;
    a_data[3] = 2.0f;
    a.set_requires_grad(true);
    
    Tensor b = autodiff::relu(a);
    
    Tensor grad_out = ones({4});
    autodiff::backward(b, &grad_out);
    
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(a));
    
    Tensor grad_a = autodiff::GradientRegistry::instance().get_grad(a);
    float* grad_a_data = grad_a.data_ptr<float>();
    ASSERT_EQ(grad_a_data[0], 0.0f);
    ASSERT_EQ(grad_a_data[1], 0.0f);
    ASSERT_EQ(grad_a_data[2], 1.0f);
    ASSERT_EQ(grad_a_data[3], 1.0f);
    
    autodiff::GradientTape::instance().disable();
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
}

TEST(autodiff_matmul_backward) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
    autodiff::GradientTape::instance().enable();
    
    Tensor a({2, 3});
    Tensor b({3, 2});
    a.fill(1.0f);
    b.fill(1.0f);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    
    Tensor c = autodiff::matmul(a, b);
    
    Tensor grad_out = ones({2, 2});
    autodiff::backward(c, &grad_out);
    
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(a));
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(b));
    
    autodiff::GradientTape::instance().disable();
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
}

TEST(autodiff_chain_rule) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
    autodiff::GradientTape::instance().enable();
    
    Tensor a({2, 2});
    a.fill(2.0f);
    a.set_requires_grad(true);
    
    Tensor b = autodiff::multiply(a, a);
    Tensor c = autodiff::add(b, b);
    
    Tensor grad_out = ones({2, 2});
    autodiff::backward(c, &grad_out);
    
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(a));
    
    Tensor grad_a = autodiff::GradientRegistry::instance().get_grad(a);
    float* grad_a_data = grad_a.data_ptr<float>();
    for (size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(grad_a_data[i], 8.0f);
    }
    
    autodiff::GradientTape::instance().disable();
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();
}

TEST(autodiff_tape_recording) {
    autodiff::GradientTape::instance().clear();
    ASSERT_EQ(autodiff::GradientTape::instance().nodes().size(), 0);
    
    autodiff::GradientTape::instance().enable();
    
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.set_requires_grad(true);
    
    Tensor c = autodiff::add(a, b);
    
    ASSERT_EQ(autodiff::GradientTape::instance().nodes().size(), 1);
    
    autodiff::GradientTape::instance().disable();
    autodiff::GradientTape::instance().clear();
}
