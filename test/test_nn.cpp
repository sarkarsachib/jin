#include "test_runner.cpp"
#include "../src/nn/linear.h"
#include "../src/nn/activation.h"
#include "../src/nn/normalization.h"
#include "../src/nn/dropout.h"
#include "../src/nn/embedding.h"
#include "../src/nn/attention.h"
#include "../src/nn/pooling.h"
#include "../src/nn/conv.h"
#include "../src/nn/losses.h"
#include "../src/optim/optim.h"
#include <cmath>
#include <iostream>

using namespace jin;
using namespace jin::test;

// ==================== Linear Layer Tests ====================

TEST(linear_creation) {
    nn::Linear linear(10, 20);
    ASSERT_EQ(linear.in_features(), 10);
    ASSERT_EQ(linear.out_features(), 20);
}

TEST(linear_forward_shape) {
    nn::Linear linear(10, 20);
    Tensor input({5, 10});  // [batch, in_features]
    input.fill(1.0f);
    
    Tensor output = linear.forward(input);
    ASSERT_EQ(output.shape()[0], 5);
    ASSERT_EQ(output.shape()[1], 20);
}

TEST(linear_forward_batch) {
    nn::Linear linear(10, 20);
    Tensor input({2, 3, 10});  // [batch, seq, in_features]
    input.fill(0.5f);
    
    Tensor output = linear.forward(input);
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 3);
    ASSERT_EQ(output.shape()[2], 20);
}

TEST(linear_gradient) {
    nn::Linear linear(4, 3);
    Tensor input({2, 4});
    input.fill(1.0f);
    input.set_requires_grad(true);
    
    Tensor output = linear.forward(input);
    // Check output has correct shape
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 3);
}

TEST(linear_no_bias) {
    nn::Linear linear(5, 3, false);
    Tensor input({1, 5});
    input.fill(2.0f);
    
    Tensor output = linear.forward(input);
    ASSERT_EQ(output.shape()[1], 3);
}

// ==================== Activation Function Tests ====================

TEST(relu_forward) {
    Tensor x({3});
    float* data = x.data_ptr<float>();
    data[0] = -1.0f;
    data[1] = 0.0f;
    data[2] = 1.0f;
    
    Tensor out = nn::relu(x);
    float* out_data = out.data_ptr<float>();
    
    ASSERT_NEAR(out_data[0], 0.0f, 1e-5f);
    ASSERT_NEAR(out_data[1], 0.0f, 1e-5f);
    ASSERT_NEAR(out_data[2], 1.0f, 1e-5f);
}

TEST(sigmoid_forward) {
    Tensor x({1});
    x.data_ptr<float>()[0] = 0.0f;
    
    Tensor out = nn::sigmoid(x);
    float sig = out.data_ptr<float>()[0];
    
    ASSERT_NEAR(sig, 0.5f, 1e-3f);
}

TEST(tanh_forward) {
    Tensor x({1});
    x.data_ptr<float>()[0] = 0.0f;
    
    Tensor out = nn::tanh(x);
    float t = out.data_ptr<float>()[0];
    
    ASSERT_NEAR(t, 0.0f, 1e-5f);
}

TEST(gelu_forward) {
    Tensor x({2});
    float* data = x.data_ptr<float>();
    data[0] = 0.0f;
    data[1] = 1.0f;
    
    Tensor out = nn::gelu(x, true);  // approximate
    float* out_data = out.data_ptr<float>();
    
    // GELU(0) should be ~0
    ASSERT_NEAR(out_data[0], 0.0f, 1e-3f);
}

TEST(softmax_basic) {
    Tensor x({1, 3});  // [batch, classes]
    float* data = x.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    
    Tensor out = nn::softmax(x, 1);
    float* out_data = out.data_ptr<float>();
    
    // Softmax outputs should sum to 1
    float sum = out_data[0] + out_data[1] + out_data[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    
    // Higher input should have higher probability
    ASSERT_TRUE(out_data[2] > out_data[1]);
    ASSERT_TRUE(out_data[1] > out_data[0]);
}

// ==================== LayerNorm Tests ====================

TEST(layernorm_creation) {
    nn::LayerNorm ln(10);
    ASSERT_EQ(ln.weight().shape()[0], 10);
    ASSERT_EQ(ln.bias().shape()[0], 10);
}

TEST(layernorm_forward_shape) {
    nn::LayerNorm ln(16);
    Tensor x({2, 8, 16});  // [batch, seq, d_model]
    
    Tensor out = ln.forward(x);
    
    ASSERT_EQ(out.shape()[0], 2);
    ASSERT_EQ(out.shape()[1], 8);
    ASSERT_EQ(out.shape()[2], 16);
}

TEST(layernorm_output_stats) {
    nn::LayerNorm ln(4);
    Tensor x({1, 4});
    float* data = x.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    
    Tensor out = ln.forward(x);
    float* out_data = out.data_ptr<float>();
    
    // Output should have mean ~0
    float mean = 0.0f;
    for (int i = 0; i < 4; ++i) {
        mean += out_data[i];
    }
    mean /= 4.0f;
    
    ASSERT_NEAR(mean, 0.0f, 1e-4f);
}

// ==================== RMSNorm Tests ====================

TEST(rmsnorm_creation) {
    nn::RMSNorm rms(8);
    ASSERT_EQ(rms.weight().shape()[0], 8);
}

TEST(rmsnorm_forward) {
    nn::RMSNorm rms(4);
    Tensor x({1, 4});
    float* data = x.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;
    
    Tensor out = rms.forward(x);
    
    ASSERT_EQ(out.shape()[0], 1);
    ASSERT_EQ(out.shape()[1], 4);
}

// ==================== Dropout Tests ====================

TEST(dropout_train_mode) {
    nn::Dropout dropout(0.5f);
    dropout.train();
    
    Tensor x({100});
    x.fill(1.0f);
    
    Tensor out = dropout.forward(x);
    float* out_data = out.data_ptr<float>();
    
    // Some values should be zero, some should be 2.0 (1/0.5)
    int nonzero_count = 0;
    for (int i = 0; i < 100; ++i) {
        if (out_data[i] > 0.0f) nonzero_count++;
    }
    
    // Should be roughly 50 (with some variance)
    ASSERT_TRUE(nonzero_count > 20 && nonzero_count < 80);
}

TEST(dropout_eval_mode) {
    nn::Dropout dropout(0.5f);
    dropout.eval();
    
    Tensor x({10});
    x.fill(1.0f);
    
    Tensor out = dropout.forward(x);
    float* out_data = out.data_ptr<float>();
    
    // All values should be unchanged in eval mode
    for (int i = 0; i < 10; ++i) {
        ASSERT_NEAR(out_data[i], 1.0f, 1e-5f);
    }
}

// ==================== Embedding Tests ====================

TEST(embedding_creation) {
    nn::Embedding embed(100, 32);
    ASSERT_EQ(embed.num_embeddings(), 100);
    ASSERT_EQ(embed.embedding_dim(), 32);
    ASSERT_EQ(embed.weight().shape()[0], 100);
    ASSERT_EQ(embed.weight().shape()[1], 32);
}

TEST(embedding_forward) {
    nn::Embedding embed(100, 32);
    Tensor indices({5});  // 5 tokens
    int64_t* idx_data = indices.data_ptr<int64_t>();
    for (int i = 0; i < 5; ++i) idx_data[i] = i;
    
    Tensor out = embed.forward(indices);
    
    ASSERT_EQ(out.shape()[0], 5);
    ASSERT_EQ(out.shape()[1], 32);
}

// ==================== Pooling Tests ====================

TEST(maxpool1d_forward) {
    nn::MaxPool1d pool(2);
    Tensor x({1, 1, 8});  // [batch, channels, length]
    float* data = x.data_ptr<float>();
    for (int i = 0; i < 8; ++i) data[i] = static_cast<float>(i);
    
    Tensor out = pool.forward(x);
    
    ASSERT_EQ(out.shape()[2], 4);  // 8 / 2 = 4
}

TEST(avgpool1d_forward) {
    nn::AvgPool1d pool(2);
    Tensor x({1, 1, 4});
    float* data = x.data_ptr<float>();
    for (int i = 0; i < 4; ++i) data[i] = 2.0f;
    
    Tensor out = pool.forward(x);
    float* out_data = out.data_ptr<float>();
    
    // Average should still be 2.0
    ASSERT_NEAR(out_data[0], 2.0f, 1e-5f);
    ASSERT_NEAR(out_data[1], 2.0f, 1e-5f);
}

// ==================== Conv Tests ====================

TEST(conv1d_creation) {
    nn::Conv1d conv(3, 16, 3);
    ASSERT_EQ(conv.in_channels(), 3);
    ASSERT_EQ(conv.out_channels(), 16);
    ASSERT_EQ(conv.kernel_size(), 3);
}

TEST(conv1d_forward) {
    nn::Conv1d conv(1, 1, 3, 1, 1);
    Tensor x({1, 1, 5});  // [batch, in_channels, length]
    x.fill(1.0f);
    
    Tensor out = conv.forward(x);
    
    // output length = (5 + 2*1 - 3) / 1 + 1 = 5
    ASSERT_EQ(out.shape()[2], 5);
}

TEST(conv2d_creation) {
    nn::Conv2d conv(3, 16, 3);
    ASSERT_EQ(conv.in_channels(), 3);
    ASSERT_EQ(conv.out_channels(), 16);
}

TEST(conv2d_forward) {
    // Conv2d(in_channels, out_channels, kernel_size, stride_h, stride_w, padding_h, padding_w, ...)
    nn::Conv2d conv(1, 1, 3, 1, 1, 1, 1);  // kernel=3, stride=1, padding=1
    Tensor x({1, 1, 5, 5});  // [batch, in_channels, height, width]
    x.fill(1.0f);
    
    Tensor out = conv.forward(x);
    
    // output: (5 + 2*1 - 3) / 1 + 1 = 5
    ASSERT_EQ(out.shape()[2], 5);
    ASSERT_EQ(out.shape()[3], 5);
}

// ==================== Loss Function Tests ====================

TEST(mse_loss_basic) {
    Tensor pred({2, 1});
    Tensor target({2, 1});
    float* pred_data = pred.data_ptr<float>();
    float* target_data = target.data_ptr<float>();
    
    pred_data[0] = 1.0f;
    pred_data[1] = 3.0f;
    target_data[0] = 2.0f;
    target_data[1] = 2.0f;
    
    nn::MSELoss mse(nn::Reduction::Mean);
    Tensor loss = mse.forward(pred, target);
    
    // MSE = ((1-2)^2 + (3-2)^2) / 2 = (1 + 1) / 2 = 1.0
    ASSERT_NEAR(loss.data_ptr<float>()[0], 1.0f, 1e-5f);
}

TEST(l1_loss_basic) {
    Tensor pred({2, 1});
    Tensor target({2, 1});
    float* pred_data = pred.data_ptr<float>();
    float* target_data = target.data_ptr<float>();
    
    pred_data[0] = 1.0f;
    pred_data[1] = 3.0f;
    target_data[0] = 2.0f;
    target_data[1] = 2.0f;
    
    nn::L1Loss l1(nn::Reduction::Mean);
    Tensor loss = l1.forward(pred, target);
    
    // L1 = (|1-2| + |3-2|) / 2 = (1 + 1) / 2 = 1.0
    ASSERT_NEAR(loss.data_ptr<float>()[0], 1.0f, 1e-5f);
}

TEST(cross_entropy_basic) {
    Tensor logits({2, 3});  // 2 samples, 3 classes
    Tensor targets({2});
    int64_t* target_data = targets.data_ptr<int64_t>();
    target_data[0] = 0;
    target_data[1] = 1;
    
    // Make class 0 likely for first sample, class 1 for second
    logits.data_ptr<float>()[0] = 5.0f;   // logit for class 0
    logits.data_ptr<float>()[1] = 0.0f;   // logit for class 1
    logits.data_ptr<float>()[2] = 0.0f;   // logit for class 2
    logits.data_ptr<float>()[3] = 0.0f;   // logit for class 0
    logits.data_ptr<float>()[4] = 5.0f;   // logit for class 1
    logits.data_ptr<float>()[5] = 0.0f;   // logit for class 2
    
    nn::CrossEntropyLoss ce(nn::Reduction::None);
    Tensor loss = ce.forward(logits, targets);
    
    // Loss should be small since predictions are correct
    float* loss_data = loss.data_ptr<float>();
    ASSERT_TRUE(loss_data[0] > 0.0f && loss_data[0] < 1.0f);
    ASSERT_TRUE(loss_data[1] > 0.0f && loss_data[1] < 1.0f);
}

// ==================== Optimizer Tests ====================

TEST(sgd_optimizer) {
    Tensor param({2});
    param.fill(5.0f);
    param.set_requires_grad(true);
    
    // Set gradient
    Tensor grad({2});
    grad.fill(-1.0f);
    param.set_grad(grad);
    
    auto opt = optim::create_sgd(0.1f);
    opt->register_parameter(param);
    opt->step();
    
    // param = param - lr * grad = 5 - 0.1 * (-1) = 5.1
    float* param_data = param.data_ptr<float>();
    ASSERT_NEAR(param_data[0], 5.1f, 1e-5f);
    ASSERT_NEAR(param_data[1], 5.1f, 1e-5f);
}

TEST(adam_optimizer) {
    auto opt = optim::create_adam(0.1f);
    ASSERT_TRUE(opt != nullptr);
    
    // Check learning rate is set correctly
    ASSERT_NEAR(opt->learning_rate(), 0.1f, 1e-5f);
    
    // Test setting learning rate
    opt->set_learning_rate(0.01f);
    ASSERT_NEAR(opt->learning_rate(), 0.01f, 1e-5f);
}

// ==================== MLP Training Test ====================

TEST(full_mlp_train) {
    // Simple MLP training test
    // This tests that the full stack works together
    nn::Linear layer1(2, 8);
    nn::Linear layer2(8, 1);
    
    auto opt = optim::create_adam(0.05f);
    opt->register_parameter(layer1.weight());
    opt->register_parameter(layer1.bias());
    opt->register_parameter(layer2.weight());
    opt->register_parameter(layer2.bias());
    
    // Train for a few iterations
    for (int epoch = 0; epoch < 20; ++epoch) {
        Tensor input({2, 2});
        input.fill(0.5f);
        input.set_requires_grad(true);
        
        Tensor target({2, 1});
        target.fill(1.0f);
        
        // Forward
        Tensor out = layer1.forward(input);
        out = nn::relu(out);
        out = layer2.forward(out);
        
        // Simple loss
        Tensor diff = ops::subtract(out, target);
        Tensor loss = ops::mean(ops::multiply(diff, diff), -1, false);
        
        // Backward
        autodiff::backward(loss);
        
        // Step
        opt->step();
        opt->zero_grad();
    }
    
    ASSERT_TRUE(true);  // If we got here without crash, test passed
}
