#include "common.h"
#include "../utils/shapes.h"
#include <random>
#include <cmath>

namespace jin {
namespace nn {

ForwardMode ForwardModeGuard::mode_ = ForwardMode::Train;

ForwardModeGuard::ForwardModeGuard(ForwardMode mode) : prev_mode_(mode_) {
    mode_ = mode;
}

ForwardModeGuard::~ForwardModeGuard() {
    mode_ = prev_mode_;
}

Shape compute_output_shape(Shape input, size_t dim, int padding, int stride, int dilation, int kernel_size) {
    if (dim < 0 || dim >= static_cast<int>(input.size())) {
        return input;
    }
    
    int in_size = static_cast<int>(input[dim]);
    int out_size = (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    Shape output = input;
    output[dim] = static_cast<size_t>(out_size);
    return output;
}

Tensor initialize_tensor(const Shape& shape, Initializer init) {
    Tensor t(shape, DType::FP32, Device::cpu());
    float* data = t.data_ptr<float>();
    size_t n = t.size();
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    switch (init.type) {
        case Initializer::Default:
        case Initializer::Zeros: {
            for (size_t i = 0; i < n; ++i) data[i] = 0.0f;
            break;
        }
        case Initializer::Ones: {
            for (size_t i = 0; i < n; ++i) data[i] = 1.0f;
            break;
        }
        case Initializer::Normal: {
            std::normal_distribution<float> dist(init.param1, init.param2);
            for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
            break;
        }
        case Initializer::XavierUniform: {
            float fan_in = static_cast<float>(shape[shape.size() - 1]);
            float fan_out = static_cast<float>(shape[0]);
            float bound = init.param1 * std::sqrt(6.0f / (fan_in + fan_out));
            std::uniform_real_distribution<float> dist_uniform(-bound, bound);
            for (size_t i = 0; i < n; ++i) data[i] = dist_uniform(gen);
            break;
        }
        case Initializer::XavierNormal: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            float fan_in = static_cast<float>(shape[shape.size() - 1]);
            float fan_out = static_cast<float>(shape[0]);
            float std = init.param1 * std::sqrt(2.0f / (fan_in + fan_out));
            std::normal_distribution<float> dist2(0.0f, std);
            for (size_t i = 0; i < n; ++i) data[i] = dist2(gen);
            break;
        }
        case Initializer::KaimingUniform: {
            float fan_in = static_cast<float>(shape[shape.size() - 1]);
            float bound = init.param1 * std::sqrt(3.0f / fan_in);
            std::uniform_real_distribution<float> dist_uniform(-bound, bound);
            for (size_t i = 0; i < n; ++i) data[i] = dist_uniform(gen);
            break;
        }
        case Initializer::KaimingNormal: {
            float fan_in = static_cast<float>(shape[shape.size() - 1]);
            float std = init.param1 / std::sqrt(fan_in);
            std::normal_distribution<float> dist(0.0f, std);
            for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
            break;
        }
    }
    
    return t;
}

} // namespace nn
} // namespace jin
