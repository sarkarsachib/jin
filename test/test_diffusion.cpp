#include "test_runner.cpp"

#include "../src/diffusion/scheduler.h"
#include "../src/diffusion/sampler.h"
#include "../src/diffusion/trainer.h"
#include "../src/diffusion/model.h"
#include "../src/autodiff/backward.h"
#include "../src/autodiff/grad_registry.h"
#include "../src/autodiff/tape.h"
#include "../src/optim/optim.h"

#include <cmath>

using namespace jin;
using namespace jin::test;

TEST(diffusion_scheduler_linear_beta_monotonic) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 50;
    cfg.schedule_type = "linear";
    cfg.beta_start = 1e-4f;
    cfg.beta_end = 2e-2f;

    diffusion::NoiseScheduler sched(cfg);

    float prev = sched.get_beta(1);
    for (size_t t = 2; t <= cfg.timesteps; ++t) {
        float b = sched.get_beta(t);
        ASSERT_TRUE(b >= prev);
        prev = b;
    }
}

TEST(diffusion_scheduler_linear_alpha_cumprod_decreasing) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 50;
    cfg.schedule_type = "linear";

    diffusion::NoiseScheduler sched(cfg);

    float prev = sched.get_alpha(0);
    for (size_t t = 1; t <= cfg.timesteps; ++t) {
        float a = sched.get_alpha(t);
        ASSERT_TRUE(a <= prev + 1e-6f);
        prev = a;
    }
}

TEST(diffusion_scheduler_cosine_beta_bounds) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 100;
    cfg.schedule_type = "cosine";

    diffusion::NoiseScheduler sched(cfg);

    for (size_t t = 1; t <= cfg.timesteps; ++t) {
        float b = sched.get_beta(t);
        ASSERT_TRUE(b > 0.0f);
        ASSERT_TRUE(b < 1.0f);
    }
}

TEST(diffusion_scheduler_posterior_variance_valid) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 100;
    cfg.schedule_type = "linear";

    diffusion::NoiseScheduler sched(cfg);

    for (size_t t = 1; t <= cfg.timesteps; ++t) {
        float pv = sched.get_posterior_variance(t);
        ASSERT_TRUE(pv >= 0.0f);
        ASSERT_TRUE(pv <= sched.get_beta(t) + 1e-5f);
    }
}

TEST(diffusion_scheduler_scale_sample_t0_identity) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 10;
    diffusion::NoiseScheduler sched(cfg);

    Tensor x0({2, 3, 4, 4});
    x0.fill(0.25f);
    Tensor eps({2, 3, 4, 4});
    eps.fill(0.0f);

    Tensor xt = sched.scale_sample(x0, 0, eps);
    ASSERT_EQ(xt.size(), x0.size());

    for (size_t i = 0; i < x0.size(); ++i) {
        ASSERT_NEAR(xt.data_ptr<float>()[i], x0.data_ptr<float>()[i], 1e-6f);
    }
}

TEST(diffusion_scheduler_scale_sample_noise_effect) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 10;
    diffusion::NoiseScheduler sched(cfg);

    Tensor x0({1, 1, 2, 2});
    x0.fill(1.0f);
    Tensor eps({1, 1, 2, 2});
    eps.fill(1.0f);

    Tensor xt = sched.scale_sample(x0, 5, eps);

    float a = sched.get_alpha(5);
    float expected = std::sqrt(a) * 1.0f + std::sqrt(1.0f - a) * 1.0f;
    ASSERT_NEAR(xt.data_ptr<float>()[0], expected, 1e-4f);
}

TEST(diffusion_unet_output_shape_32) {
    nn::UNet unet(3, 3, 16, {8});
    Tensor x({2, 3, 32, 32});
    x.fill(0.1f);

    Tensor y = unet.forward(x, 10);
    ASSERT_EQ(y.shape()[0], 2);
    ASSERT_EQ(y.shape()[1], 3);
    ASSERT_EQ(y.shape()[2], 32);
    ASSERT_EQ(y.shape()[3], 32);
}

TEST(diffusion_time_embedding_shape) {
    nn::TimeEmbedding te(32);
    Tensor t({4});
    float* td = t.data_ptr<float>();
    td[0] = 1.0f;
    td[1] = 2.0f;
    td[2] = 3.0f;
    td[3] = 4.0f;

    Tensor e = te.forward(t);
    ASSERT_EQ(e.shape()[0], 4);
    ASSERT_EQ(e.shape()[1], 32);
}

TEST(diffusion_unet_timestep_changes_output) {
    nn::UNet unet(3, 3, 16, {8});
    Tensor x({1, 3, 32, 32});
    x.fill(0.05f);

    Tensor y1 = unet.forward(x, 1);
    Tensor y2 = unet.forward(x, 10);

    // Not guaranteed large, but should differ with high probability.
    ASSERT_NE(y1.data_ptr<float>()[0], y2.data_ptr<float>()[0]);
}

TEST(diffusion_resnet_skip_identity_when_zero_params) {
    nn::ResnetBlock block(4, 4, 16);
    for (auto& p : block.parameters()) {
        p.fill(0.0f);
    }

    Tensor x({1, 4, 8, 8});
    x.fill_random(-1.0f, 1.0f);
    Tensor t({1, 16});
    t.fill(0.0f);

    Tensor y = block.forward(x, t);

    const float* xd = x.data_ptr<float>();
    const float* yd = y.data_ptr<float>();
    for (size_t i = 0; i < x.size(); ++i) {
        ASSERT_NEAR(xd[i], yd[i], 1e-5f);
    }
}

TEST(diffusion_unet_backward_input_grad) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();

    nn::UNet unet(3, 3, 8, {8});
    Tensor x({1, 3, 16, 16});
    x.fill(0.2f);
    x.set_requires_grad(true);

    Tensor y = unet.forward(x, 3);

    Tensor grad_out = ones(y.shape());
    autodiff::backward(y, &grad_out);

    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(x));
}

TEST(diffusion_unet_backward_param_grad) {
    autodiff::GradientTape::instance().clear();
    autodiff::GradientRegistry::instance().clear_all();

    nn::UNet unet(3, 3, 8, {8});
    Tensor x({1, 3, 16, 16});
    x.fill(0.2f);
    x.set_requires_grad(true);

    Tensor y = unet.forward(x, 3);

    Tensor grad_out = ones(y.shape());
    autodiff::backward(y, &grad_out);

    auto params = unet.parameters();
    ASSERT_TRUE(params.size() > 0);
    ASSERT_TRUE(autodiff::GradientRegistry::instance().has_grad(params[0]));
}

TEST(diffusion_ddpm_sampler_shape) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 20;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    diffusion::DDPMSampler sampler(sched, unet);
    sampler.set_seed(123);

    Tensor out = sampler.sample(2, 16, 16, 10);
    ASSERT_EQ(out.shape()[0], 2);
    ASSERT_EQ(out.shape()[1], 3);
    ASSERT_EQ(out.shape()[2], 16);
    ASSERT_EQ(out.shape()[3], 16);
}

TEST(diffusion_ddpm_sampler_deterministic_seed) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 20;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    diffusion::DDPMSampler sampler(sched, unet);

    sampler.set_seed(42);
    Tensor a = sampler.sample(1, 8, 8, 5);

    sampler.set_seed(42);
    Tensor b = sampler.sample(1, 8, 8, 5);

    ASSERT_NEAR(a.data_ptr<float>()[0], b.data_ptr<float>()[0], 1e-6f);
}

TEST(diffusion_ddim_sampler_deterministic_seed) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 20;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    diffusion::DDPMSampler sampler(sched, unet);

    sampler.set_seed(7);
    Tensor a = sampler.sample_deterministic(1, 8, 8, 6);

    sampler.set_seed(7);
    Tensor b = sampler.sample_deterministic(1, 8, 8, 6);

    ASSERT_NEAR(a.data_ptr<float>()[0], b.data_ptr<float>()[0], 1e-6f);
}

TEST(diffusion_guidance_scale_changes_output) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 20;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    diffusion::DDPMSampler sampler(sched, unet);
    sampler.set_seed(9);

    Tensor a = sampler.sample_deterministic(1, 8, 8, 6, 1.0f);

    sampler.set_seed(9);
    Tensor b = sampler.sample_deterministic(1, 8, 8, 6, 2.0f);

    ASSERT_NE(a.data_ptr<float>()[0], b.data_ptr<float>()[0]);
}

TEST(diffusion_ddpm_num_steps_changes_output) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 50;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    diffusion::DDPMSampler sampler(sched, unet);
    sampler.set_seed(99);

    Tensor a = sampler.sample_deterministic(1, 8, 8, 10);

    sampler.set_seed(99);
    Tensor b = sampler.sample_deterministic(1, 8, 8, 20);

    ASSERT_NE(a.data_ptr<float>()[0], b.data_ptr<float>()[0]);
}

TEST(diffusion_trainer_compute_loss_scalar) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 10;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    auto opt = optim::create_adam(1e-2f);

    diffusion::DiffusionTrainer trainer(unet, sched, *opt);
    trainer.set_seed(0);

    Tensor x0({2, 3, 8, 8});
    x0.fill_random(-1.0f, 1.0f);

    Tensor loss = trainer.compute_loss(x0);
    ASSERT_EQ(loss.size(), 1);
    ASSERT_TRUE(loss.item() >= 0.0f);
}

TEST(diffusion_trainer_training_updates_params) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 10;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    auto opt = optim::create_adam(1e-2f);

    diffusion::DiffusionTrainer trainer(unet, sched, *opt);
    trainer.set_seed(123);

    Tensor x0({2, 3, 8, 8});
    x0.fill_random(-1.0f, 1.0f);

    auto params = unet.parameters();
    float before = params[0].data_ptr<float>()[0];

    trainer.training_step(x0);

    params = unet.parameters();
    float after = params[0].data_ptr<float>()[0];

    ASSERT_NE(before, after);
}

TEST(diffusion_trainer_loss_decreases_fixed_noise) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 20;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    auto opt = optim::create_adam(5e-3f);

    diffusion::DiffusionTrainer trainer(unet, sched, *opt);
    trainer.set_seed(1);

    Tensor x0({2, 3, 8, 8});
    x0.fill(0.1f);

    size_t t = 10;
    Tensor noise({2, 3, 8, 8});
    noise.fill(0.0f);

    float first = 0.0f;
    float last = 0.0f;

    for (int i = 0; i < 10; ++i) {
        autodiff::GradientTape::instance().clear();
        autodiff::GradientRegistry::instance().clear_all();
        opt->zero_grad();

        Tensor loss = trainer.compute_loss(x0, t, noise);
        if (i == 0) first = loss.item();

        autodiff::backward(loss, nullptr);
        opt->step();
        trainer.update_ema(0.9f);

        last = loss.item();
    }

    ASSERT_TRUE(last < first);
}

TEST(diffusion_trainer_ema_updates) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 10;
    diffusion::NoiseScheduler sched(cfg);

    nn::UNet unet(3, 3, 8, {8});
    auto opt = optim::create_adam(1e-2f);

    diffusion::DiffusionTrainer trainer(unet, sched, *opt);
    trainer.set_seed(0);

    Tensor x0({2, 3, 8, 8});
    x0.fill_random(-1.0f, 1.0f);

    trainer.training_step(x0);

    auto params = unet.parameters();
    auto ema = trainer.ema_parameters();

    ASSERT_EQ(params.size(), ema.size());
    ASSERT_NE(params[0].data_ptr<float>()[0], ema[0].data_ptr<float>()[0]);
}

TEST(diffusion_model_generate_shape) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 20;

    diffusion::DiffusionModel model(cfg, 3, 3, 8, {8});
    model.sampler().set_seed(0);

    Tensor out = model.generate(2, 16, 16, 8);
    ASSERT_EQ(out.shape()[0], 2);
    ASSERT_EQ(out.shape()[1], 3);
    ASSERT_EQ(out.shape()[2], 16);
    ASSERT_EQ(out.shape()[3], 16);
}

TEST(diffusion_model_generate_deterministic_seed) {
    diffusion::SchedulerConfig cfg;
    cfg.timesteps = 20;

    diffusion::DiffusionModel model(cfg, 3, 3, 8, {8});

    model.sampler().set_seed(11);
    Tensor a = model.generate(1, 8, 8, 6);

    model.sampler().set_seed(11);
    Tensor b = model.generate(1, 8, 8, 6);

    ASSERT_NEAR(a.data_ptr<float>()[0], b.data_ptr<float>()[0], 1e-6f);
}
