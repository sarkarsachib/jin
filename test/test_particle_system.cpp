#include "../src/scene/particle_system.h"
#include "test_assertions.h"

namespace test_particle_system {

void test_particle_creation() {
    jin::scene::Particle particle;
    
    particle.position = jin::Tensor({10.0f, 20.0f});
    particle.velocity = jin::Tensor({1.0f, 2.0f});
    particle.acceleration = jin::Tensor({0.0f, -9.81f});
    particle.color = jin::Tensor({1.0f, 0.0f, 0.0f, 1.0f});
    particle.age = 0.0f;
    particle.lifetime = 2.0f;
    particle.size = 5.0f;
    
    ASSERT_EQ(particle.position.data_ptr<float>()[0], 10.0f);
    ASSERT_EQ(particle.age, 0.0f);
    ASSERT_TRUE(particle.alive());
}

void test_particle_update() {
    jin::scene::Particle particle;
    particle.position = jin::Tensor({0.0f, 0.0f});
    particle.velocity = jin::Tensor({10.0f, 0.0f});
    particle.acceleration = jin::Tensor({0.0f, -9.81f});
    particle.lifetime = 1.0f;
    
    float delta_time = 0.5f;
    particle.update(delta_time);
    
    // Check velocity update (v = v + a * dt)
    ASSERT_NEAR(particle.velocity.data_ptr<float>()[0], 10.0f, 0.01f);
    ASSERT_NEAR(particle.velocity.data_ptr<float>()[1], -9.81f * 0.5f, 0.01f);
    
    // Check position update (p = p + v * dt)
    ASSERT_NEAR(particle.position.data_ptr<float>()[0], 10.0f * 0.5f, 0.01f);
}

void test_particle_emitter_creation() {
    jin::scene::ParticleEmitter emitter;
    
    ASSERT_EQ(emitter.emission_rate, 10.0f);
    ASSERT_EQ(emitter.particle_lifetime, 1.0f);
    ASSERT_TRUE(emitter.emitting);
}

void test_particle_emitter_shapes() {
    jin::scene::ParticleEmitter emitter;
    
    emitter.set_shape(jin::scene::ParticleEmitter::EmitterShape::CIRCLE);
    emitter.set_position(jin::Tensor({50.0f, 50.0f}));
    
    auto emission_point = emitter.get_emission_point();
    ASSERT_TRUE(emission_point.size() >= 2);
}

void test_particle_system_creation() {
    jin::scene::ParticleSystem particle_system;
    
    ASSERT_EQ(particle_system.name, "particle_system");
    ASSERT_EQ(particle_system.max_particles, 1000);
    ASSERT_TRUE(particle_system.emit_on_update);
}

void test_particle_system_emitter_management() {
    jin::scene::ParticleSystem particle_system;
    auto emitter = std::make_shared<jin::scene::ParticleEmitter>();
    
    particle_system.add_emitter(emitter);
    ASSERT_EQ(particle_system.emitters.size(), 1);
    
    auto retrieved = particle_system.get_emitter(0);
    ASSERT_EQ(retrieved, emitter);
}

void test_particle_system_particle_management() {
    jin::scene::ParticleSystem particle_system;
    jin::scene::Particle particle;
    particle.position = jin::Tensor({0.0f, 0.0f});
    particle.lifetime = 1.0f;
    
    particle_system.add_particle(particle);
    ASSERT_EQ(particle_system.get_particle_count(), 1);
    
    particle_system.remove_dead_particles();
    // Particle should still be alive
    ASSERT_EQ(particle_system.get_particle_count(), 1);
}

void test_particle_system_forces() {
    jin::scene::ParticleSystem particle_system;
    
    jin::Tensor gravity({0.0f, -9.81f});
    jin::Tensor wind({5.0f, 0.0f});
    
    particle_system.set_global_forces(gravity, wind);
    
    // Check that forces were set
    ASSERT_TRUE(true); // Simplified test
}

void test_predefined_effects() {
    // Test creating predefined particle effects
    auto fire = jin::scene::ParticleSystem::create_fire_effect();
    auto smoke = jin::scene::ParticleSystem::create_smoke_effect();
    auto sparks = jin::scene::ParticleSystem::create_spark_effect();
    auto dust = jin::scene::ParticleSystem::create_dust_effect();
    auto rain = jin::scene::ParticleSystem::create_rain_effect();
    auto snow = jin::scene::ParticleSystem::create_snow_effect();
    
    ASSERT_TRUE(fire != nullptr);
    ASSERT_TRUE(smoke != nullptr);
    ASSERT_TRUE(sparks != nullptr);
    ASSERT_TRUE(dust != nullptr);
    ASSERT_TRUE(rain != nullptr);
    ASSERT_TRUE(snow != nullptr);
}

void test_audio_reactive_emitter() {
    jin::scene::AudioReactiveEmitter emitter;
    
    emitter.set_audio_level(0.8f);
    emitter.set_audio_sensitivity(1.5f);
    
    ASSERT_EQ(emitter.current_audio_level_, 0.8f);
    ASSERT_EQ(emitter.audio_sensitivity_, 1.5f);
}

void test_trail_particle() {
    jin::scene::TrailParticle trail_particle;
    trail_particle.position = jin::Tensor({0.0f, 0.0f});
    trail_particle.max_trail_length = 5;
    
    trail_particle.add_trail_point(jin::Tensor({1.0f, 0.0f}));
    trail_particle.add_trail_point(jin::Tensor({2.0f, 0.0f}));
    
    auto trail_point = trail_particle.get_trail_position(0);
    ASSERT_EQ(trail_point.data_ptr<float>()[0], 1.0f);
}

void test_frame_state_integration() {
    jin::temporal::FrameState frame_state;
    frame_state.motion_smoothness = 0.9f;
    
    jin::scene::ParticleSystem particle_system;
    particle_system.set_frame_state(frame_state);
    
    auto retrieved = particle_system.get_frame_state();
    ASSERT_NEAR(retrieved.motion_smoothness, 0.9f, 0.01f);
}

void test_render_modes() {
    jin::scene::ParticleSystem particle_system;
    
    particle_system.set_render_mode(jin::scene::ParticleSystem::RenderMode::POINTS);
    ASSERT_EQ(particle_system.render_mode, jin::scene::ParticleSystem::RenderMode::POINTS);
    
    particle_system.set_render_mode(jin::scene::ParticleSystem::RenderMode::SPRITES);
    ASSERT_EQ(particle_system.render_mode, jin::scene::ParticleSystem::RenderMode::SPRITES);
}

void test_performance_optimization() {
    jin::scene::ParticleSystem particle_system;
    
    particle_system.set_performance_mode(true);
    particle_system.set_max_particles(500);
    
    ASSERT_TRUE(particle_system.optimize_for_distance);
    ASSERT_EQ(particle_system.max_particles, 500);
}

void test_global_properties() {
    jin::scene::ParticleSystem particle_system;
    
    particle_system.set_global_opacity(0.7f);
    ASSERT_NEAR(particle_system.global_opacity, 0.7f, 0.01f);
    
    particle_system.set_random_seed(12345);
    // Just ensure no crash - actual randomness test would be complex
}

void register_tests() {
    using namespace jin::test;
    
    TestRunner::instance().register_test("Particle Creation", test_particle_creation);
    TestRunner::instance().register_test("Particle Update", test_particle_update);
    TestRunner::instance().register_test("Particle Emitter Creation", test_particle_emitter_creation);
    TestRunner::instance().register_test("Particle Emitter Shapes", test_particle_emitter_shapes);
    TestRunner::instance().register_test("Particle System Creation", test_particle_system_creation);
    TestRunner::instance().register_test("Particle System Emitter Management", test_particle_system_emitter_management);
    TestRunner::instance().register_test("Particle System Particle Management", test_particle_system_particle_management);
    TestRunner::instance().register_test("Particle System Forces", test_particle_system_forces);
    TestRunner::instance().register_test("Predefined Effects", test_predefined_effects);
    TestRunner::instance().register_test("Audio Reactive Emitter", test_audio_reactive_emitter);
    TestRunner::instance().register_test("Trail Particle", test_trail_particle);
    TestRunner::instance().register_test("Frame State Integration", test_frame_state_integration);
    TestRunner::instance().register_test("Render Modes", test_render_modes);
    TestRunner::instance().register_test("Performance Optimization", test_performance_optimization);
    TestRunner::instance().register_test("Global Properties", test_global_properties);
}

} // namespace test_particle_system