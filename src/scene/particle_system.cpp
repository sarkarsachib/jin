#include "particle_system.h"
#include <algorithm>
#include <cmath>

namespace jin {
namespace scene {

// Particle implementation
void Particle::update(float delta_time) {
    // Update physics
    velocity = velocity + acceleration * delta_time;
    position = position + velocity * delta_time;
    
    // Update rotation
    rotation += angular_velocity * delta_time;
    
    // Update age
    age += delta_time;
}

// ParticleEmitter implementation
ParticleEmitter::ParticleEmitter() : uniform_dist_(0.0f, 1.0f) {
    random_engine_.seed(std::random_device{}());
}

float ParticleEmitter::random_float(float min, float max) {
    return uniform_dist_(random_engine_) * (max - min) + min;
}

Tensor ParticleEmitter::random_vector2d(const Tensor& range) {
    return Tensor({
        random_float(range.data_ptr<float>()[0], range.data_ptr<float>()[1]),
        random_float(range.data_ptr<float>()[0], range.data_ptr<float>()[1])
    });
}

Particle ParticleEmitter::create_particle() const {
    Particle particle;
    
    // Set initial position based on emitter shape
    particle.position = get_emission_point();
    
    // Set initial velocity
    Tensor vel = initial_velocity + random_vector2d(velocity_variance);
    particle.velocity = vel;
    
    // Set physics
    particle.acceleration = Tensor({0.0f, 0.0f}); // Will be set by system
    particle.age = 0.0f;
    particle.lifetime = particle_lifetime;
    
    // Set appearance
    particle.color = start_color;
    particle.size = random_float(size_range.data_ptr<float>()[0], size_range.data_ptr<float>()[1]);
    particle.rotation = 0.0f;
    particle.angular_velocity = random_float(rotation_speed_range.first, rotation_speed_range.second);
    
    return particle;
}

Tensor ParticleEmitter::get_emission_point() const {
    switch (shape) {
        case EmitterShape::POINT:
            return position;
            
        case EmitterShape::LINE: {
            // Emit along a line (simplified)
            float t = uniform_dist_(random_engine_);
            return Tensor({
                position.data_ptr<float>()[0] + t * 10.0f,
                position.data_ptr<float>()[1]
            });
        }
        
        case EmitterShape::CIRCLE: {
            // Emit in a circle
            float angle = uniform_dist_(random_engine_) * 2.0f * 3.14159f;
            float radius = 5.0f; // Fixed radius for now
            return Tensor({
                position.data_ptr<float>()[0] + std::cos(angle) * radius,
                position.data_ptr<float>()[1] + std::sin(angle) * radius
            });
        }
        
        case EmitterShape::BOX: {
            // Emit within a box
            return Tensor({
                position.data_ptr<float>()[0] + random_float(-5.0f, 5.0f),
                position.data_ptr<float>()[1] + random_float(-5.0f, 5.0f)
            });
        }
        
        case EmitterShape::SPHERE:
            // For 2D, treat as circle
            return get_emission_point();
            
        default:
            return position;
    }
}

void ParticleEmitter::update(float delta_time) {
    emission_timer += delta_time * emission_rate;
    
    // Emit particles based on emission rate
    while (emission_timer >= 1.0f) {
        emission_timer -= 1.0f;
        if (emitting) {
            // Particle will be created by the system
        }
    }
}

// ParticleSystem implementation
ParticleSystem::ParticleSystem() {
    name = "particle_system";
    random_engine_.seed(std::random_device{}());
}

void ParticleSystem::add_emitter(std::shared_ptr<ParticleEmitter> emitter) {
    if (emitter) {
        emitters.push_back(emitter);
    }
}

void ParticleSystem::remove_emitter(size_t index) {
    if (index < emitters.size()) {
        emitters.erase(emitters.begin() + index);
    }
}

void ParticleSystem::clear_emitters() {
    emitters.clear();
}

std::shared_ptr<ParticleEmitter> ParticleSystem::get_emitter(size_t index) const {
    if (index < emitters.size()) {
        return emitters[index];
    }
    return nullptr;
}

void ParticleSystem::add_particle(const Particle& particle) {
    if (static_cast<int>(particles.size()) < max_particles) {
        particles.push_back(particle);
    }
}

void ParticleSystem::remove_dead_particles() {
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return !p.alive(); }),
        particles.end()
    );
}

void ParticleSystem::clear_particles() {
    particles.clear();
}

void ParticleSystem::set_global_forces(const Tensor& gravity, const Tensor& wind) {
    global_gravity = gravity;
    global_wind = wind;
}

void ParticleSystem::emit_particles(float delta_time) {
    if (!emit_on_update) return;
    
    for (auto& emitter : emitters) {
        emitter->update(delta_time);
        
        if (emitter->emitting) {
            // Calculate how many particles to emit
            int particles_to_emit = static_cast<int>(emitter->emission_rate * delta_time);
            
            for (int i = 0; i < particles_to_emit; ++i) {
                if (static_cast<int>(particles.size()) < max_particles) {
                    Particle particle = emitter->create_particle();
                    add_particle(particle);
                }
            }
        }
    }
}

void ParticleSystem::update_particles(float delta_time) {
    float simulation_delta = delta_time * simulation_speed;
    
    for (auto& particle : particles) {
        // Apply global forces
        particle.acceleration = global_gravity + global_wind;
        
        // Update particle physics
        particle.update(simulation_delta);
        
        // Update color over lifetime (fade out)
        float t = particle.normalized_age();
        float* start_color = particle.color.data_ptr<float>();
        float* end_color = Tensor({0.0f, 0.0f, 0.0f, 0.0f}).data_ptr<float>();
        
        if (particle.color.size() >= 4) {
            particle.color.data_ptr<float>()[0] = start_color[0] * (1.0f - t) + end_color[0] * t; // R
            particle.color.data_ptr<float>()[1] = start_color[1] * (1.0f - t) + end_color[1] * t; // G
            particle.color.data_ptr<float>()[2] = start_color[2] * (1.0f - t) + end_color[2] * t; // B
            particle.color.data_ptr<float>()[3] = start_color[3] * (1.0f - t); // A (fade out)
        }
    }
    
    // Remove dead particles
    remove_dead_particles();
}

void ParticleSystem::apply_forces(float delta_time) {
    // Additional force application logic can go here
    // For now, forces are applied in update_particles
}

void ParticleSystem::update(float delta_time) {
    emit_particles(delta_time);
    update_particles(delta_time);
    apply_forces(delta_time);
    needs_update_ = true;
}

Tensor ParticleSystem::render(int width, int height, const Camera2D& camera) const {
    update_cached_data();
    
    switch (render_mode) {
        case RenderMode::POINTS:
            return render_points(camera);
        case RenderMode::SPRITES:
            return render_sprites(camera);
        case RenderMode::LINES:
            return render_lines(camera);
        case RenderMode::TRAILS:
            return render_trails(camera);
        default:
            return render_sprites(camera);
    }
}

void ParticleSystem::set_frame_state(const temporal::FrameState& frame_state) {
    SceneNode::set_frame_state(frame_state);
    
    // Update particles based on frame state
    // This could be used for audio-reactive effects or other temporal behaviors
    needs_update_ = true;
}

void ParticleSystem::set_random_seed(int seed) {
    random_engine_.seed(seed);
}

std::shared_ptr<ParticleSystem> ParticleSystem::create_fire_effect() {
    auto system = std::make_shared<ParticleSystem>();
    auto emitter = std::make_shared<ParticleEmitter>();
    
    // Configure for fire effect
    emitter->set_emission_rate(20.0f);
    emitter->particle_lifetime = 2.0f;
    emitter->start_color = Tensor({1.0f, 0.5f, 0.0f, 1.0f}); // Orange
    emitter->end_color = Tensor({0.2f, 0.2f, 0.2f, 0.0f});    // Dark gray, transparent
    emitter->initial_velocity = Tensor({0.0f, 2.0f});          // Upward velocity
    emitter->velocity_variance = Tensor({1.0f, 1.0f});        // Random variation
    emitter->size_range = Tensor({2.0f, 6.0f});               // Size range
    
    system->add_emitter(emitter);
    system->set_global_forces(Tensor({0.0f, 1.0f}), Tensor({0.0f, 0.0f})); // Slight upward force
    system->set_render_mode(RenderMode::SPRITES);
    
    return system;
}

std::shared_ptr<ParticleSystem> ParticleSystem::create_smoke_effect() {
    auto system = std::make_shared<ParticleSystem>();
    auto emitter = std::make_shared<ParticleEmitter>();
    
    // Configure for smoke effect
    emitter->set_emission_rate(5.0f);
    emitter->particle_lifetime = 4.0f;
    emitter->start_color = Tensor({0.5f, 0.5f, 0.5f, 0.6f}); // Light gray, semi-transparent
    emitter->end_color = Tensor({0.3f, 0.3f, 0.3f, 0.0f});    // Darker gray, transparent
    emitter->initial_velocity = Tensor({0.0f, 1.0f});          // Slow upward velocity
    emitter->velocity_variance = Tensor({0.5f, 0.5f});       // Less variation
    emitter->size_range = Tensor({8.0f, 15.0f});              // Larger particles
    
    system->add_emitter(emitter);
    system->set_global_forces(Tensor({0.0f, 0.5f}), Tensor({0.0f, 0.0f})); // Gentle upward force
    system->set_render_mode(RenderMode::SPRITES);
    
    return system;
}

std::shared_ptr<ParticleSystem> ParticleSystem::create_spark_effect() {
    auto system = std::make_shared<ParticleSystem>();
    auto emitter = std::make_shared<ParticleEmitter>();
    
    // Configure for spark effect
    emitter->set_emission_rate(15.0f);
    emitter->particle_lifetime = 0.5f;
    emitter->start_color = Tensor({1.0f, 1.0f, 0.0f, 1.0f}); // Yellow
    emitter->end_color = Tensor({1.0f, 0.0f, 0.0f, 0.0f});   // Red, transparent
    emitter->initial_velocity = Tensor({0.0f, -3.0f});        // Upward with gravity
    emitter->velocity_variance = Tensor({2.0f, 2.0f});       // High variation
    emitter->size_range = Tensor({1.0f, 3.0f});              // Small particles
    
    system->add_emitter(emitter);
    system->set_global_forces(Tensor({0.0f, -9.81f}), Tensor({0.0f, 0.0f})); // Gravity
    system->set_render_mode(RenderMode::POINTS);
    
    return system;
}

std::shared_ptr<ParticleSystem> ParticleSystem::create_dust_effect() {
    auto system = std::make_shared<ParticleSystem>();
    auto emitter = std::make_shared<ParticleEmitter>();
    
    // Configure for dust effect
    emitter->set_emission_rate(3.0f);
    emitter->particle_lifetime = 3.0f;
    emitter->start_color = Tensor({0.8f, 0.7f, 0.6f, 0.4f}); // Brownish, semi-transparent
    emitter->end_color = Tensor({0.6f, 0.5f, 0.4f, 0.0f});   // Darker, transparent
    emitter->initial_velocity = Tensor({0.5f, 0.0f});         // Slow horizontal movement
    emitter->velocity_variance = Tensor({1.0f, 0.5f});       // Moderate variation
    emitter->size_range = Tensor({1.0f, 4.0f});              // Variable size
    
    system->add_emitter(emitter);
    system->set_global_forces(Tensor({0.0f, -2.0f}), Tensor({0.0f, 0.0f})); // Gentle gravity
    system->set_render_mode(RenderMode::SPRITES);
    
    return system;
}

std::shared_ptr<ParticleSystem> ParticleSystem::create_rain_effect() {
    auto system = std::make_shared<ParticleSystem>();
    auto emitter = std::make_shared<ParticleEmitter>();
    
    // Configure for rain effect
    emitter->set_emission_rate(50.0f);
    emitter->particle_lifetime = 1.0f;
    emitter->start_color = Tensor({0.7f, 0.8f, 1.0f, 0.8f}); // Blue, semi-transparent
    emitter->end_color = Tensor({0.5f, 0.6f, 0.8f, 0.0f});   // Darker blue, transparent
    emitter->initial_velocity = Tensor({0.0f, -10.0f});       // Fast downward
    emitter->velocity_variance = Tensor({1.0f, 2.0f});       // Variation
    emitter->size_range = Tensor({1.0f, 2.0f});              // Thin streaks
    
    system->add_emitter(emitter);
    system->set_global_forces(Tensor({0.0f, -9.81f}), Tensor({0.0f, 0.0f})); // Gravity
    system->set_render_mode(RenderMode::LINES);
    
    return system;
}

std::shared_ptr<ParticleSystem> ParticleSystem::create_snow_effect() {
    auto system = std::make_shared<ParticleSystem>();
    auto emitter = std::make_shared<ParticleEmitter>();
    
    // Configure for snow effect
    emitter->set_emission_rate(8.0f);
    emitter->particle_lifetime = 5.0f;
    emitter->start_color = Tensor({1.0f, 1.0f, 1.0f, 0.9f}); // White
    emitter->end_color = Tensor({1.0f, 1.0f, 1.0f, 0.0f});   // Transparent
    emitter->initial_velocity = Tensor({0.5f, -2.0f});       // Gentle downward drift
    emitter->velocity_variance = Tensor({2.0f, 1.0f});       // Wind drift
    emitter->size_range = Tensor({2.0f, 5.0f});              // Variable snowflakes
    
    system->add_emitter(emitter);
    system->set_global_forces(Tensor({0.0f, -1.0f}), Tensor({1.0f, 0.0f})); // Gravity + wind
    system->set_render_mode(RenderMode::SPRITES);
    
    return system;
}

void ParticleSystem::update_cached_data() const {
    if (!needs_update_) return;
    
    // Update cached render data
    // This would involve optimizing particle rendering data
    needs_update_ = false;
}

Tensor ParticleSystem::render_points(const Camera2D& camera) const {
    Tensor render_buffer = zeros({720, 1280, 4}); // Default size
    
    for (const auto& particle : particles) {
        if (should_cull_particle(particle, camera)) continue;
        
        // Convert world position to screen coordinates
        Tensor screen_pos = apply_global_transforms(particle.position, camera);
        
        if (screen_pos.size() >= 2) {
            int x = static_cast<int>(screen_pos.data_ptr<float>()[0]);
            int y = static_cast<int>(screen_pos.data_ptr<float>()[1]);
            
            // Simple point rendering
            if (x >= 0 && x < 1280 && y >= 0 && y < 720 && particle.color.size() >= 4) {
                size_t idx = (y * 1280 + x) * 4;
                if (idx + 3 < render_buffer.size()) {
                    float* buffer_data = render_buffer.data_ptr<float>();
                    buffer_data[idx] = particle.color.data_ptr<float>()[0];     // R
                    buffer_data[idx + 1] = particle.color.data_ptr<float>()[1]; // G
                    buffer_data[idx + 2] = particle.color.data_ptr<float>()[2]; // B
                    buffer_data[idx + 3] = particle.color.data_ptr<float>()[3]; // A
                }
            }
        }
    }
    
    return render_buffer;
}

Tensor ParticleSystem::render_sprites(const Camera2D& camera) const {
    // Simplified sprite rendering
    // In a real implementation, this would render particle sprites with proper texture mapping
    
    Tensor render_buffer = zeros({720, 1280, 4});
    
    for (const auto& particle : particles) {
        if (should_cull_particle(particle, camera)) continue;
        
        Tensor screen_pos = apply_global_transforms(particle.position, camera);
        
        if (screen_pos.size() >= 2) {
            int center_x = static_cast<int>(screen_pos.data_ptr<float>()[0]);
            int center_y = static_cast<int>(screen_pos.data_ptr<float>()[1]);
            int radius = static_cast<int>(particle.size);
            
            // Simple circular sprite rendering
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int x = center_x + dx;
                    int y = center_y + dy;
                    
                    if (x >= 0 && x < 1280 && y >= 0 && y < 720) {
                        if (dx * dx + dy * dy <= radius * radius) {
                            size_t idx = (y * 1280 + x) * 4;
                            if (idx + 3 < render_buffer.size()) {
                                float* buffer_data = render_buffer.data_ptr<float>();
                                buffer_data[idx] = particle.color.data_ptr<float>()[0];     // R
                                buffer_data[idx + 1] = particle.color.data_ptr<float>()[1]; // G
                                buffer_data[idx + 2] = particle.color.data_ptr<float>()[2]; // B
                                buffer_data[idx + 3] = particle.color.data_ptr<float>()[3]; // A
                            }
                        }
                    }
                }
            }
        }
    }
    
    return render_buffer;
}

Tensor ParticleSystem::render_lines(const Camera2D& camera) const {
    // Line rendering (for rain, etc.)
    Tensor render_buffer = zeros({720, 1280, 4});
    
    for (const auto& particle : particles) {
        if (should_cull_particle(particle, camera)) continue;
        
        // Render as line segment in direction of velocity
        Tensor screen_pos = apply_global_transforms(particle.position, camera);
        Tensor screen_vel = particle.velocity * 0.1f; // Scale for visibility
        
        if (screen_pos.size() >= 2 && screen_vel.size() >= 2) {
            int start_x = static_cast<int>(screen_pos.data_ptr<float>()[0]);
            int start_y = static_cast<int>(screen_pos.data_ptr<float>()[1]);
            int end_x = static_cast<int>(screen_pos.data_ptr<float>()[0] + screen_vel.data_ptr<float>()[0]);
            int end_y = static_cast<int>(screen_pos.data_ptr<float>()[1] + screen_vel.data_ptr<float>()[1]);
            
            // Simple line drawing algorithm (Bresenham's line algorithm simplified)
            int dx = std::abs(end_x - start_x);
            int dy = std::abs(end_y - start_y);
            int sx = (start_x < end_x) ? 1 : -1;
            int sy = (start_y < end_y) ? 1 : -1;
            int err = dx - dy;
            
            int x = start_x;
            int y = start_y;
            
            while (true) {
                if (x >= 0 && x < 1280 && y >= 0 && y < 720 && particle.color.size() >= 4) {
                    size_t idx = (y * 1280 + x) * 4;
                    if (idx + 3 < render_buffer.size()) {
                        float* buffer_data = render_buffer.data_ptr<float>();
                        buffer_data[idx] = particle.color.data_ptr<float>()[0];     // R
                        buffer_data[idx + 1] = particle.color.data_ptr<float>()[1]; // G
                        buffer_data[idx + 2] = particle.color.data_ptr<float>()[2]; // B
                        buffer_data[idx + 3] = particle.color.data_ptr<float>()[3]; // A
                    }
                }
                
                if (x == end_x && y == end_y) break;
                
                int e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    x += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    y += sy;
                }
            }
        }
    }
    
    return render_buffer;
}

Tensor ParticleSystem::render_trails(const Camera2D& camera) const {
    // Trail rendering (simplified - just render current positions)
    return render_sprites(camera);
}

bool ParticleSystem::should_cull_particle(const Particle& particle, const Camera2D& camera) const {
    if (!optimize_for_distance) return false;
    
    // Simple distance culling
    Tensor screen_pos = apply_global_transforms(particle.position, camera);
    
    if (screen_pos.size() >= 2) {
        float x = screen_pos.data_ptr<float>()[0];
        float y = screen_pos.data_ptr<float>()[1];
        
        return (x < -cull_distance || x > 1280 + cull_distance ||
                y < -cull_distance || y > 720 + cull_distance);
    }
    
    return false;
}

Tensor ParticleSystem::apply_global_transforms(const Tensor& position, const Camera2D& camera) const {
    // Apply world transform and camera transform
    Transform2D world_transform = get_world_transform();
    Tensor world_pos = world_transform.position + position;
    
    return camera.world_to_screen(world_pos);
}

// TrailParticle implementation
void TrailParticle::add_trail_point(const Tensor& position) {
    trail_positions.push_back(position);
    
    // Limit trail length
    while (static_cast<int>(trail_positions.size()) > max_trail_length) {
        trail_positions.erase(trail_positions.begin());
    }
}

void TrailParticle::update_trail() {
    // Add current position to trail
    add_trail_point(position);
}

Tensor TrailParticle::get_trail_position(int index) const {
    if (index >= 0 && index < static_cast<int>(trail_positions.size())) {
        return trail_positions[index];
    }
    return Tensor();
}

// AudioReactiveEmitter implementation
void AudioReactiveEmitter::update(float delta_time) {
    ParticleEmitter::update(delta_time);
    
    // Adjust emission based on audio level
    if (current_audio_level_ > 0.1f) {
        emission_rate = base_emission_rate * (1.0f + current_audio_level_ * audio_sensitivity_);
    } else {
        emission_rate = base_emission_rate;
    }
}

Particle AudioReactiveEmitter::create_particle() const {
    Particle particle = ParticleEmitter::create_particle();
    
    // Modify particle properties based on audio
    if (current_audio_level_ > 0.0f) {
        // Make particles brighter and faster with higher audio levels
        float brightness = std::min(1.0f, particle.color.data_ptr<float>()[0] * (1.0f + current_audio_level_));
        particle.color.data_ptr<float>()[0] = brightness; // R
        particle.color.data_ptr<float>()[1] = brightness; // G
        particle.color.data_ptr<float>()[2] = brightness; // B
        
        // Increase velocity based on audio
        particle.velocity = particle.velocity * (1.0f + current_audio_level_ * 0.5f);
    }
    
    return particle;
}

float AudioReactiveEmitter::get_frequency_energy(const std::pair<float, float>& frequency_range) const {
    if (frequency_data_.size() == 0) return 0.0f;
    
    // Simplified frequency analysis
    // In a real implementation, this would properly analyze frequency bands
    return current_audio_level_;
}

} // namespace scene
} // namespace jin