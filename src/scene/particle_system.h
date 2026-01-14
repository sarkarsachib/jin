#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include "scene_graph.h"
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <functional>

namespace jin {
namespace scene {

/**
 * Particle - Individual particle in a particle system
 */
struct Particle {
    Tensor position;          // Current position [x, y]
    Tensor velocity;         // Current velocity [vx, vy]
    Tensor acceleration;     // Current acceleration [ax, ay]
    Tensor color;           // RGBA color [r, g, b, a]
    float age = 0.0f;       // Current age in seconds
    float lifetime = 1.0f;  // Total lifetime in seconds
    float size = 1.0f;      // Particle size
    float rotation = 0.0f;   // Rotation angle in radians
    float angular_velocity = 0.0f; // Rotation speed
    
    // Lifecycle state
    bool alive() const { return age < lifetime; }
    float normalized_age() const { return lifetime > 0.0f ? age / lifetime : 1.0f; }
    
    // Update particle physics
    void update(float delta_time);
};

/**
 * ParticleEmitter - Base class for particle emission
 */
class ParticleEmitter {
public:
    // Emitter properties
    Tensor position;                    // Emitter position [x, y]
    enum class EmitterShape { POINT, LINE, CIRCLE, BOX, SPHERE };
    EmitterShape shape = EmitterShape::POINT;
    
    // Emission parameters
    float emission_rate = 10.0f;       // Particles per second
    float particle_lifetime = 1.0f;     // Default particle lifetime
    Tensor initial_velocity = {0.0f, 0.0f}; // Initial velocity range
    Tensor velocity_variance = {1.0f, 1.0f};  // Random velocity variation
    
    // Particle appearance
    Tensor start_color = {1.0f, 1.0f, 1.0f, 1.0f}; // RGBA
    Tensor end_color = {0.0f, 0.0f, 0.0f, 0.0f};   // RGBA (fade out)
    Tensor size_range = {1.0f, 3.0f};             // [min, max] size
    float rotation_speed_range = {-2.0f, 2.0f};    // [min, max] rotation speed
    
    // Emission timing
    bool emitting = true;
    float emission_timer = 0.0f;
    
    ParticleEmitter();
    virtual ~ParticleEmitter() = default;
    
    // Position and shape
    void set_position(const Tensor& pos) { position = pos; }
    void set_shape(EmitterShape new_shape) { shape = new_shape; }
    
    // Emission control
    void start_emission() { emitting = true; }
    void stop_emission() { emitting = false; }
    void set_emission_rate(float rate) { emission_rate = std::max(0.0f, rate); }
    
    // Particle generation
    virtual Particle create_particle() const;
    virtual Tensor get_emission_point() const;
    
    // Update emitter
    virtual void update(float delta_time);
    
protected:
    std::mt19937 random_engine_;
    std::uniform_real_distribution<float> uniform_dist_;
    
    // Random number generation
    float random_float(float min, float max);
    Tensor random_vector2d(const Tensor& range);
};

/**
 * ParticleSystem - Complete particle system for effects
 */
class ParticleSystem : public SceneNode {
public:
    // System properties
    std::vector<Particle> particles;
    std::vector<std::shared_ptr<ParticleEmitter>> emitters;
    
    // Performance settings
    int max_particles = 1000;
    bool optimize_for_distance = true; // Reduce particles far from camera
    float cull_distance = 500.0f;      // Distance beyond which particles are culled
    
    // Global forces
    Tensor global_gravity = {0.0f, -9.81f}; // Gravity force [x, y]
    Tensor global_wind = {0.0f, 0.0f};      // Wind force [x, y]
    
    // Rendering properties
    enum class RenderMode {
        POINTS,        // Render as points
        SPRITES,       // Render as sprites
        LINES,         // Render as line segments
        TRAILS         // Render with motion trails
    };
    
    RenderMode render_mode = RenderMode::SPRITES;
    Tensor particle_texture;             // Texture for sprite rendering
    BlendMode blend_mode = BlendMode::ADD;
    float global_opacity = 1.0f;
    
    // Animation properties
    bool emit_on_update = true;
    float simulation_speed = 1.0f;      // Multiplier for simulation speed
    
    ParticleSystem();
    ~ParticleSystem() = default;
    
    // Emitter management
    void add_emitter(std::shared_ptr<ParticleEmitter> emitter);
    void remove_emitter(size_t index);
    void clear_emitters();
    std::shared_ptr<ParticleEmitter> get_emitter(size_t index) const;
    
    // Particle management
    void add_particle(const Particle& particle);
    void remove_dead_particles();
    void clear_particles();
    int get_particle_count() const { return static_cast<int>(particles.size()); }
    
    // Particle properties
    void set_max_particles(int max_count) { max_particles = std::max(1, max_count); }
    void set_global_forces(const Tensor& gravity, const Tensor& wind);
    void set_render_mode(RenderMode mode) { render_mode = mode; }
    void set_global_opacity(float opacity) { global_opacity = std::clamp(opacity, 0.0f, 1.0f); }
    
    // Animation integration
    void emit_particles(float delta_time);
    void update_particles(float delta_time);
    void apply_forces(float delta_time);
    
    // Update loop
    void update(float delta_time) override;
    
    // Rendering
    Tensor render(int width, int height, const Camera2D& camera) const override;
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state) override;
    
    // Utility methods
    void set_random_seed(int seed);
    void set_performance_mode(bool optimize_distance) { optimize_for_distance = optimize_distance; }
    
    // Predefined particle effects
    static std::shared_ptr<ParticleSystem> create_fire_effect();
    static std::shared_ptr<ParticleSystem> create_smoke_effect();
    static std::shared_ptr<ParticleSystem> create_spark_effect();
    static std::shared_ptr<ParticleSystem> create_dust_effect();
    static std::shared_ptr<ParticleSystem> create_rain_effect();
    static std::shared_ptr<ParticleSystem> create_snow_effect();
    
private:
    std::mt19937 random_engine_;
    mutable Tensor cached_render_;
    mutable bool needs_update_ = true;
    
    void update_cached_data() const;
    Tensor render_points(const Camera2D& camera) const;
    Tensor render_sprites(const Camera2D& camera) const;
    Tensor render_lines(const Camera2D& camera) const;
    Tensor render_trails(const Camera2D& camera) const;
    
    bool should_cull_particle(const Particle& particle, const Camera2D& camera) const;
    Tensor apply_global_transforms(const Tensor& position, const Camera2D& camera) const;
};

/**
 * TrailParticle - Extended particle with trail rendering
 */
struct TrailParticle : Particle {
    std::vector<Tensor> trail_positions;   // Historical positions for trail
    int max_trail_length = 10;
    
    void add_trail_point(const Tensor& position);
    void update_trail();
    Tensor get_trail_position(int index) const;
};

/**
 * AudioReactiveEmitter - Emitter that responds to audio
 */
class AudioReactiveEmitter : public ParticleEmitter {
public:
    // Audio integration
    void set_audio_level(float level) { current_audio_level_ = std::clamp(level, 0.0f, 1.0f); }
    void set_frequency_data(const Tensor& frequencies) { frequency_data_ = frequencies; }
    
    // Audio-reactive properties
    void set_audio_sensitivity(float sensitivity) { audio_sensitivity_ = std::max(0.0f, sensitivity); }
    void set_frequency_bands(const std::vector<std::pair<float, float>>& bands) { frequency_bands_ = bands; }
    
    Particle create_particle() const override;
    void update(float delta_time) override;
    
private:
    float current_audio_level_ = 0.0f;
    Tensor frequency_data_;
    float audio_sensitivity_ = 1.0f;
    std::vector<std::pair<float, float>> frequency_bands_;
    
    float get_frequency_energy(const std::pair<float, float>& frequency_range) const;
};

} // namespace scene
} // namespace jin