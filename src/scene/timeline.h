#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include "scene_graph.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <algorithm>
#include <chrono>

namespace jin {
namespace scene {

/**
 * Timeline - Frame-accurate event sequencing for animations
 */
class Timeline {
public:
    /**
     * Event - Represents a single timed event in the timeline
     */
    struct Event {
        int frame_number;                    // Frame when event triggers (0-based)
        std::string event_type;              // Type: "animation_start", "sound", "transition", etc.
        std::map<std::string, std::string> params;  // Event parameters
        
        // Comparison for sorting by frame number
        bool operator<(const Event& other) const {
            return frame_number < other.frame_number;
        }
        
        bool operator==(const Event& other) const {
            return frame_number == other.frame_number && event_type == other.event_type;
        }
    };
    
    // Timeline state
    int current_frame = 0;
    float frame_rate = 24.0f;               // 24 FPS default
    float duration_seconds = 0.0f;           // Total timeline duration
    bool playing = false;
    bool looping = false;
    
    Timeline();
    ~Timeline() = default;
    
    // Event management
    void add_event(int frame, const std::string& type, const std::map<std::string, std::string>& params = {});
    void remove_event(int frame, const std::string& type);
    void clear_events();
    
    // Event querying
    std::vector<Event> get_events_at_frame(int frame) const;
    std::vector<Event> get_events_of_type(const std::string& type) const;
    std::vector<Event> get_current_events() const;
    
    // Timeline control
    void play();
    void pause();
    void stop();
    void seek(int frame_number);
    void set_frame_rate(float fps);
    
    // Advance timeline
    void step(float delta_time);
    void advance_frame();
    
    // Duration management
    void set_duration_seconds(float duration);
    float get_duration_seconds() const { return duration_seconds; }
    int get_total_frames() const;
    
    // Callbacks for event handling
    void set_event_callback(const std::string& event_type, std::function<void(const Event&)> callback);
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state);
    temporal::FrameState get_frame_state() const;
    
    // Serialization
    std::string serialize() const;
    void deserialize(const std::string& data);
    
private:
    std::vector<Event> events_;
    std::map<std::string, std::function<void(const Event&)>> event_callbacks_;
    temporal::FrameState frame_state_;
    
    void sort_events();
    void trigger_events(const std::vector<Event>& events_to_trigger);
    void update_duration();
};

/**
 * AnimationTrack - Manages a sequence of keyframes for animating properties
 */
class AnimationTrack {
public:
    /**
     * Keyframe - Single animation keyframe
     */
    struct Keyframe {
        int frame;                           // Frame number
        Tensor value;                        // Value at this frame
        std::string easing = "linear";       // Easing function: "linear", "ease_in", "ease_out", "ease_in_out"
        
        bool operator<(const Keyframe& other) const {
            return frame < other.frame;
        }
    };
    
    std::string property_name;               // Property being animated (e.g., "position", "scale", "opacity")
    std::vector<Keyframe> keyframes;
    
    AnimationTrack(const std::string& property);
    ~AnimationTrack() = default;
    
    // Keyframe management
    void add_keyframe(int frame, const Tensor& value, const std::string& easing = "linear");
    void remove_keyframe(int frame);
    void clear_keyframes();
    
    // Value sampling
    Tensor sample_value(int frame) const;
    
    // Animation control
    void set_duration(int total_frames);
    int get_duration() const;
    
    // Interpolation methods
    static Tensor interpolate_linear(const Tensor& from, const Tensor& to, float t);
    static Tensor interpolate_ease_in(const Tensor& from, const Tensor& to, float t);
    static Tensor interpolate_ease_out(const Tensor& from, const Tensor& to, float t);
    static Tensor interpolate_ease_in_out(const Tensor& from, const Tensor& to, float t);
    
private:
    void sort_keyframes();
    Tensor apply_easing(const Tensor& value, const std::string& easing) const;
};

/**
 * Animation - Complete animation with multiple tracks
 */
class Animation {
public:
    std::string name;
    std::string target_node;                 // Name of SceneNode this animation affects
    std::vector<std::shared_ptr<AnimationTrack>> tracks;
    
    Animation(const std::string& name, const std::string& target_node);
    ~Animation() = default;
    
    // Track management
    std::shared_ptr<AnimationTrack> add_track(const std::string& property);
    void remove_track(const std::string& property);
    std::shared_ptr<AnimationTrack> get_track(const std::string& property) const;
    
    // Animation playback
    void play(int start_frame = 0);
    void stop();
    void set_looping(bool looping);
    
    // Apply animation at specific frame
    void apply(int frame, SceneNode& target_node) const;
    
    // Duration
    void set_duration(int total_frames);
    int get_duration() const;
    
private:
    bool playing_ = false;
    int start_frame_ = 0;
    bool looping_ = false;
    
public:
    bool is_playing() const { return playing_; }
};

/**
 * AnimationController - Manages multiple animations and timeline playback
 */
class AnimationController {
public:
    std::vector<std::shared_ptr<Animation>> animations;
    Timeline timeline;
    
    AnimationController();
    ~AnimationController() = default;
    
    // Animation management
    std::shared_ptr<Animation> add_animation(const std::string& name, const std::string& target_node);
    void remove_animation(const std::string& name);
    std::shared_ptr<Animation> get_animation(const std::string& name) const;
    
    // Timeline integration
    void play();
    void pause();
    void stop();
    void seek(int frame);
    
    // Update and apply all animations
    void update(class SceneGraph& scene_graph, float delta_time);
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state);
};

} // namespace scene
} // namespace jin