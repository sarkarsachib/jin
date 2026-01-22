#include "timeline.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace jin {
namespace scene {

// Timeline implementation
Timeline::Timeline() : current_frame(0), frame_rate(24.0f), duration_seconds(0.0f), 
                       playing(false), looping(false) {}

void Timeline::add_event(int frame, const std::string& type, const std::map<std::string, std::string>& params) {
    Event event;
    event.frame_number = frame;
    event.event_type = type;
    event.params = params;
    
    events_.push_back(event);
    sort_events();
    update_duration();
}

void Timeline::remove_event(int frame, const std::string& type) {
    events_.erase(
        std::remove_if(events_.begin(), events_.end(),
            [frame, type](const Event& event) {
                return event.frame_number == frame && event.event_type == type;
            }),
        events_.end()
    );
    update_duration();
}

void Timeline::clear_events() {
    events_.clear();
    update_duration();
}

std::vector<Timeline::Event> Timeline::get_events_at_frame(int frame) const {
    std::vector<Event> result;
    for (const auto& event : events_) {
        if (event.frame_number == frame) {
            result.push_back(event);
        }
    }
    return result;
}

std::vector<Timeline::Event> Timeline::get_events_of_type(const std::string& type) const {
    std::vector<Event> result;
    for (const auto& event : events_) {
        if (event.event_type == type) {
            result.push_back(event);
        }
    }
    return result;
}

std::vector<Timeline::Event> Timeline::get_current_events() const {
    return get_events_at_frame(current_frame);
}

void Timeline::play() {
    playing = true;
}

void Timeline::pause() {
    playing = false;
}

void Timeline::stop() {
    playing = false;
    current_frame = 0;
}

void Timeline::seek(int frame_number) {
    int total_frames = get_total_frames();
    current_frame = std::max(0, std::min(frame_number, total_frames));
    
    // Trigger events at new frame
    auto events = get_current_events();
    trigger_events(events);
}

void Timeline::set_frame_rate(float fps) {
    frame_rate = std::max(1.0f, fps);
    update_duration();
}

void Timeline::step(float delta_time) {
    if (!playing) return;
    
    // Convert delta time to frame advancement
    float frames_to_advance = delta_time * frame_rate;
    int new_frame = current_frame + static_cast<int>(std::round(frames_to_advance));
    
    // Check for looping
    int total_frames = get_total_frames();
    if (new_frame >= total_frames) {
        if (looping) {
            new_frame = 0;
        } else {
            new_frame = total_frames - 1;
            playing = false;
        }
    }
    
    // Trigger events at new frame
    auto events = get_events_at_frame(new_frame);
    trigger_events(events);
    
    current_frame = new_frame;
}

void Timeline::advance_frame() {
    if (!playing) return;
    
    int total_frames = get_total_frames();
    int new_frame = current_frame + 1;
    
    if (new_frame >= total_frames) {
        if (looping) {
            new_frame = 0;
        } else {
            new_frame = total_frames - 1;
            playing = false;
        }
    }
    
    current_frame = new_frame;
}

void Timeline::set_duration_seconds(float duration) {
    duration_seconds = std::max(0.0f, duration);
}

int Timeline::get_total_frames() const {
    if (events_.empty()) return 0;
    
    int max_frame = 0;
    for (const auto& event : events_) {
        max_frame = std::max(max_frame, event.frame_number);
    }
    return max_frame + 1; // +1 because frames are 0-based
}

void Timeline::set_event_callback(const std::string& event_type, std::function<void(const Event&)> callback) {
    event_callbacks_[event_type] = callback;
}

void Timeline::set_frame_state(const temporal::FrameState& frame_state) {
    frame_state_ = frame_state;
}

temporal::FrameState Timeline::get_frame_state() const {
    return frame_state_;
}

std::string Timeline::serialize() const {
    // Simple JSON-like serialization
    std::stringstream ss;
    ss << "{";
    ss << "\"current_frame\":" << current_frame << ",";
    ss << "\"frame_rate\":" << frame_rate << ",";
    ss << "\"duration_seconds\":" << duration_seconds << ",";
    ss << "\"playing\":" << (playing ? "true" : "false") << ",";
    ss << "\"looping\":" << (looping ? "true" : "false") << ",";
    ss << "\"events\":[";
    
    for (size_t i = 0; i < events_.size(); ++i) {
        const auto& event = events_[i];
        ss << "{";
        ss << "\"frame_number\":" << event.frame_number << ",";
        ss << "\"event_type\":\"" << event.event_type << "\",";
        ss << "\"params\":{";
        
        for (auto it = event.params.begin(); it != event.params.end(); ++it) {
            ss << "\"" << it->first << "\":\"" << it->second << "\"";
            if (std::next(it) != event.params.end()) ss << ",";
        }
        ss << "}";
        ss << "}";
        if (i < events_.size() - 1) ss << ",";
    }
    
    ss << "]}";
    return ss.str();
}

void Timeline::deserialize(const std::string& data) {
    // Basic JSON parsing (simplified)
    // This is a very basic implementation - in production, use a proper JSON library
    events_.clear();
    
    // Find events array
    size_t events_start = data.find("\"events\":[");
    if (events_start == std::string::npos) return;
    
    size_t events_end = data.find("]", events_start);
    if (events_end == std::string::npos) return;
    
    std::string events_str = data.substr(events_start + 10, events_end - events_start - 10);
    
    // Simple parsing of individual events
    size_t pos = 0;
    while (pos < events_str.length()) {
        size_t event_start = events_str.find("{", pos);
        if (event_start == std::string::npos) break;
        
        size_t event_end = events_str.find("}", event_start);
        if (event_end == std::string::npos) break;
        
        std::string event_str = events_str.substr(event_start + 1, event_end - event_start - 1);
        
        Event event;
        
        // Parse frame_number
        size_t frame_pos = event_str.find("\"frame_number\":");
        if (frame_pos != std::string::npos) {
            size_t value_start = event_str.find(":", frame_pos) + 1;
            size_t value_end = event_str.find(",", value_start);
            if (value_end == std::string::npos) value_end = event_str.find("}", value_start);
            event.frame_number = std::stoi(event_str.substr(value_start, value_end - value_start));
        }
        
        // Parse event_type
        size_t type_pos = event_str.find("\"event_type\":");
        if (type_pos != std::string::npos) {
            size_t value_start = event_str.find(":", type_pos) + 2; // Skip ":"
            size_t value_end = event_str.find("\"", value_start + 1);
            event.event_type = event_str.substr(value_start, value_end - value_start);
        }
        
        events_.push_back(event);
        pos = event_end + 1;
    }
    
    sort_events();
    update_duration();
}

void Timeline::sort_events() {
    std::sort(events_.begin(), events_.end());
}

void Timeline::trigger_events(const std::vector<Event>& events_to_trigger) {
    for (const auto& event : events_to_trigger) {
        auto it = event_callbacks_.find(event.event_type);
        if (it != event_callbacks_.end()) {
            it->second(event);
        }
    }
}

void Timeline::update_duration() {
    int total_frames = get_total_frames();
    duration_seconds = static_cast<float>(total_frames) / frame_rate;
}

// AnimationTrack implementation
AnimationTrack::AnimationTrack(const std::string& property) : property_name(property) {}

void AnimationTrack::add_keyframe(int frame, const Tensor& value, const std::string& easing) {
    Keyframe keyframe;
    keyframe.frame = frame;
    keyframe.value = value;
    keyframe.easing = easing;
    
    keyframes.push_back(keyframe);
    sort_keyframes();
}

void AnimationTrack::remove_keyframe(int frame) {
    keyframes.erase(
        std::remove_if(keyframes.begin(), keyframes.end(),
            [frame](const Keyframe& kf) { return kf.frame == frame; }),
        keyframes.end()
    );
}

void AnimationTrack::clear_keyframes() {
    keyframes.clear();
}

Tensor AnimationTrack::sample_value(int frame) const {
    if (keyframes.empty()) {
        return Tensor({1}); // Return scalar tensor
    }
    
    if (keyframes.size() == 1 || frame <= keyframes.front().frame) {
        return keyframes.front().value;
    }
    
    if (frame >= keyframes.back().frame) {
        return keyframes.back().value;
    }
    
    // Find surrounding keyframes
    size_t next_index = 0;
    for (size_t i = 0; i < keyframes.size() - 1; ++i) {
        if (frame >= keyframes[i].frame && frame <= keyframes[i + 1].frame) {
            next_index = i + 1;
            break;
        }
    }
    
    const Keyframe& from = keyframes[next_index - 1];
    const Keyframe& to = keyframes[next_index];
    
    // Calculate interpolation factor
    float t = static_cast<float>(frame - from.frame) / static_cast<float>(to.frame - from.frame);
    t = std::clamp(t, 0.0f, 1.0f);
    
    // Apply easing
    if (from.easing == "ease_in") {
        t = t * t;
    } else if (from.easing == "ease_out") {
        t = 1.0f - (1.0f - t) * (1.0f - t);
    } else if (from.easing == "ease_in_out") {
        if (t < 0.5f) {
            t = 2.0f * t * t;
        } else {
            t = 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;
        }
    }
    
    return interpolate_linear(from.value, to.value, t);
}

void AnimationTrack::set_duration(int total_frames) {
    // Ensure there's a keyframe at the end
    if (keyframes.empty() || keyframes.back().frame < total_frames - 1) {
        // Create final keyframe if needed
        Tensor last_value = keyframes.empty() ? Tensor({1}) : keyframes.back().value;
        add_keyframe(total_frames - 1, last_value, keyframes.back().easing);
    }
}

int AnimationTrack::get_duration() const {
    if (keyframes.empty()) return 0;
    return keyframes.back().frame;
}

Tensor AnimationTrack::interpolate_linear(const Tensor& from, const Tensor& to, float t) {
    return from * (1.0f - t) + to * t;
}

Tensor AnimationTrack::interpolate_ease_in(const Tensor& from, const Tensor& to, float t) {
    t = t * t;
    return from * (1.0f - t) + to * t;
}

Tensor AnimationTrack::interpolate_ease_out(const Tensor& from, const Tensor& to, float t) {
    t = 1.0f - (1.0f - t) * (1.0f - t);
    return from * (1.0f - t) + to * t;
}

Tensor AnimationTrack::interpolate_ease_in_out(const Tensor& from, const Tensor& to, float t) {
    if (t < 0.5f) {
        t = 2.0f * t * t;
    } else {
        t = 1.0f - std::pow(-2.0f * t + 2.0f, 2.0f) / 2.0f;
    }
    return from * (1.0f - t) + to * t;
}

void AnimationTrack::sort_keyframes() {
    std::sort(keyframes.begin(), keyframes.end());
}

// Animation implementation
Animation::Animation(const std::string& name, const std::string& target_node) 
    : name(name), target_node(target_node) {}

std::shared_ptr<AnimationTrack> Animation::add_track(const std::string& property) {
    auto track = std::make_shared<AnimationTrack>(property);
    tracks.push_back(track);
    return track;
}

void Animation::remove_track(const std::string& property) {
    tracks.erase(
        std::remove_if(tracks.begin(), tracks.end(),
            [&property](const std::shared_ptr<AnimationTrack>& track) {
                return track->property_name == property;
            }),
        tracks.end()
    );
}

std::shared_ptr<AnimationTrack> Animation::get_track(const std::string& property) const {
    for (const auto& track : tracks) {
        if (track->property_name == property) {
            return track;
        }
    }
    return nullptr;
}

void Animation::play(int start_frame) {
    playing_ = true;
    start_frame_ = start_frame;
}

void Animation::stop() {
    playing_ = false;
}

void Animation::set_looping(bool looping) {
    looping_ = looping;
}

void Animation::apply(int frame, class SceneNode& target_node) const {
    for (const auto& track : tracks) {
        Tensor value = track->sample_value(frame);
        
        // Apply the value to the appropriate property
        if (track->property_name == "position") {
            // Handle position (x, y)
            if (value.size() >= 2) {
                Tensor pos = value.view({2}).slice(0, 2);
                target_node.set_local_position(pos);
            }
        } else if (track->property_name == "rotation") {
            // Handle rotation (angle)
            if (value.size() >= 1) {
                float angle = value.data_ptr<float>()[0];
                target_node.set_rotation(angle);
            }
        } else if (track->property_name == "scale") {
            // Handle scale (x, y)
            if (value.size() >= 2) {
                Tensor scale = value.view({2}).slice(0, 2);
                target_node.set_scale(scale);
            }
        } else if (track->property_name == "opacity") {
            // Handle opacity (0.0-1.0)
            if (value.size() >= 1) {
                float opacity = value.data_ptr<float>()[0];
                target_node.opacity = opacity;
            }
        }
    }
}

void Animation::set_duration(int total_frames) {
    for (auto& track : tracks) {
        track->set_duration(total_frames);
    }
}

int Animation::get_duration() const {
    int max_duration = 0;
    for (const auto& track : tracks) {
        max_duration = std::max(max_duration, track->get_duration());
    }
    return max_duration;
}

// AnimationController implementation
AnimationController::AnimationController() {}

std::shared_ptr<Animation> AnimationController::add_animation(const std::string& name, const std::string& target_node) {
    auto animation = std::make_shared<Animation>(name, target_node);
    animations.push_back(animation);
    return animation;
}

void AnimationController::remove_animation(const std::string& name) {
    animations.erase(
        std::remove_if(animations.begin(), animations.end(),
            [&name](const std::shared_ptr<Animation>& anim) {
                return anim->name == name;
            }),
        animations.end()
    );
}

std::shared_ptr<Animation> AnimationController::get_animation(const std::string& name) const {
    for (const auto& animation : animations) {
        if (animation->name == name) {
            return animation;
        }
    }
    return nullptr;
}

void AnimationController::play() {
    timeline.play();
    for (const auto& animation : animations) {
        animation->play();
    }
}

void AnimationController::pause() {
    timeline.pause();
}

void AnimationController::stop() {
    timeline.stop();
    for (const auto& animation : animations) {
        animation->stop();
    }
}

void AnimationController::seek(int frame) {
    timeline.seek(frame);
    
    // Apply animations at new frame
    // This would need to be connected to the scene graph
    // For now, it's a placeholder for the API
}

void AnimationController::update(class SceneGraph& scene_graph, float delta_time) {
    timeline.step(delta_time);
    
    // Apply all animations at current timeline frame
    int current_frame = timeline.current_frame;
    
    for (const auto& animation : animations) {
        if (animation->is_playing()) {
            auto target_node = scene_graph.find_node(animation->target_node);
            if (target_node) {
                animation->apply(current_frame, *target_node);
            }
        }
    }
}

void AnimationController::set_frame_state(const temporal::FrameState& frame_state) {
    timeline.set_frame_state(frame_state);
}

} // namespace scene
} // namespace jin