#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>
#include <unordered_map>

namespace jin {
namespace scene {

// Forward declarations
class SceneNode;
class SceneGraph;
class Camera2D;

/**
 * Transform2D - 2D transformation (position, rotation, scale)
 */
struct Transform2D {
    Tensor position;    // [x, y] - relative to parent
    Tensor rotation;   // [angle] in radians
    Tensor scale;       // [x, y] - uniform scale
    
    Transform2D() {
        position = Tensor({2}, jin::DType::FP32, jin::Device::cpu());
        rotation = Tensor({1}, jin::DType::FP32, jin::Device::cpu());
        scale = Tensor({2}, jin::DType::FP32, jin::Device::cpu());
        
        position.fill(0.0f);
        rotation.fill(0.0f);
        scale.fill(1.0f);
    }
    
    // Apply transform to a point
    Tensor transform_point(const Tensor& point) const;
    
    // Get world transform matrix (3x3)
    Tensor get_transform_matrix() const;
    
    // Compose with another transform
    Transform2D compose(const Transform2D& other) const;
};

/**
 * SceneNode - Base class for all scene graph nodes
 */
class SceneNode {
public:
    std::string name;
    Transform2D local_transform;
    bool visible = true;
    int z_order = 0;
    float opacity = 1.0f;
    
    // Parent-child relationships
    std::weak_ptr<SceneNode> parent;
    std::vector<std::shared_ptr<SceneNode>> children;
    
    // Lifecycle
    virtual ~SceneNode() = default;
    
    // Transform management
    Transform2D get_world_transform() const;
    void set_world_position(const Tensor& position);
    void set_local_position(const Tensor& position);
    void set_rotation(float angle_radians);
    void set_scale(const Tensor& scale);
    
    // Hierarchy management
    void add_child(std::shared_ptr<SceneNode> child);
    void remove_child(const std::string& child_name);
    void remove_child(std::shared_ptr<SceneNode> child);
    std::shared_ptr<SceneNode> find_child(const std::string& name) const;
    
    // Traversal
    void traverse_depth_first(std::function<void(SceneNode&)> visitor);
    void traverse_breadth_first(std::function<void(SceneNode&)> visitor);
    
    // Update hook
    virtual void update(float delta_time);
    
    // Render hook (to be implemented by subclasses)
    virtual Tensor render(int width, int height, const Camera2D& camera) const;
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state);
    temporal::FrameState get_frame_state() const;
    
protected:
    temporal::FrameState frame_state_;
    
    void mark_dirty();
    bool is_dirty() const { return dirty_; }
    
private:
    bool dirty_ = true;
    mutable Transform2D cached_world_transform_;
    mutable bool world_transform_valid_ = false;
    
    void invalidate_world_transform();
};

/**
 * SceneGraph - Root node and scene management
 */
class SceneGraph {
public:
    std::shared_ptr<SceneNode> root;
    
    SceneGraph();
    ~SceneGraph() = default;
    
    // Node management
    std::shared_ptr<SceneNode> find_node(const std::string& name);
    void add_node(const std::string& name, std::shared_ptr<SceneNode> node);
    void remove_node(const std::string& name);
    
    // Traversal
    void traverse(std::function<void(SceneNode&)> visitor);
    void traverse_depth_first(std::function<void(SceneNode&)> visitor);
    void traverse_breadth_first(std::function<void(SceneNode&)> visitor);
    
    // Update all nodes
    void update(float delta_time);
    
    // Global dirty flag management
    void mark_all_dirty();
    
    // Frame state management
    void set_global_frame_state(const temporal::FrameState& frame_state);
    temporal::FrameState get_global_frame_state() const;
    
private:
    std::unordered_map<std::string, std::shared_ptr<SceneNode>> nodes_;
    temporal::FrameState global_frame_state_;
};

/**
 * Camera2D - 2D camera for scene viewing
 */
class Camera2D {
public:
    Tensor position;      // Camera position in world space
    float rotation = 0.0f;  // Camera rotation in radians
    float zoom = 1.0f;      // Camera zoom factor
    Tensor viewport_size;   // [width, height] in pixels
    
    Camera2D() {
        position = Tensor({2}, jin::DType::FP32);
        viewport_size = Tensor({2}, jin::DType::FP32);
        
        position.fill(0.0f);
        viewport_size.data_ptr<float>()[0] = 1280.0f;
        viewport_size.data_ptr<float>()[1] = 720.0f;
    }
    
    // Convert world coordinates to screen coordinates
    Tensor world_to_screen(const Tensor& world_point) const;
    
    // Convert screen coordinates to world coordinates
    Tensor screen_to_world(const Tensor& screen_point) const;
    
    // Get camera transform matrix
    Tensor get_view_matrix() const;
    
    // Get projection matrix (orthographic)
    Tensor get_projection_matrix() const;
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state);
};

} // namespace scene
} // namespace jin