#include "scene_graph.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>

namespace jin {
namespace scene {

// Transform2D implementation
Tensor Transform2D::transform_point(const Tensor& point) const {
    // Apply 2D transformation: rotation -> scale -> translation
    Tensor result = point;
    
    // Apply scale
    if (scale.size() >= 2) {
        result = result * scale;
    }
    
    // Apply rotation
    if (rotation.size() >= 1 && result.size() >= 2) {
        float angle = rotation.data_ptr<float>()[0];
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);
        
        if (result.size() >= 2) {
            float x = result.data_ptr<float>()[0];
            float y = result.data_ptr<float>()[1];
            result.data_ptr<float>()[0] = x * cos_a - y * sin_a;
            result.data_ptr<float>()[1] = x * sin_a + y * cos_a;
        }
    }
    
    // Apply translation
    if (position.size() >= 2) {
        result = result + position;
    }
    
    return result;
}

Tensor Transform2D::get_transform_matrix() const {
    // Create 3x3 transformation matrix for 2D graphics
    // [ s_x*cos(θ)  -s_y*sin(θ)  t_x ]
    // [ s_x*sin(θ)   s_y*cos(θ)  t_y ]
    // [     0             0       1   ]
    
    Tensor matrix = zeros({3, 3});
    float* data = matrix.data_ptr<float>();
    
    float sx = (scale.size() >= 2) ? scale.data_ptr<float>()[0] : 1.0f;
    float sy = (scale.size() >= 2) ? scale.data_ptr<float>()[1] : 1.0f;
    float angle = (rotation.size() >= 1) ? rotation.data_ptr<float>()[0] : 0.0f;
    float tx = (position.size() >= 2) ? position.data_ptr<float>()[0] : 0.0f;
    float ty = (position.size() >= 2) ? position.data_ptr<float>()[1] : 0.0f;
    
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);
    
    // Row 0
    data[0] = sx * cos_a;
    data[1] = -sy * sin_a;
    data[2] = tx;
    
    // Row 1
    data[3] = sx * sin_a;
    data[4] = sy * cos_a;
    data[5] = ty;
    
    // Row 2
    data[6] = 0.0f;
    data[7] = 0.0f;
    data[8] = 1.0f;
    
    return matrix;
}

Transform2D Transform2D::compose(const Transform2D& other) const {
    // Compose this transform with another (this * other)
    Transform2D result;
    
    // Combine scales
    if (scale.size() >= 2 && other.scale.size() >= 2) {
        result.scale = Tensor({
            scale.data_ptr<float>()[0] * other.scale.data_ptr<float>()[0],
            scale.data_ptr<float>()[1] * other.scale.data_ptr<float>()[1]
        });
    }
    
    // Combine rotations
    if (rotation.size() >= 1 && other.rotation.size() >= 1) {
        result.rotation = Tensor({
            rotation.data_ptr<float>()[0] + other.rotation.data_ptr<float>()[0]
        });
    }
    
    // Combine positions (accounting for parent's transform)
    if (position.size() >= 2 && other.position.size() >= 2) {
        // Apply this transform to other's position
        Tensor combined_pos = other.position;
        
        // Apply rotation
        if (rotation.size() >= 1) {
            float angle = rotation.data_ptr<float>()[0];
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            
            if (combined_pos.size() >= 2) {
                float x = combined_pos.data_ptr<float>()[0];
                float y = combined_pos.data_ptr<float>()[1];
                combined_pos.data_ptr<float>()[0] = x * cos_a - y * sin_a;
                combined_pos.data_ptr<float>()[1] = x * sin_a + y * cos_a;
            }
        }
        
        // Apply scale
        if (scale.size() >= 2) {
            combined_pos = combined_pos * scale;
        }
        
        // Add to this position
        result.position = position + combined_pos;
    }
    
    return result;
}

// SceneNode implementation
Transform2D SceneNode::get_world_transform() const {
    if (!world_transform_valid_) {
        cached_world_transform_ = local_transform;
        
        // Walk up the hierarchy and compose transforms
        std::shared_ptr<SceneNode> current = parent.lock();
        while (current) {
            cached_world_transform_ = current->local_transform.compose(cached_world_transform_);
            current = current->parent.lock();
        }
        
        world_transform_valid_ = true;
    }
    
    return cached_world_transform_;
}

void SceneNode::set_world_position(const Tensor& position) {
    // Calculate local position from desired world position
    Tensor local_pos = position;
    
    // Account for parent's world transform
    std::shared_ptr<SceneNode> current = parent.lock();
    while (current) {
        Transform2D parent_transform = current->get_world_transform();
        // Inverse transform: local_pos = world_pos - parent_pos (simplified)
        local_pos = local_pos - parent_transform.position;
        current = current->parent.lock();
    }
    
    set_local_position(local_pos);
}

void SceneNode::set_local_position(const Tensor& position) {
    local_transform.position = position;
    mark_dirty();
}

void SceneNode::set_rotation(float angle_radians) {
    local_transform.rotation = Tensor({angle_radians});
    mark_dirty();
}

void SceneNode::set_scale(const Tensor& scale) {
    local_transform.scale = scale;
    mark_dirty();
}

void SceneNode::add_child(std::shared_ptr<SceneNode> child) {
    if (!child) return;
    
    // Remove from current parent if any
    if (!child->parent.expired()) {
        auto current_parent = child->parent.lock();
        if (current_parent) {
            current_parent->remove_child(child);
        }
    }
    
    children.push_back(child);
    child->parent = shared_from_this();
    mark_dirty();
}

void SceneNode::remove_child(const std::string& child_name) {
    auto it = std::find_if(children.begin(), children.end(),
        [&child_name](const std::shared_ptr<SceneNode>& child) {
            return child->name == child_name;
        });
    
    if (it != children.end()) {
        (*it)->parent.reset();
        children.erase(it);
        mark_dirty();
    }
}

void SceneNode::remove_child(std::shared_ptr<SceneNode> child) {
    auto it = std::find(children.begin(), children.end(), child);
    if (it != children.end()) {
        (*it)->parent.reset();
        children.erase(it);
        mark_dirty();
    }
}

std::shared_ptr<SceneNode> SceneNode::find_child(const std::string& name) const {
    for (const auto& child : children) {
        if (child->name == name) {
            return child;
        }
    }
    return nullptr;
}

void SceneNode::traverse_depth_first(std::function<void(SceneNode&)> visitor) {
    visitor(*this);
    for (auto& child : children) {
        child->traverse_depth_first(visitor);
    }
}

void SceneNode::traverse_breadth_first(std::function<void(SceneNode&)> visitor) {
    std::vector<SceneNode*> current_level;
    current_level.push_back(this);
    
    while (!current_level.empty()) {
        std::vector<SceneNode*> next_level;
        for (SceneNode* node : current_level) {
            visitor(*node);
            for (auto& child : node->children) {
                next_level.push_back(child.get());
            }
        }
        current_level = next_level;
    }
}

void SceneNode::update(float delta_time) {
    // Default implementation: update frame state if available
    if (frame_state_.frame_index >= 0) {
        frame_state_.timestamp += delta_time;
    }
}

Tensor SceneNode::render(int width, int height, const Camera2D& camera) const {
    // Default implementation: return empty tensor
    return zeros({height, width, 4}); // RGBA
}

void SceneNode::set_frame_state(const temporal::FrameState& frame_state) {
    frame_state_ = frame_state;
}

temporal::FrameState SceneNode::get_frame_state() const {
    return frame_state_;
}

void SceneNode::mark_dirty() {
    dirty_ = true;
    world_transform_valid_ = false;
    
    // Propagate to children
    for (auto& child : children) {
        child->mark_dirty();
    }
}

void SceneNode::invalidate_world_transform() {
    world_transform_valid_ = false;
}

// SceneGraph implementation
SceneGraph::SceneGraph() {
    root = std::make_shared<SceneNode>();
    root->name = "root";
}

std::shared_ptr<SceneNode> SceneGraph::find_node(const std::string& name) {
    if (root->name == name) {
        return root;
    }
    
    auto it = nodes_.find(name);
    return (it != nodes_.end()) ? it->second : nullptr;
}

void SceneGraph::add_node(const std::string& name, std::shared_ptr<SceneNode> node) {
    if (!node) return;
    
    node->name = name;
    nodes_[name] = node;
    
    // If this is the first node or no root children, add to root
    if (root->children.empty()) {
        root->add_child(node);
    }
}

void SceneGraph::remove_node(const std::string& name) {
    auto it = nodes_.find(name);
    if (it != nodes_.end()) {
        // Remove from root's children if present
        root->remove_child(it->second);
        nodes_.erase(it);
    }
}

void SceneGraph::traverse(std::function<void(SceneNode&)> visitor) {
    if (root) {
        root->traverse_depth_first(visitor);
    }
}

void SceneGraph::traverse_depth_first(std::function<void(SceneNode&)> visitor) {
    if (root) {
        root->traverse_depth_first(visitor);
    }
}

void SceneGraph::traverse_breadth_first(std::function<void(SceneNode&)> visitor) {
    if (root) {
        root->traverse_breadth_first(visitor);
    }
}

void SceneGraph::update(float delta_time) {
    if (root) {
        root->traverse_depth_first([delta_time](SceneNode& node) {
            node.update(delta_time);
        });
    }
}

void SceneGraph::mark_all_dirty() {
    if (root) {
        root->traverse_depth_first([](SceneNode& node) {
            node.mark_dirty();
        });
    }
}

void SceneGraph::set_global_frame_state(const temporal::FrameState& frame_state) {
    global_frame_state_ = frame_state;
}

temporal::FrameState SceneGraph::get_global_frame_state() const {
    return global_frame_state_;
}

// Camera2D implementation
Tensor Camera2D::world_to_screen(const Tensor& world_point) const {
    // Simple orthographic projection
    Tensor screen_point = world_point - position;
    
    // Apply zoom
    if (zoom != 1.0f) {
        screen_point = screen_point / zoom;
    }
    
    // Apply rotation (simplified - just for camera rotation)
    if (rotation != 0.0f) {
        float cos_a = std::cos(rotation);
        float sin_a = std::sin(rotation);
        
        if (screen_point.size() >= 2) {
            float x = screen_point.data_ptr<float>()[0];
            float y = screen_point.data_ptr<float>()[1];
            screen_point.data_ptr<float>()[0] = x * cos_a - y * sin_a;
            screen_point.data_ptr<float>()[1] = x * sin_a + y * cos_a;
        }
    }
    
    // Convert to screen coordinates (origin at center)
    if (screen_point.size() >= 2 && viewport_size.size() >= 2) {
        screen_point.data_ptr<float>()[0] += viewport_size.data_ptr<float>()[0] / 2.0f;
        screen_point.data_ptr<float>()[1] += viewport_size.data_ptr<float>()[1] / 2.0f;
    }
    
    return screen_point;
}

Tensor Camera2D::screen_to_world(const Tensor& screen_point) const {
    // Convert from screen coordinates (origin at center)
    Tensor world_point = screen_point;
    
    if (world_point.size() >= 2 && viewport_size.size() >= 2) {
        world_point.data_ptr<float>()[0] -= viewport_size.data_ptr<float>()[0] / 2.0f;
        world_point.data_ptr<float>()[1] -= viewport_size.data_ptr<float>()[1] / 2.0f;
    }
    
    // Apply inverse rotation
    if (rotation != 0.0f) {
        float cos_a = std::cos(-rotation);
        float sin_a = std::sin(-rotation);
        
        if (world_point.size() >= 2) {
            float x = world_point.data_ptr<float>()[0];
            float y = world_point.data_ptr<float>()[1];
            world_point.data_ptr<float>()[0] = x * cos_a - y * sin_a;
            world_point.data_ptr<float>()[1] = x * sin_a + y * cos_a;
        }
    }
    
    // Apply inverse zoom
    if (zoom != 1.0f) {
        world_point = world_point * zoom;
    }
    
    // Add camera position
    world_point = world_point + position;
    
    return world_point;
}

Tensor Camera2D::get_view_matrix() const {
    // View matrix for orthographic camera
    Tensor view_matrix = zeros({3, 3});
    float* data = view_matrix.data_ptr<float>();
    
    // Apply zoom
    data[0] = zoom;
    data[4] = zoom;
    
    // Apply rotation
    if (rotation != 0.0f) {
        float cos_a = std::cos(-rotation);
        float sin_a = std::sin(-rotation);
        float temp = data[0];
        data[0] = temp * cos_a;
        data[3] = -temp * sin_a;
        temp = data[4];
        data[1] = temp * sin_a;
        data[4] = temp * cos_a;
    }
    
    // Apply translation
    if (position.size() >= 2) {
        data[2] = -position.data_ptr<float>()[0];
        data[5] = -position.data_ptr<float>()[1];
    }
    
    data[8] = 1.0f;
    
    return view_matrix;
}

Tensor Camera2D::get_projection_matrix() const {
    // Orthographic projection matrix
    Tensor proj_matrix = zeros({3, 3});
    float* data = proj_matrix.data_ptr<float>();
    
    if (viewport_size.size() >= 2) {
        float width = viewport_size.data_ptr<float>()[0];
        float height = viewport_size.data_ptr<float>()[1];
        
        // Scale to screen space
        data[0] = 2.0f / width;
        data[4] = 2.0f / height;
        data[8] = 1.0f;
        
        // Translate to screen center
        data[2] = -1.0f;
        data[5] = -1.0f;
    }
    
    return proj_matrix;
}

void Camera2D::set_frame_state(const temporal::FrameState& frame_state) {
    // Update camera position based on frame state motion
    if (frame_state.motion_code.size() >= 2) {
        position = frame_state.motion_code.view({2}).slice(0, 2);
    }
}

} // namespace scene
} // namespace jin