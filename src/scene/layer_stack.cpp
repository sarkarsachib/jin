#include "layer_stack.h"
#include <iostream>
#include <set>

namespace jin {
namespace scene {

// LayerStack implementation
LayerStack::LayerStack() {}

void LayerStack::add_layer(int z_order, std::shared_ptr<SceneNode> node, const std::string& name) {
    if (!node) return;
    
    Layer layer;
    layer.z_order = z_order;
    layer.node = node;
    layer.name = name.empty() ? "layer_" + std::to_string(layers_.size()) : name;
    
    layers_.push_back(layer);
    update_layer_index();
    sort_layers();
}

void LayerStack::remove_layer(const std::string& name) {
    size_t index = find_layer_by_name(name);
    if (index < layers_.size()) {
        layers_.erase(layers_.begin() + index);
        update_layer_index();
    }
}

void LayerStack::remove_layer(std::shared_ptr<SceneNode> node) {
    auto it = std::find_if(layers_.begin(), layers_.end(),
        [&node](const Layer& layer) {
            return layer.node == node;
        });
    
    if (it != layers_.end()) {
        layers_.erase(it);
        update_layer_index();
    }
}

void LayerStack::clear_layers() {
    layers_.clear();
    layer_index_by_name_.clear();
}

void LayerStack::set_layer_visibility(const std::string& name, bool visible) {
    Layer* layer = get_layer(name);
    if (layer) {
        layer->visible = visible;
    }
}

void LayerStack::set_layer_opacity(const std::string& name, float opacity) {
    Layer* layer = get_layer(name);
    if (layer) {
        layer->opacity = std::clamp(opacity, 0.0f, 1.0f);
    }
}

void LayerStack::set_layer_z_order(const std::string& name, int z_order) {
    Layer* layer = get_layer(name);
    if (layer) {
        layer->z_order = z_order;
        sort_layers();
    }
}

std::shared_ptr<SceneNode> LayerStack::get_layer_node(const std::string& name) const {
    const Layer* layer = get_layer(name);
    return layer ? layer->node : nullptr;
}

Layer* LayerStack::get_layer(const std::string& name) {
    size_t index = find_layer_by_name(name);
    if (index < layers_.size()) {
        return &layers_[index];
    }
    return nullptr;
}

const Layer* LayerStack::get_layer(const std::string& name) const {
    size_t index = find_layer_by_name(name);
    if (index < layers_.size()) {
        return &layers_[index];
    }
    return nullptr;
}

void LayerStack::sort_layers() {
    std::sort(layers_.begin(), layers_.end());
}

std::vector<Layer*> LayerStack::get_render_order() {
    std::vector<Layer*> render_order;
    for (auto& layer : layers_) {
        if (layer.visible) {
            render_order.push_back(&layer);
        }
    }
    return render_order;
}

std::vector<const Layer*> LayerStack::get_render_order() const {
    std::vector<const Layer*> render_order;
    for (const auto& layer : layers_) {
        if (layer.visible) {
            render_order.push_back(&layer);
        }
    }
    return render_order;
}

std::vector<std::shared_ptr<SceneNode>> LayerStack::get_visible_nodes() const {
    std::vector<std::shared_ptr<SceneNode>> visible_nodes;
    for (const auto& layer : layers_) {
        if (layer.visible) {
            visible_nodes.push_back(layer.node);
        }
    }
    return visible_nodes;
}

std::vector<Layer*> LayerStack::get_layers_in_range(int min_z, int max_z) {
    std::vector<Layer*> layers_in_range;
    for (auto& layer : layers_) {
        if (layer.z_order >= min_z && layer.z_order <= max_z && layer.visible) {
            layers_in_range.push_back(&layer);
        }
    }
    return layers_in_range;
}

std::vector<Layer*> LayerStack::get_layers_above(int z_order) {
    std::vector<Layer*> layers_above;
    for (auto& layer : layers_) {
        if (layer.z_order > z_order && layer.visible) {
            layers_above.push_back(&layer);
        }
    }
    return layers_above;
}

std::vector<Layer*> LayerStack::get_layers_below(int z_order) {
    std::vector<Layer*> layers_below;
    for (auto& layer : layers_) {
        if (layer.z_order < z_order && layer.visible) {
            layers_below.push_back(&layer);
        }
    }
    return layers_below;
}

void LayerStack::set_frame_state(const temporal::FrameState& frame_state) {
    frame_state_ = frame_state;
    
    // Update all visible nodes with frame state
    for (const auto& layer : layers_) {
        if (layer.visible && layer.node) {
            layer.node->set_frame_state(frame_state);
        }
    }
}

temporal::FrameState LayerStack::get_frame_state() const {
    return frame_state_;
}

void LayerStack::print_layers() const {
    std::cout << "LayerStack (" << layers_.size() << " layers):\n";
    for (const auto& layer : layers_) {
        std::cout << "  Z=" << layer.z_order 
                  << ", Visible=" << (layer.visible ? "Yes" : "No")
                  << ", Opacity=" << layer.opacity
                  << ", Name=" << layer.name << "\n";
    }
}

void LayerStack::update_layer_index() {
    layer_index_by_name_.clear();
    for (size_t i = 0; i < layers_.size(); ++i) {
        layer_index_by_name_[layers_[i].name] = i;
    }
}

size_t LayerStack::find_layer_by_name(const std::string& name) const {
    auto it = layer_index_by_name_.find(name);
    return (it != layer_index_by_name_.end()) ? it->second : layers_.size();
}

void LayerStack::validate_layer_integrity() const {
    // Basic integrity check
    if (layers_.size() != layer_index_by_name_.size()) {
        std::cerr << "Warning: Layer count mismatch in LayerStack\n";
    }
    
    // Check for duplicate names
    std::set<std::string> names;
    for (const auto& layer : layers_) {
        if (names.count(layer.name)) {
            std::cerr << "Warning: Duplicate layer name: " << layer.name << "\n";
        }
        names.insert(layer.name);
    }
}

// LayerRenderer implementation
void LayerRenderer::set_layer_filter(const std::vector<int>& allowed_z_orders) {
    allowed_z_orders_ = allowed_z_orders;
    use_layer_filter_ = !allowed_z_orders.empty();
}

void LayerRenderer::clear_layer_filter() {
    allowed_z_orders_.clear();
    use_layer_filter_ = false;
}

Tensor LayerRenderer::render_layer(const Layer& layer, int width, int height, const Camera2D& camera) const {
    if (!layer.visible || !layer.node) {
        return zeros({height, width, 4});
    }
    
    // Apply layer filter if enabled
    if (use_layer_filter_) {
        bool allowed = false;
        for (int allowed_z : allowed_z_orders_) {
            if (layer.z_order == allowed_z) {
                allowed = true;
                break;
            }
        }
        if (!allowed) {
            return zeros({height, width, 4});
        }
    }
    
    // Render the layer with camera transform
    Tensor rendered = layer.node->render(width, height, camera);
    
    // Apply layer opacity if not 1.0
    if (layer.opacity != 1.0f && rendered.size() >= 4) {
        // Apply opacity to alpha channel (simplified)
        size_t pixel_count = rendered.size() / 4;
        float* data = rendered.data_ptr<float>();
        for (size_t i = 0; i < pixel_count; ++i) {
            data[i * 4 + 3] *= layer.opacity;
        }
    }
    
    return rendered;
}

// ParallaxCamera2D implementation
void ParallaxCamera2D::set_parallax_factor(int z_order, float factor) {
    parallax_factors_[z_order] = std::clamp(factor, 0.0f, 1.0f);
}

float ParallaxCamera2D::get_parallax_factor(int z_order) const {
    auto it = parallax_factors_.find(z_order);
    return (it != parallax_factors_.end()) ? it->second : 1.0f;
}

Tensor ParallaxCamera2D::render_with_parallax(const LayerStack& layer_stack, int width, int height, const Camera2D& base_camera) const {
    // Render background with parallax effect
    Tensor final_frame = zeros({height, width, 4});
    
    auto render_order = layer_stack.get_render_order();
    
    for (const Layer* layer : render_order) {
        if (!layer->visible || !layer->node) continue;
        
        // Calculate parallax offset
        float parallax_factor = get_parallax_factor(layer->z_order);
        Tensor parallax_offset = base_camera.position * (1.0f - parallax_factor);
        
        // Create a temporary camera with parallax offset
        Camera2D parallax_camera = base_camera;
        parallax_camera.position = base_camera.position - parallax_offset;
        
        // Render layer with parallax camera
        Tensor layer_render = layer->node->render(width, height, parallax_camera);
        
        // Composite layer onto final frame
        final_frame = final_frame + layer_render; // Simple additive composite
    }
    
    return final_frame;
}

} // namespace scene
} // namespace jin