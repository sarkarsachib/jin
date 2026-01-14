#pragma once

#include "../core/tensor.h"
#include "../temporal/frame_state.h"
#include "scene_graph.h"
#include <vector>
#include <memory>
#include <map>
#include <algorithm>

namespace jin {
namespace scene {

/**
 * Layer - Represents a single renderable layer with z-order
 */
struct Layer {
    int z_order;                                   // Z-order for sorting (higher = in front)
    std::shared_ptr<SceneNode> node;                // Scene node to render
    bool visible = true;                            // Layer visibility
    float opacity = 1.0f;                           // Layer opacity (0.0-1.0)
    std::string name;                               // Layer name for identification
    
    // Comparison operator for sorting
    bool operator<(const Layer& other) const {
        return z_order < other.z_order;
    }
    
    bool operator==(const Layer& other) const {
        return z_order == other.z_order && node == other.node;
    }
};

/**
 * LayerStack - Manages depth ordering (z-order) for 2D rendering
 */
class LayerStack {
public:
    LayerStack();
    ~LayerStack() = default;
    
    // Layer management
    void add_layer(int z_order, std::shared_ptr<SceneNode> node, const std::string& name = "");
    void remove_layer(const std::string& name);
    void remove_layer(std::shared_ptr<SceneNode> node);
    void clear_layers();
    
    // Layer properties
    void set_layer_visibility(const std::string& name, bool visible);
    void set_layer_opacity(const std::string& name, float opacity);
    void set_layer_z_order(const std::string& name, int z_order);
    
    // Query layers
    std::shared_ptr<SceneNode> get_layer_node(const std::string& name) const;
    Layer* get_layer(const std::string& name);
    const Layer* get_layer(const std::string& name) const;
    
    // Sorting and order management
    void sort_layers();
    std::vector<Layer*> get_render_order();         // Returns layers sorted by z-order
    std::vector<const Layer*> get_render_order() const;
    
    // Batch operations
    std::vector<std::shared_ptr<SceneNode>> get_visible_nodes() const;
    std::vector<Layer*> get_layers_in_range(int min_z, int max_z);
    
    // Layer range queries
    std::vector<Layer*> get_layers_above(int z_order);
    std::vector<Layer*> get_layers_below(int z_order);
    
    // Frame state integration
    void set_frame_state(const temporal::FrameState& frame_state);
    temporal::FrameState get_frame_state() const;
    
    // Debug and inspection
    void print_layers() const;
    int get_layer_count() const { return static_cast<int>(layers_.size()); }
    
private:
    std::vector<Layer> layers_;
    std::map<std::string, size_t> layer_index_by_name_;  // For fast lookup by name
    temporal::FrameState frame_state_;
    
    void update_layer_index();
    size_t find_layer_by_name(const std::string& name) const;
    void validate_layer_integrity() const;
};

/**
 * LayerRenderer - Extended camera functionality for layer support
 */
class LayerRenderer {
public:
    // Layer filtering
    void set_layer_filter(const std::vector<int>& allowed_z_orders);
    void clear_layer_filter();
    
    // Culling parameters
    float cull_margin = 0.0f;  // Extra margin around viewport for culling
    
    // Render layer at specific z-order
    Tensor render_layer(const Layer& layer, int width, int height, const Camera2D& camera) const;
    
private:
    std::vector<int> allowed_z_orders_;  // Empty = render all layers
    bool use_layer_filter_ = false;
};

/**
 * ParallaxCamera2D - Camera with parallax layer support for backgrounds
 */
class ParallaxCamera2D : public Camera2D {
public:
    // Parallax factor (0.0 = fixed to screen, 1.0 = moves with camera)
    void set_parallax_factor(int z_order, float factor);
    float get_parallax_factor(int z_order) const;
    
    // Parallax-specific rendering
    Tensor render_with_parallax(const LayerStack& layer_stack, int width, int height, const Camera2D& base_camera) const;
    
private:
    std::map<int, float> parallax_factors_;  // z_order -> parallax_factor
};

} // namespace scene
} // namespace jin