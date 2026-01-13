#pragma once

#include "frame_state.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <cmath>

namespace jin {
namespace temporal {

class CoherenceGraph {
public:
    struct Node {
        int frame_idx;
        FrameState state;
        std::vector<int> connected_frames;  // Frames with consistency edges
        float coherence_weight;              // Edge weight for coherence
        
        Node(int idx, const FrameState& s) : frame_idx(idx), state(s), coherence_weight(1.0f) {
            connected_frames.clear();
        }
    };
    
    CoherenceGraph() : max_frames_(1000), coherence_threshold_(0.8f) {}
    
    // Build graph: add edges between consecutive frames
    void add_frame(int idx, const FrameState& state) {
        Node new_node(idx, state);
        
        // Link to previous frame if exists
        if (!nodes_.empty()) {
            int prev_idx = nodes_.back().frame_idx;
            float coherence = compute_coherence(nodes_.back().state, state);
            
            // Add bidirectional edges
            new_node.connected_frames.push_back(prev_idx);
            nodes_.back().connected_frames.push_back(idx);
            
            // Set coherence weight
            new_node.coherence_weight = coherence;
            nodes_.back().coherence_weight = coherence;
        }
        
        nodes_.push_back(new_node);
        
        // Keep only last max_frames_ to prevent memory blowup
        if (nodes_.size() > static_cast<size_t>(max_frames_)) {
            nodes_.erase(nodes_.begin());
        }
    }
    
    // Find node by frame index
    Node* find_node(int frame_idx) {
        for (auto& node : nodes_) {
            if (node.frame_idx == frame_idx) {
                return &node;
            }
        }
        return nullptr;
    }
    
    const Node* find_node(int frame_idx) const {
        for (const auto& node : nodes_) {
            if (node.frame_idx == frame_idx) {
                return &node;
            }
        }
        return nullptr;
    }
    
    // Query coherence between any two frames
    float coherence(int frame_a, int frame_b) const {
        if (frame_a == frame_b) return 1.0f;
        
        const Node* node_a = find_node(frame_a);
        const Node* node_b = find_node(frame_b);
        
        if (!node_a || !node_b) return 0.0f;
        
        // Direct connection
        auto it_a = std::find(node_a->connected_frames.begin(), 
                             node_a->connected_frames.end(), frame_b);
        if (it_a != node_a->connected_frames.end()) {
            return compute_coherence(node_a->state, node_b->state);
        }
        
        // Multi-hop path through graph
        return coherence_through_path(frame_a, frame_b);
    }
    
    // Identify discontinuities
    std::vector<int> find_discontinuities(float threshold) const {
        std::vector<int> discontinuities;
        
        if (nodes_.size() < 2) return discontinuities;
        
        for (size_t i = 1; i < nodes_.size(); ++i) {
            float prev_coherence = compute_coherence(nodes_[i-1].state, nodes_[i].state);
            
            if (prev_coherence < threshold) {
                discontinuities.push_back(nodes_[i].frame_idx);
            }
        }
        
        return discontinuities;
    }
    
    // Find longest coherent subsequence
    std::vector<int> longest_coherent_subsequence(float min_coherence) const {
        if (nodes_.empty()) return {};
        
        std::vector<int> best_sequence;
        std::vector<int> current_sequence;
        
        for (size_t i = 0; i < nodes_.size(); ++i) {
            current_sequence.clear();
            current_sequence.push_back(nodes_[i].frame_idx);
            
            for (size_t j = i + 1; j < nodes_.size(); ++j) {
                float coh = coherence(nodes_[i].frame_idx, nodes_[j].frame_idx);
                if (coh >= min_coherence) {
                    current_sequence.push_back(nodes_[j].frame_idx);
                } else {
                    break;  // Break on first discontinuity
                }
            }
            
            if (current_sequence.size() > best_sequence.size()) {
                best_sequence = current_sequence;
            }
        }
        
        return best_sequence;
    }
    
    // Get recent frames (last K frames)
    std::vector<Node> get_recent_frames(int k) const {
        if (nodes_.empty() || k <= 0) return {};
        
        int start = std::max(0, static_cast<int>(nodes_.size()) - k);
        return std::vector<Node>(nodes_.begin() + start, nodes_.end());
    }
    
    // Get all frames in sequence
    const std::vector<Node>& get_all_frames() const { return nodes_; }
    
    // Clear the graph
    void clear() {
        nodes_.clear();
    }
    
    // Configuration
    void set_max_frames(int max_frames) {
        max_frames_ = std::max(1, max_frames);
    }
    
    void set_coherence_threshold(float threshold) {
        coherence_threshold_ = std::clamp(threshold, 0.0f, 1.0f);
    }
    
    size_t size() const { return nodes_.size(); }
    bool empty() const { return nodes_.empty(); }

private:
    std::vector<Node> nodes_;
    int max_frames_;
    float coherence_threshold_;
    
    // Compute coherence between two frame states
    float compute_coherence(const FrameState& a, const FrameState& b) const {
        float coherence = 0.0f;
        int components = 0;
        
        // Character latent coherence
        if (a.character_latent.size() > 0 && b.character_latent.size() > 0) {
            Tensor diff = a.character_latent - b.character_latent;
            float char_coherence = 1.0f / (1.0f + std::sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)).item()));
            coherence += char_coherence;
            components++;
        }
        
        // Scene latent coherence
        if (a.scene_latent.size() > 0 && b.scene_latent.size() > 0) {
            Tensor diff = a.scene_latent - b.scene_latent;
            float scene_coherence = 1.0f / (1.0f + std::sqrt(jin::ops::sum(jin::ops::pow(diff, 2.0f)).item()));
            coherence += scene_coherence;
            components++;
        }
        
        // Motion smoothness (already computed in FrameState)
        coherence += std::min(a.motion_smoothness, b.motion_smoothness);
        components++;
        
        // Scene stability
        coherence += std::min(a.scene_stability, b.scene_stability);
        components++;
        
        return components > 0 ? coherence / components : 1.0f;
    }
    
    // Find coherence through graph traversal (BFS)
    float coherence_through_path(int frame_a, int frame_b) const {
        std::unordered_map<int, float> distances;
        std::queue<int> queue;
        
        distances[frame_a] = 1.0f;
        queue.push(frame_a);
        
        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            
            if (current == frame_b) {
                return distances[current];
            }
            
            const Node* node = find_node(current);
            if (!node) continue;
            
            for (int neighbor : node->connected_frames) {
                if (distances.find(neighbor) == distances.end()) {
                    const Node* neighbor_node = find_node(neighbor);
                    if (neighbor_node) {
                        float edge_weight = compute_coherence(node->state, neighbor_node->state);
                        distances[neighbor] = distances[current] * edge_weight;
                        queue.push(neighbor);
                    }
                }
            }
        }
        
        return 0.0f;  // No path found
    }
};

} // namespace temporal
} // namespace jin