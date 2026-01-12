#pragma once

#include "graph.h"
#include <vector>
#include <memory>

namespace jin {
namespace autodiff {

class GradientTape {
public:
    static GradientTape& instance();
    
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool is_enabled() const { return enabled_; }
    
    void record(const std::vector<Tensor>& inputs, const Tensor& output,
                GraphNode::BackwardFunc backward_fn, const std::string& op_name);
    
    void clear();
    
    const std::vector<std::shared_ptr<GraphNode>>& nodes() const { return nodes_; }
    
private:
    GradientTape() : enabled_(false) {}
    
    bool enabled_;
    std::vector<std::shared_ptr<GraphNode>> nodes_;
};

class TapeGuard {
public:
    TapeGuard() { GradientTape::instance().enable(); }
    ~TapeGuard() { GradientTape::instance().disable(); }
};

} // namespace autodiff
} // namespace jin
