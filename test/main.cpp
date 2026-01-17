#include "test_runner.cpp"
#include "test_allocator.cpp"
#include "test_tensor.cpp"
#include "test_ops.cpp"
#include "test_autodiff.cpp"
#include "test_utils.cpp"
#include "test_nn.cpp"
#include "test_diffusion.cpp"

// Temporal Layer 4 tests
#include "test_frame_state.cpp"
#include "test_latent_lock.cpp"
#include "test_drift_suppressor.cpp"
#include "test_coherence_graph.cpp"
#include "test_optical_flow.cpp"
#include "test_temporal_attention.cpp"
#include "test_motion_encoder.cpp"
#include "test_scene_memory.cpp"
#include "test_consistency_loss.cpp"
#include "test_long_sequence.cpp"

int main(int argc, char** argv) {
    std::string filter = "";
    
    if (argc > 1) {
        filter = argv[1];
    }
    
    int failed = jin::test::TestRunner::instance().run_all(filter);
    
    return failed;
}
