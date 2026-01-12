#include "test_runner.cpp"
#include "test_allocator.cpp"
#include "test_tensor.cpp"
#include "test_ops.cpp"
#include "test_autodiff.cpp"
#include "test_utils.cpp"

int main(int argc, char** argv) {
    std::string filter = "";
    
    if (argc > 1) {
        filter = argv[1];
    }
    
    int failed = jin::test::TestRunner::instance().run_all(filter);
    
    return failed;
}
