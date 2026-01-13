#pragma once

#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <cmath>
#include <sstream>

namespace jin {
namespace test {

struct TestCase {
    std::string name;
    std::function<void()> func;
    bool passed;
    std::string error_msg;
};

class TestRunner {
public:
    static TestRunner& instance() {
        static TestRunner runner;
        return runner;
    }
    
    void register_test(const std::string& name, std::function<void()> func) {
        tests_.push_back({name, func, false, ""});
    }
    
    int run_all(const std::string& filter = "") {
        int passed = 0;
        int failed = 0;
        
        std::cout << "\n=== Running Jin Core Tests ===\n\n";
        
        for (auto& test : tests_) {
            if (!filter.empty() && test.name.find(filter) == std::string::npos) {
                continue;
            }
            
            std::cout << "[ RUN      ] " << test.name << std::endl;
            
            try {
                test.func();
                test.passed = true;
                std::cout << "\033[32m[       OK ]\033[0m " << test.name << std::endl;
                passed++;
            } catch (const std::exception& e) {
                test.passed = false;
                test.error_msg = e.what();
                std::cout << "\033[31m[  FAILED  ]\033[0m " << test.name << std::endl;
                std::cout << "  Error: " << e.what() << std::endl;
                failed++;
            }
        }
        
        std::cout << "\n=== Test Summary ===\n";
        std::cout << "\033[32mPassed: " << passed << "\033[0m\n";
        if (failed > 0) {
            std::cout << "\033[31mFailed: " << failed << "\033[0m\n";
        }
        std::cout << "Total:  " << (passed + failed) << "\n\n";
        
        return failed;
    }
    
    size_t test_count() const { return tests_.size(); }
    
private:
    TestRunner() = default;
    std::vector<TestCase> tests_;
};

class TestRegistrar {
public:
    TestRegistrar(const std::string& name, std::function<void()> func) {
        TestRunner::instance().register_test(name, func);
    }
};

#define TEST(name) \
    void test_##name(); \
    static jin::test::TestRegistrar registrar_##name(#name, test_##name); \
    void test_##name()

#define ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #expr << " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #a << " == " << #b \
                << " (values: " << (a) << " != " << (b) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define ASSERT_NEAR(a, b, epsilon) \
    do { \
        auto diff = std::abs((a) - (b)); \
        if (diff > (epsilon)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #a << " ≈ " << #b \
                << " (values: " << (a) << " vs " << (b) << ", diff: " << diff << " > " << (epsilon) << ") at " \
                << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define ASSERT_NE(a, b) \
    do { \
        if ((a) == (b)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #a << " != " << #b \
                << " (values: " << (a) << " == " << (b) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define ASSERT_GE(a, b) \
    do { \
        if ((a) < (b)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #a << " >= " << #b \
                << " (values: " << (a) << " < " << (b) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define ASSERT_GT(a, b) \
    do { \
        if ((a) <= (b)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #a << " > " << #b \
                << " (values: " << (a) << " <= " << (b) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define ASSERT_LT(a, b) \
    do { \
        if ((a) >= (b)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #a << " < " << #b \
                << " (values: " << (a) << " >= " << (b) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

#define ASSERT_LE(a, b) \
    do { \
        if ((a) > (b)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " << #a << " <= " << #b \
                << " (values: " << (a) << " > " << (b) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while(0)

} // namespace test
} // namespace jin
