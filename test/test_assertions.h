#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>

namespace jin {
namespace test {

// Simple test assertion macros
#define ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            throw std::runtime_error("Assertion failed: " #expr); \
        } \
    } while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            throw std::runtime_error("Assertion failed: " #a " == " #b); \
        } \
    } while(0)

#define ASSERT_NEAR(a, b, epsilon) \
    do { \
        if (std::abs((a) - (b)) > (epsilon)) { \
            throw std::runtime_error("Assertion failed: " #a " ≈ " #b); \
        } \
    } while(0)

} // namespace test
} // namespace jin