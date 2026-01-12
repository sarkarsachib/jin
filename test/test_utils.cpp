#include "test_runner.cpp"
#include "../src/utils/shapes.h"
#include "../src/utils/logger.h"

using namespace jin;
using namespace jin::test;

TEST(shapes_compute_size) {
    Shape shape = {2, 3, 4};
    ASSERT_EQ(compute_size(shape), 24);
}

TEST(shapes_compute_strides) {
    Shape shape = {2, 3, 4};
    Shape strides = compute_strides(shape);
    ASSERT_EQ(strides[0], 12);
    ASSERT_EQ(strides[1], 4);
    ASSERT_EQ(strides[2], 1);
}

TEST(shapes_can_broadcast) {
    ASSERT_TRUE(can_broadcast({2, 3}, {2, 3}));
    ASSERT_TRUE(can_broadcast({1, 3}, {2, 3}));
    ASSERT_TRUE(can_broadcast({2, 1}, {2, 3}));
    ASSERT_FALSE(can_broadcast({2, 3}, {3, 2}));
}

TEST(shapes_broadcast_shapes) {
    Shape a = {1, 3};
    Shape b = {2, 3};
    Shape result = broadcast_shapes(a, b);
    ASSERT_EQ(result[0], 2);
    ASSERT_EQ(result[1], 3);
}

TEST(shapes_transpose) {
    Shape shape = {2, 3, 4};
    Shape transposed = infer_shape_after_transpose(shape, {2, 1, 0});
    ASSERT_EQ(transposed[0], 4);
    ASSERT_EQ(transposed[1], 3);
    ASSERT_EQ(transposed[2], 2);
}

TEST(shapes_sum) {
    Shape shape = {2, 3, 4};
    Shape result = infer_shape_after_sum(shape, 1, false);
    ASSERT_EQ(result.size(), 2);
    ASSERT_EQ(result[0], 2);
    ASSERT_EQ(result[1], 4);
}

TEST(shapes_sum_keepdims) {
    Shape shape = {2, 3, 4};
    Shape result = infer_shape_after_sum(shape, 1, true);
    ASSERT_EQ(result.size(), 3);
    ASSERT_EQ(result[0], 2);
    ASSERT_EQ(result[1], 1);
    ASSERT_EQ(result[2], 4);
}

TEST(logger_format_bytes) {
    ASSERT_EQ(format_bytes(512), "512.00 B");
    ASSERT_EQ(format_bytes(1024), "1.00 KB");
    ASSERT_EQ(format_bytes(1024 * 1024), "1.00 MB");
}
