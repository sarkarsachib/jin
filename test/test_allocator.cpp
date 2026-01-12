#include "test_runner.cpp"
#include "../src/core/allocator.h"

using namespace jin;
using namespace jin::test;

TEST(allocator_basic) {
    DefaultAllocator alloc;
    void* ptr = alloc.allocate(1024);
    ASSERT_TRUE(ptr != nullptr);
    ASSERT_EQ(alloc.allocated_bytes(), 1024);
    alloc.deallocate(ptr);
}

TEST(arena_allocator_basic) {
    ArenaAllocator alloc(4096);
    void* ptr1 = alloc.allocate(512);
    void* ptr2 = alloc.allocate(512);
    ASSERT_TRUE(ptr1 != nullptr);
    ASSERT_TRUE(ptr2 != nullptr);
    ASSERT_TRUE(ptr1 != ptr2);
}

TEST(arena_allocator_alignment) {
    ArenaAllocator alloc(4096);
    void* ptr = alloc.allocate(100, 64);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    ASSERT_EQ(addr % 64, 0);
}

TEST(pool_allocator_basic) {
    PoolAllocator alloc(64, 10);
    void* ptr1 = alloc.allocate(64);
    void* ptr2 = alloc.allocate(64);
    ASSERT_TRUE(ptr1 != nullptr);
    ASSERT_TRUE(ptr2 != nullptr);
    alloc.deallocate(ptr1);
    alloc.deallocate(ptr2);
}

TEST(pool_allocator_reuse) {
    PoolAllocator alloc(64, 10);
    void* ptr1 = alloc.allocate(64);
    alloc.deallocate(ptr1);
    void* ptr2 = alloc.allocate(64);
    ASSERT_EQ(ptr1, ptr2);
}

TEST(allocator_stress_test) {
    DefaultAllocator alloc;
    std::vector<void*> ptrs;
    
    for (int i = 0; i < 1000; ++i) {
        ptrs.push_back(alloc.allocate(128));
    }
    
    for (void* ptr : ptrs) {
        alloc.deallocate(ptr);
    }
    
    ASSERT_EQ(alloc.allocated_bytes(), 0);
}
