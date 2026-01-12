#include "allocator.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace jin {

size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// ArenaAllocator::Block implementation
ArenaAllocator::Block::Block(size_t sz) : size(sz), used(0) {
    data = static_cast<uint8_t*>(std::aligned_alloc(CACHE_LINE_SIZE, align_up(sz, CACHE_LINE_SIZE)));
    if (!data) {
        throw std::bad_alloc();
    }
}

ArenaAllocator::Block::~Block() {
    if (data) {
        std::free(data);
    }
}

// ArenaAllocator implementation
ArenaAllocator::ArenaAllocator(size_t block_size) 
    : block_size_(block_size), total_allocated_(0) {
}

ArenaAllocator::~ArenaAllocator() = default;

ArenaAllocator::Block* ArenaAllocator::get_or_create_block(size_t required_size) {
    // Try to find a block with enough space
    for (auto& block : blocks_) {
        if (block->size - block->used >= required_size) {
            return block.get();
        }
    }
    
    // Create a new block
    size_t new_block_size = std::max(block_size_, required_size);
    blocks_.push_back(std::make_unique<Block>(new_block_size));
    return blocks_.back().get();
}

void* ArenaAllocator::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t aligned_size = align_up(size, alignment);
    Block* block = get_or_create_block(aligned_size);
    
    // Align the current position
    size_t aligned_used = align_up(block->used, alignment);
    if (aligned_used + aligned_size > block->size) {
        // Need a new block
        size_t new_block_size = std::max(block_size_, aligned_size);
        blocks_.push_back(std::make_unique<Block>(new_block_size));
        block = blocks_.back().get();
        aligned_used = 0;
    }
    
    void* ptr = block->data + aligned_used;
    block->used = aligned_used + aligned_size;
    total_allocated_ += aligned_size;
    
    return ptr;
}

void ArenaAllocator::deallocate(void* ptr) {
    // Arena allocator doesn't deallocate individual pointers
    // Memory is freed when reset() is called or destructor runs
}

void ArenaAllocator::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : blocks_) {
        block->used = 0;
    }
    total_allocated_ = 0;
}

// PoolAllocator::Block implementation
PoolAllocator::Block::Block(size_t object_size, size_t count) : capacity(count) {
    size_t total_size = align_up(object_size * count, CACHE_LINE_SIZE);
    data = static_cast<uint8_t*>(std::aligned_alloc(CACHE_LINE_SIZE, total_size));
    if (!data) {
        throw std::bad_alloc();
    }
    
    // Initialize free list for this block
    free_list.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        free_list.push_back(data + i * object_size);
    }
}

PoolAllocator::Block::~Block() {
    if (data) {
        std::free(data);
    }
}

// PoolAllocator implementation
PoolAllocator::PoolAllocator(size_t object_size, size_t objects_per_block)
    : object_size_(align_up(object_size, CACHE_LINE_SIZE)),
      objects_per_block_(objects_per_block),
      total_allocated_(0) {
}

PoolAllocator::~PoolAllocator() = default;

PoolAllocator::Block* PoolAllocator::create_block() {
    blocks_.push_back(std::make_unique<Block>(object_size_, objects_per_block_));
    Block* block = blocks_.back().get();
    
    // Add all objects to the free list
    free_list_.insert(free_list_.end(), block->free_list.begin(), block->free_list.end());
    
    return block;
}

void* PoolAllocator::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (size > object_size_) {
        throw std::bad_alloc();
    }
    
    if (free_list_.empty()) {
        create_block();
    }
    
    void* ptr = free_list_.back();
    free_list_.pop_back();
    total_allocated_ += object_size_;
    
    return ptr;
}

void PoolAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    free_list_.push_back(ptr);
    total_allocated_ -= object_size_;
}

// DefaultAllocator implementation
void* DefaultAllocator::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t aligned_size = align_up(size, alignment);
    void* ptr = std::aligned_alloc(alignment, aligned_size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    allocations_[ptr] = aligned_size;
    total_allocated_ += aligned_size;
    
    return ptr;
}

void DefaultAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        total_allocated_ -= it->second;
        allocations_.erase(it);
        std::free(ptr);
    }
}

Allocator* get_default_allocator() {
    static DefaultAllocator allocator;
    return &allocator;
}

} // namespace jin
