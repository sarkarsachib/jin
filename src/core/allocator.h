#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace jin {

constexpr size_t CACHE_LINE_SIZE = 64;

class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* allocate(size_t size, size_t alignment = CACHE_LINE_SIZE) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual size_t allocated_bytes() const = 0;
};

class ArenaAllocator : public Allocator {
public:
    explicit ArenaAllocator(size_t block_size = 1024 * 1024);
    ~ArenaAllocator() override;
    
    void* allocate(size_t size, size_t alignment = CACHE_LINE_SIZE) override;
    void deallocate(void* ptr) override;
    size_t allocated_bytes() const override { return total_allocated_; }
    
    void reset();
    
private:
    struct Block {
        uint8_t* data;
        size_t size;
        size_t used;
        
        Block(size_t sz);
        ~Block();
    };
    
    size_t block_size_;
    std::vector<std::unique_ptr<Block>> blocks_;
    size_t total_allocated_;
    mutable std::mutex mutex_;
    
    Block* get_or_create_block(size_t required_size);
};

class PoolAllocator : public Allocator {
public:
    explicit PoolAllocator(size_t object_size, size_t objects_per_block = 1024);
    ~PoolAllocator() override;
    
    void* allocate(size_t size, size_t alignment = CACHE_LINE_SIZE) override;
    void deallocate(void* ptr) override;
    size_t allocated_bytes() const override { return total_allocated_; }
    
private:
    struct Block {
        uint8_t* data;
        size_t capacity;
        std::vector<void*> free_list;
        
        Block(size_t object_size, size_t count);
        ~Block();
    };
    
    size_t object_size_;
    size_t objects_per_block_;
    std::vector<std::unique_ptr<Block>> blocks_;
    std::vector<void*> free_list_;
    size_t total_allocated_;
    mutable std::mutex mutex_;
    
    Block* create_block();
};

class DefaultAllocator : public Allocator {
public:
    DefaultAllocator() = default;
    ~DefaultAllocator() override = default;
    
    void* allocate(size_t size, size_t alignment = CACHE_LINE_SIZE) override;
    void deallocate(void* ptr) override;
    size_t allocated_bytes() const override { return total_allocated_; }
    
private:
    std::unordered_map<void*, size_t> allocations_;
    size_t total_allocated_ = 0;
    mutable std::mutex mutex_;
};

Allocator* get_default_allocator();

size_t align_up(size_t size, size_t alignment);

} // namespace jin
