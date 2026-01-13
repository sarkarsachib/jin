#include "cuda_runtime.h"
#include "../../gpu/gpu_common.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace jin {
namespace gpu {
namespace cuda {

// Static member initialization
#ifdef __CUDACC__
bool CudaRuntime::initialized_ = false;
int CudaRuntime::init_count_ = 0;
#endif

// Global CUDA runtime instance
std::unique_ptr<CudaRuntime>& get_global_cuda_runtime() {
    static std::unique_ptr<CudaRuntime> instance;
    return instance;
}

#ifdef __CUDACC__
// CudaMemoryPool implementation
CudaMemoryPool::CudaMemoryPool(int device_id) : device_id_(device_id), total_pool_size_(0) {
    cudaSetDevice(device_id_);
}

CudaMemoryPool::~CudaMemoryPool() {
    for (auto& alloc : allocations_) {
        if (!alloc.is_free) {
            if (alloc.type == GpuMemoryType::DEVICE) {
                cudaFree(alloc.ptr);
            } else if (alloc.type == GpuMemoryType::UNIFIED) {
                cudaFree(alloc.ptr);
            } else if (alloc.type == GpuMemoryType::PINNED) {
                cudaFreeHost(alloc.ptr);
            }
        }
    }
}

void* CudaMemoryPool::allocate(size_t size, GpuMemoryType type) {
    cudaSetDevice(device_id_);
    
    // Try to find a free block of sufficient size
    for (auto& alloc : allocations_) {
        if (alloc.is_free && alloc.size >= size) {
            alloc.is_free = false;
            total_allocated_ += size;
            peak_allocated_ = std::max(peak_allocated_, total_allocated_);
            return alloc.ptr;
        }
    }
    
    // No suitable block found, allocate new
    void* ptr = nullptr;
    cudaError_t error;
    
    switch (type) {
        case GpuMemoryType::DEVICE:
            error = cudaMalloc(&ptr, size);
            break;
        case GpuMemoryType::UNIFIED:
            error = cudaMallocManaged(&ptr, size);
            break;
        case GpuMemoryType::PINNED:
            error = cudaMallocHost(&ptr, size);
            break;
        default:
            error = cudaMalloc(&ptr, size);
    }
    
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA memory allocation failed: " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
    
    allocations_.push_back({ptr, size, type, false});
    total_allocated_ += size;
    peak_allocated_ = std::max(peak_allocated_, total_allocated_);
    
    return ptr;
}

void CudaMemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    
    cudaSetDevice(device_id_);
    
    for (auto& alloc : allocations_) {
        if (alloc.ptr == ptr && !alloc.is_free) {
            alloc.is_free = true;
            total_allocated_ -= alloc.size;
            
            // Free small allocations to prevent memory leaks
            if (alloc.size < 1024 * 1024) { // Less than 1MB
                if (alloc.type == GpuMemoryType::DEVICE || alloc.type == GpuMemoryType::UNIFIED) {
                    cudaFree(alloc.ptr);
                } else if (alloc.type == GpuMemoryType::PINNED) {
                    cudaFreeHost(alloc.ptr);
                }
                allocations_.erase(std::remove(allocations_.begin(), allocations_.end(), alloc));
            }
            break;
        }
    }
}

void CudaMemoryPool::reset() {
    cudaSetDevice(device_id_);
    
    for (auto& alloc : allocations_) {
        if (!alloc.is_free) {
            if (alloc.type == GpuMemoryType::DEVICE || alloc.type == GpuMemoryType::UNIFIED) {
                cudaFree(alloc.ptr);
            } else if (alloc.type == GpuMemoryType::PINNED) {
                cudaFreeHost(alloc.ptr);
            }
            alloc.is_free = true;
        }
    }
    
    total_allocated_ = 0;
    allocations_.clear();
}

// CudaStream implementation
CudaStream::CudaStream(int device_id) : device_id_(device_id) {
    cudaSetDevice(device_id_);
    cudaStreamCreate(&stream_);
}

CudaStream::~CudaStream() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CudaStream::synchronize() {
    cudaSetDevice(device_id_);
    cudaStreamSynchronize(stream_);
}

bool CudaStream::is_synchronized() const {
    cudaSetDevice(device_id_);
    cudaError_t error = cudaStreamQuery(stream_);
    return error == cudaSuccess;
}

void CudaStream::wait_event(void* cuda_event) {
    cudaSetDevice(device_id_);
    cudaStreamWaitEvent(stream_, static_cast<cudaEvent_t>(cuda_event), 0);
}

// CudaEvent implementation
CudaEvent::CudaEvent(int device_id, bool uses_timing) : device_id_(device_id) {
    cudaSetDevice(device_id_);
    cudaEventCreateWithFlags(&event_, uses_timing ? cudaEventDefault : cudaEventDisableTiming);
}

CudaEvent::~CudaEvent() {
    if (event_) {
        cudaEventDestroy(event_);
    }
}

void CudaEvent::record() {
    cudaSetDevice(device_id_);
    cudaEventRecord(event_);
}

void CudaEvent::wait() {
    cudaSetDevice(device_id_);
    cudaEventSynchronize(event_);
}

// CudaKernel implementation
CudaKernel::CudaKernel(cudaFunction_t function, const std::string& name) 
    : function_(function), name_(name) {}

void CudaKernel::launch(GpuStream& stream, const Shape& grid, const Shape& block, 
                       GpuKernelParams& params) {
    CudaStream& cuda_stream = static_cast<CudaStream&>(stream);
    cudaSetDevice(0); // Assume device 0 for now
    
    // Calculate total threads and check bounds
    size_t total_threads = 1;
    for (size_t dim : grid) total_threads *= dim;
    
    // Launch kernel
    cudaError_t error = cudaLaunchKernel(
        function_,
        static_cast<unsigned int>(grid.x),
        static_cast<unsigned int>(grid.y), 
        static_cast<unsigned int>(grid.z),
        const_cast<void**>(params.arguments),
        nullptr,
        cuda_stream.get_cuda_stream()
    );
    
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA kernel launch failed: " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}

// CudaRuntime implementation
CudaRuntime::CudaRuntime() : device_count_(0), current_device_(0) {
    initialize_cuda();
}

CudaRuntime::~CudaRuntime() {
    cleanup_cuda();
}

void CudaRuntime::initialize_cuda() {
    if (!initialized_) {
        cudaError_t error = cudaGetDeviceCount(&device_count_);
        if (error != cudaSuccess && error != cudaErrorNoDevice) {
            std::ostringstream oss;
            oss << "CUDA initialization failed: " << cudaGetErrorString(error);
            throw std::runtime_error(oss.str());
        }
        initialized_ = true;
    }
    init_count_++;
    
    device_count_ = 0;
    cudaGetDeviceCount(&device_count_);
    
    if (device_count_ > 0) {
        current_device_ = 0;
        set_device(0);
    }
}

void CudaRuntime::cleanup_cuda() {
    init_count_--;
    if (init_count_ <= 0 && initialized_) {
        // Clean up memory pools
        memory_pools_.clear();
        initialized_ = false;
    }
}

int CudaRuntime::get_device_count() {
    return device_count_;
}

std::string CudaRuntime::get_device_name(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        return "Invalid device";
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return std::string(prop.name);
}

GpuCapabilities CudaRuntime::get_device_capabilities(int device_id) {
    if (device_id < 0 || device_id >= device_count_) {
        return GpuCapabilities{};
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    GpuCapabilities caps;
    caps.compute_capability_supported = true;
    caps.max_threads_per_block = prop.maxThreadsPerBlock;
    caps.max_shared_memory_per_block = prop.sharedMemPerBlock;
    caps.warp_size = prop.warpSize;
    caps.compute_capability = static_cast<float>(prop.major) + static_cast<float>(prop.minor) / 10.0f;
    
    // Memory info
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    caps.total_memory = total_mem;
    caps.available_memory = free_mem;
    
    // Max blocks per dimension
    caps.max_blocks_per_dimension = prop.maxGridSize[0];
    
    return caps;
}

void CudaRuntime::set_device(int device_id) {
    if (device_id >= 0 && device_id < device_count_) {
        cudaSetDevice(device_id);
        current_device_ = device_id;
    }
}

int CudaRuntime::get_current_device() {
    cudaGetDevice(&current_device_);
    return current_device_;
}

void* CudaRuntime::allocate_device(size_t size) {
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    check_cuda_error(error, "Device memory allocation");
    return ptr;
}

void CudaRuntime::deallocate_device(void* ptr) {
    if (ptr) {
        cudaError_t error = cudaFree(ptr);
        check_cuda_error(error, "Device memory deallocation");
    }
}

void* CudaRuntime::allocate_pinned(size_t size) {
    void* ptr = nullptr;
    cudaError_t error = cudaMallocHost(&ptr, size);
    check_cuda_error(error, "Pinned memory allocation");
    return ptr;
}

void CudaRuntime::deallocate_pinned(void* ptr) {
    if (ptr) {
        cudaError_t error = cudaFreeHost(ptr);
        check_cuda_error(error, "Pinned memory deallocation");
    }
}

void CudaRuntime::copy_device_to_device(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    check_cuda_error(error, "Device to device copy");
}

void CudaRuntime::copy_host_to_device(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    check_cuda_error(error, "Host to device copy");
}

void CudaRuntime::copy_device_to_host(void* dst, const void* src, size_t size) {
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    check_cuda_error(error, "Device to host copy");
}

void CudaRuntime::copy_host_to_device_async(void* dst, const void* src, size_t size, GpuStream& stream) {
    CudaStream& cuda_stream = static_cast<CudaStream&>(stream);
    cudaError_t error = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cuda_stream.get_cuda_stream());
    check_cuda_error(error, "Async host to device copy");
}

void CudaRuntime::copy_device_to_host_async(void* dst, const void* src, size_t size, GpuStream& stream) {
    CudaStream& cuda_stream = static_cast<CudaStream&>(stream);
    cudaError_t error = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, cuda_stream.get_cuda_stream());
    check_cuda_error(error, "Async device to host copy");
}

std::unique_ptr<GpuStream> CudaRuntime::create_stream() {
    return std::make_unique<CudaStream>(current_device_);
}

void CudaRuntime::destroy_stream(GpuStream* stream) {
    delete stream;
}

std::unique_ptr<GpuEvent> CudaRuntime::create_event() {
    return std::make_unique<CudaEvent>(current_device_);
}

void CudaRuntime::destroy_event(GpuEvent* event) {
    delete event;
}

void CudaRuntime::synchronize() {
    cudaDeviceSynchronize();
}

bool CudaRuntime::is_available(int device_id) {
    return device_id >= 0 && device_id < device_count_ && device_count_ > 0;
}

std::unique_ptr<GpuMemoryPool> CudaRuntime::create_memory_pool() {
    return std::make_unique<CudaMemoryPool>(current_device_);
}

std::unique_ptr<GpuKernel> CudaRuntime::compile_kernel(const std::string& source, 
                                                      const std::string& kernel_name) {
    // This is a simplified implementation
    // In a real implementation, you would:
    // 1. Compile CUDA source to PTX using nvcc or CUDA compiler API
    // 2. Load the compiled kernel
    // 3. Return a CudaKernel with the loaded function
    
    // For now, throw an exception to indicate this needs implementation
    throw std::runtime_error("CUDA kernel compilation not yet implemented - requires NVCC integration");
}

std::string CudaRuntime::version() const {
    int version;
    cudaRuntimeGetVersion(&version);
    return std::to_string(version);
}

void CudaRuntime::check_cuda_error(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error in " << operation << ": " << cudaGetErrorString(error);
        throw std::runtime_error(oss.str());
    }
}

bool CudaRuntime::is_cuda_available() {
    int count = 0;
    cudaError_t error = cudaGetDeviceCount(&count);
    return error == cudaSuccess && count > 0;
}

#endif // __CUDACC__

} // namespace cuda
} // namespace gpu
} // namespace jin