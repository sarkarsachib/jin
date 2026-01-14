// CUDA Kernels for Jin Tensor Operations
// Element-wise operations, reductions, and matrix operations

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

// Element-wise operations
__global__ void cuda_add_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void cuda_multiply_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void cuda_subtract_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void cuda_divide_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] / b[idx];
    }
}

// Unary operations
__global__ void cuda_exp_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __expf(input[idx]);
    }
}

__global__ void cuda_log_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __logf(input[idx]);
    }
}

__global__ void cuda_relu_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void cuda_tanh_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __tanhf(input[idx]);
    }
}

__global__ void cuda_sigmoid_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + __expf(-input[idx]));
    }
}

__global__ void cuda_neg_kernel(const float* input, float* output, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = -input[idx];
    }
}

// Matrix multiplication (shared memory optimized)
__global__ void cuda_matmul_kernel(const float* a, const float* b, float* c, 
                                   int m, int n, int k) {
    // a is m x k, b is k x n, c is m x n
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// Reduction operations
__global__ void cuda_sum_kernel(const float* input, float* output, size_t size) {
    __shared__ float sdata[256];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < size) ? input[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

__global__ void cuda_mean_kernel(const float* input, float* output, size_t size) {
    __shared__ float sdata[256];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < size) ? input[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0] / size);
    }
}

// Transpose operation
__global__ void cuda_transpose_kernel(const float* input, float* output, 
                                     int rows, int cols) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Gradient kernels for autodiff
__global__ void cuda_relu_backward_kernel(const float* input, const float* grad_output, 
                                        float* grad_input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

__global__ void cuda_tanh_backward_kernel(const float* input, const float* grad_output, 
                                         float* grad_input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float tanh_val = __tanhf(input[idx]);
        grad_input[idx] = (1.0f - tanh_val * tanh_val) * grad_output[idx];
    }
}

__global__ void cuda_sigmoid_backward_kernel(const float* input, const float* grad_output, 
                                            float* grad_input, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_val = 1.0f / (1.0f + __expf(-input[idx]));
        grad_input[idx] = sigmoid_val * (1.0f - sigmoid_val) * grad_output[idx];
    }
}