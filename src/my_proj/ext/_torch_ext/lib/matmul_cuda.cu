#include "matmul.h"

#include <c10/cuda/CUDAException.h>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void matmul_cuda_kernel(const float* A, const float* B, float* C, const int m, const int k, const int n)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul_cuda(const float* A, const float* B, float* C, const int m, const int k, const int n)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_cuda_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, m, k, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
