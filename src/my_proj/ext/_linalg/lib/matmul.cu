#include "matmul.h"

#include <cstdio>

__global__ void matmul_kernel(const float* A, const float* B, float* C, const int m, const int k, const int n)
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

extern "C" {
cudaError_t matmul(const float* A, const float* B, float* C, const int m, const int k, const int n)
{
    cudaError_t error = cudaSuccess;

    float* A_dev = nullptr;
    float* B_dev = nullptr;
    float* C_dev = nullptr;

    const size_t size_A = m * k * sizeof(float);
    const size_t size_B = k * n * sizeof(float);
    const size_t size_C = m * n * sizeof(float);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        // Round up formula: (size + blockSize - 1) / blockSize
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (m + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    error = cudaMalloc(&A_dev, size_A);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMalloc(&B_dev, size_B);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMalloc(&C_dev, size_C);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(A_dev, A, size_A, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(B_dev, B, size_B, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) goto cleanup;

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(A_dev, B_dev, C_dev, m, k, n);
    error = cudaGetLastError();
    if (error != cudaSuccess) goto cleanup;

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) goto cleanup;

    error = cudaMemcpy(C, C_dev, size_C, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) goto cleanup;

cleanup:
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    return error;
}
} // extern "C"
