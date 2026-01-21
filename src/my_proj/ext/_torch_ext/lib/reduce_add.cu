#include "reduce_add.h"

#include <ATen/ATen.h>
#include <cstdio>
#include <cuda_runtime.h>

#include "./utils.h"

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BLOCK_SIZE WARP_SIZE* WARPS_PER_BLOCK
static_assert(WARP_SIZE >= WARPS_PER_BLOCK); // A warp has enough threads to reduce the partial sums of all other warps in a block

// IDs of the threads corresponding to the first lane of each warp in a block
#define THREAD_ID_LANE_0_WARP_0 0 * WARP_SIZE
#define THREAD_ID_LANE_0_WARP_1 1 * WARP_SIZE
#define THREAD_ID_LANE_0_WARP_2 2 * WARP_SIZE
#define THREAD_ID_LANE_0_WARP_3 3 * WARP_SIZE

// Helper to handle shuffle operations for different types
template <typename T> __device__ __forceinline__ T shfl_down(T val, int offset)
{
    return __shfl_down_sync(0xFFFFFFFF, val, offset);
}

// Helper overload for handling the conversion between c10::Half (torch) and __half (cuda)
template <> __device__ __forceinline__ at::Half shfl_down<at::Half>(at::Half val, int offset)
{
    __half native_val = val;
    __half result = __shfl_down_sync(0xFFFFFFFF, native_val, offset);
    return at::Half(result);
}

template <typename T, int WARP_SIZE_P> __device__ void warp_unroll(T* sum)
{
    if (WARP_SIZE_P >= 128) *sum += shfl_down(*sum, 64);
    if (WARP_SIZE_P >= 64) *sum += shfl_down(*sum, 32);
    if (WARP_SIZE_P >= 32) *sum += shfl_down(*sum, 16);
    if (WARP_SIZE_P >= 16) *sum += shfl_down(*sum, 8);
    if (WARP_SIZE_P >= 8) *sum += shfl_down(*sum, 4);
    if (WARP_SIZE_P >= 4) *sum += shfl_down(*sum, 2);
    if (WARP_SIZE_P >= 2) *sum += shfl_down(*sum, 1);
}

template <typename T, int WARPS_PER_BLOCK_P> __device__ void final_warp_unroll(T* sum)
{
    if (WARPS_PER_BLOCK_P >= 128) *sum += shfl_down(*sum, 64);
    if (WARPS_PER_BLOCK_P >= 64) *sum += shfl_down(*sum, 32);
    if (WARPS_PER_BLOCK_P >= 32) *sum += shfl_down(*sum, 16);
    if (WARPS_PER_BLOCK_P >= 16) *sum += shfl_down(*sum, 8);
    if (WARPS_PER_BLOCK_P >= 8) *sum += shfl_down(*sum, 4);
    if (WARPS_PER_BLOCK_P >= 4) *sum += shfl_down(*sum, 2);
    if (WARPS_PER_BLOCK_P >= 2) *sum += shfl_down(*sum, 1);
}

template <typename T> __global__ void kernel(T* output, const T* input, const int inputSize)
{
    T sum = static_cast<T>(0);

    // Static shared memory sized for the largest type (double = 8 bytes)
    __shared__ char shared_bytes[WARPS_PER_BLOCK * 8];
    T* shared = reinterpret_cast<T*>(shared_bytes);

    // Every thread sums 2 elements from the device memory, if within input range
    const int start = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    const int end = start + blockDim.x;
    sum = (start < inputSize ? input[start] : static_cast<T>(0)) + (end < inputSize ? input[end] : static_cast<T>(0));

    // Reduce each warp by shuffling to compute its partial sum
    warp_unroll<T, WARP_SIZE>(&sum);

    // First lane of each warp stores the warp's partial sum contiguously in the shared memory
    if (threadIdx.x == THREAD_ID_LANE_0_WARP_0) shared[0] = sum;
    if (threadIdx.x == THREAD_ID_LANE_0_WARP_1) shared[1] = sum;
    if (threadIdx.x == THREAD_ID_LANE_0_WARP_2) shared[2] = sum;
    if (threadIdx.x == THREAD_ID_LANE_0_WARP_3) shared[3] = sum;
    __syncthreads();

    // Shared memory is reduced by shuffling in the first warp of the block
    if (threadIdx.x < WARPS_PER_BLOCK) {
        sum = shared[threadIdx.x];
        final_warp_unroll<T, WARPS_PER_BLOCK>(&sum);
    }

    // First thread of each block stores the block's partial sum contiguously in the device memory
    if (threadIdx.x == 0) output[blockIdx.x] = sum;
}

template <typename T> T reduce_add_cuda(const T* input, int inputSize)
{
    if (inputSize == 0) return static_cast<T>(0);
    if (inputSize == 1) return input[0];

    T result = static_cast<T>(0);
    T* staging = nullptr;
    cudaError_t cudaError = cudaSuccess;

    while (inputSize > 1) {
        const int gridSize = div_ceil(inputSize, 2 * BLOCK_SIZE);

        if (staging == nullptr) {
            // First pass: Reduce from input to staging with size of 1 partial sum per block
            CUDA_CHECK_ERROR(&cudaError, cleanup, cudaMalloc(&staging, sizeof(T) * gridSize));
            kernel<T><<<gridSize, BLOCK_SIZE>>>(staging, input, inputSize);
        } else {
            // Further passes: Reduce from staging to staging
            kernel<T><<<gridSize, BLOCK_SIZE>>>(staging, staging, inputSize);
        }

        CUDA_CHECK_LAST_ERROR(&cudaError, cleanup);
        CUDA_CHECK_ERROR(&cudaError, cleanup, cudaDeviceSynchronize());

        inputSize = gridSize;
    }

    CUDA_CHECK_ERROR(&cudaError, cleanup, cudaMemcpy(&result, staging, sizeof(T), cudaMemcpyDeviceToHost));

cleanup:
    cudaFree(staging);
    if (cudaError != cudaSuccess) exit(cudaError);
    return result;
}

// Explicit template instantiations for all types supported by AT_DISPATCH_FLOATING_TYPES_AND_HALF
template float reduce_add_cuda<float>(const float* input, int inputSize);

template double reduce_add_cuda<double>(const double* input, int inputSize);

template at::Half reduce_add_cuda<at::Half>(const at::Half* input, int inputSize);

#ifdef STANDALONE_BUILD
int main()
{
    printf("reduce_add_cuda\n");
    return 0;
}
#endif
