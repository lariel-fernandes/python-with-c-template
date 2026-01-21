#ifndef MATMUL_H
#define MATMUL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t matmul(const float* A, const float* B, float* C, int m, int k, int n);

#ifdef __cplusplus
}
#endif

#endif
