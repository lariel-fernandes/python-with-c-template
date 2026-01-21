#ifndef TORCH_MATMUL_H
#define TORCH_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

void matmul_cpu(const float* A, const float* B, float* C, int m, int k, int n);

#ifdef WITH_CUDA
void matmul_cuda(const float* A, const float* B, float* C, int m, int k, int n);
#endif

#ifdef __cplusplus
}
#endif

#endif
