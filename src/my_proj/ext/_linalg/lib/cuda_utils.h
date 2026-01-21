#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* cuda_get_error_string(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif
