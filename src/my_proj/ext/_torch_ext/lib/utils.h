#ifndef TORCH_LIB_UTILS_H
#define TORCH_LIB_UTILS_H
#ifdef WITH_CUDA
#include <cstdio>
#include <cuda_runtime.h>

// Round up integer division
inline int div_ceil(const int dividend, const int divisor)
{
    return (dividend + divisor - 1) / divisor;
}

// If not a success: print error, store it and return true
inline bool cuda_check_error(const cudaError_t code, const char* file, const int line, cudaError_t* err_ptr)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(code));
        *err_ptr = code;
        return true;
    }
    return false;
}

// If expression errors: print error, store it and go to the indicated label
#define CUDA_CHECK_ERROR(err_ptr, label, expr)                                 \
    {                                                                          \
        if (cuda_check_error((expr), __FILE__, __LINE__, err_ptr)) goto label; \
    }

// Same as CUDA_CHECK_ERROR for async operations
#define CUDA_CHECK_LAST_ERROR(err_ptr, label)                                              \
    {                                                                                      \
        if (cuda_check_error(cudaGetLastError(), __FILE__, __LINE__, err_ptr)) goto label; \
    }

#endif
#endif
