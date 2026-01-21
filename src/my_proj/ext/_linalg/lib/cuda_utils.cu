#include "cuda_utils.h"

extern "C" {
const char* cuda_get_error_string(const cudaError_t error)
{
    return cudaGetErrorString(error);
}
}
