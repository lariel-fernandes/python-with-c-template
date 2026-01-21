#ifndef TORCH_LIB_REDUCE_ADD_H
#define TORCH_LIB_REDUCE_ADD_H
#ifdef WITH_CUDA

template <typename T> T reduce_add_cuda(const T* input, int inputSize);

#endif
#endif
