#ifndef PY_MATMUL_H
#define PY_MATMUL_H

#include <torch/torch.h>

torch::Tensor py_matmul(torch::Tensor A, torch::Tensor B);

#endif
