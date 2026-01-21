#include <torch/extension.h>

#include "./binds/matmul.h"
#include "./binds/reduce_add.h"

PYBIND11_MODULE(torch_ext, m)
{
    m.def("matmul", &py_matmul, "Custom matrix multiplication with CPU/CUDA support");
    m.def("reduce_add", &reduce_add, "Sum all elements in tensor without modifying it. Returns a new tensor of shape (1,)");
}
