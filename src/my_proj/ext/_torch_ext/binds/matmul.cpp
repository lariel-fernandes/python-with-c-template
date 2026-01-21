#include "matmul.h"

#include <torch/torch.h>

#include "../lib/matmul.h"

torch::Tensor py_matmul(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match: A.shape[1] must equal B.shape[0]");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.device() == B.device(), "Both tensors must be on the same device");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous")

    const int m = A.size(0);
    const int k = A.size(1);
    const int n = B.size(1);
    auto C = torch::empty({m, n}, A.options());

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    if (A.is_cuda()) {
#ifdef WITH_CUDA
        matmul_cuda(A_ptr, B_ptr, C_ptr, m, k, n);
#else
        TORCH_CHECK(false, "CUDA support not available. Please rebuild with CUDA.");
#endif
    } else {
        matmul_cpu(A_ptr, B_ptr, C_ptr, m, k, n);
    }

    return C;
}
