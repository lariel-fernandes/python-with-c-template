#include "reduce_add.h"

#include <torch/torch.h>

#include "../lib/reduce_add.h"

torch::Tensor reduce_add(torch::Tensor X)
{
    TORCH_CHECK(X.dim() == 1, "X must be a 1D tensor");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    const int X_size = X.size(0);

    if (X.is_cuda()) {
#ifdef WITH_CUDA
        return AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "reduce_add_cuda", [&] {
            const scalar_t* X_ptr = X.data_ptr<scalar_t>();
            scalar_t result = reduce_add_cuda<scalar_t>(X_ptr, X_size);
            return torch::full({1}, result, X.options());
        });
#else
        TORCH_CHECK(false, "CUDA support not available. Please rebuild with CUDA.");
#endif
    } else {
        return AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "reduce_add_cpu", [&] {
            const scalar_t* X_ptr = X.data_ptr<scalar_t>();
            scalar_t result = static_cast<scalar_t>(0);
            for (int i = 0; i < X_size; i++)
                result += X_ptr[i];
            return torch::full({1}, result, X.options());
        });
    }
}
