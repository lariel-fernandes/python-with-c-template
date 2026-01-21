import torch
import torch.nn as nn

from my_proj.ext.torch_ext import matmul as _matmul


class MatmulFunction(torch.autograd.Function):
    """Custom autograd function for float32 matrix multiplication."""

    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(A, B)
        return _matmul(A, B)

    @staticmethod
    def backward(ctx, grad_C: torch.Tensor):  # noqa
        A, B = ctx.saved_tensors
        grad_A = grad_B = None

        # Using torch.matmul because the transposition produces a non-contiguous
        # memory layout, which our custom matmul implementation doesn't support
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_C, B.t())

        if ctx.needs_input_grad[1]:
            grad_B = torch.matmul(A.t(), grad_C)

        return grad_A, grad_B


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """float32 matric multiplication with autograd support."""
    return MatmulFunction.apply(A, B)


class MatmulLayer(nn.Module):
    """Custom layer for float32 matrix multiplication."""

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return matmul(A, B)

    def extra_repr(self):
        return "device-agnostic float32 matrix multiplication"
