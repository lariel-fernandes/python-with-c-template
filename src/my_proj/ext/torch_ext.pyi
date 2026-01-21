from __future__ import annotations
import torch

__all__: list[str] = ["matmul", "reduce_add"]

def matmul(arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
    """
    Custom matrix multiplication with CPU/CUDA support
    """

def reduce_add(arg0: torch.Tensor) -> torch.Tensor:
    """
    Sum all elements in tensor without modifying it. Returns a new tensor of shape (1,)
    """
