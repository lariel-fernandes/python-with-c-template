from my_proj.ext import math

try:
    from my_proj.ext import linalg
except ImportError:
    linalg = None

__all__ = [
    "math",
    "linalg",
]

# torch_fast_gatv2
#  torch native module
#  graph input generator
#  csr converter
#  numpy dummy with received weights, compare to torch native module using extracted weights and csr graph input
#  cuda program v1, replace dummy and compare, minimal optimization in ada
#  figure out the backward pass, have a numpy dummy too, compare to native
#  implement backward pass in cuda, replace dummy and compare, minimal optim in ada
#  wrap cuda programs in autograd func and layer, build the challenger module
#  move benchmark to dedicated package, organize, adjust outputs for single param set
#  further optimize in ada, optimize in blackwell
#  adjust benchmark for multi param set and plots (e.g. 1Mx128 nodes, 256x64 edges, 4x8 heads, ~1M fp16 params))

# chores
#   try out the vscode debugger with torch imports commenting, int main uncommenting and a launcher that compiles and runs the single .cu file
#   adjust docs of reduce add for the new supported types
#   unit test reduce add

# features
#   solve the non-contiguous transposition in backward pass
#   learn cutlass and CuTe, learn cuda streams

# build issues
#  check if docker build still works
#  find a way to run clang-tidy in the command line
#  accelerate the uv sync build with ninja and/or ccache
