import os
import time

import torch
from my_proj.ext.torch_ext import reduce_add

print(f"PID: {os.getpid()}")

torch.cuda.synchronize()

start = time.perf_counter()

X = torch.randn((1_000_000,), dtype=torch.float32, device="cuda")

print("Total time:", time.perf_counter() - start)

torch.cuda.synchronize()

Y = reduce_add(X)

torch.cuda.synchronize()

print("Result:     ", result := Y.item())
print("Expected:   ", expected := reduce_add(X.to("cpu")).item())
print("Rel. Error: ", (result - expected)/expected)
