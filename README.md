# Python with C and CUDA example

## References
- https://docs.python.org/3/c-api
- https://numpy.org/doc/stable/reference/c-api
- https://docs.nvidia.com/cuda/cuda-programming-guide/
- https://pytorch.org/cppdocs/
- https://pytorch.org/tutorials/advanced/cpp_extension.html
- https://pybind11.readthedocs.io/

## Prerequisites
- uv
- gcc
- clang-format

## Recommended Packages
- cuda 12.6 or later

## Install

Install SO packages required for building each of the extension modules:
```bash
find src/*/ext -name "packages.json" -exec jq -r '.[].build[], .[].runtime[]' {} \
 | sort -u | xargs sudo apt-get install -y
```

Lock and synchronize Python packages:
```bash
uv sync
```

## Use Python virtual environment
Activate:
```bash
source .venv/bin/activate
```
Deactivate:
```bash
deactivate
```

## Re-generate torch extension stubs
```bash
.venv/bin/python -c '
  import torch
  import pybind11_stubgen;

  pybind11_stubgen.main(["my_proj.ext.torch_ext", "-o", "src"]);
'
```

## Test

### PyTest
```bash
uv run pytest ./src/tests -v
```

### Test standalone CUDA program
```bash
make clean run PROGRAM=reduce_add
```

## Lint/Format

Python:
```bash
uvx ruff format
uvx ruff check --fix
```

C/C++/CUDA:
```bash
find src -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" | xargs clang-format -i
```

## Build docker

Prepare the build context:
```bash
mkdir -p build
uv export --no-dev --no-emit-project > build/requirements.txt
find src/my_proj/ext -name "packages.json" -exec jq -r '.[].build[]' {} \; | sort -u > build/packages-build.txt
find src/my_proj/ext -name "packages.json" -exec jq -r '.[].runtime[]' {} \; | sort -u > build/packages-runtime.txt
````

Build:
```bash
docker buildx build --pull -t my-proj . \
  --build-arg OS_VERSION=ubuntu24.04 \
  --build-arg CUDA_VERSION=12.6.0 \
  --build-arg PYTHON_VERSION=$(cat .python-version)
```

## Profiling

### CUDA Profiling

Build and run with `ncu`:
```bash
uv sync && ncu .venv/bin/python examples/torch_reduce_add.py
```

## Debugging

### CUDA Debugging

Build with debug symbols and run with `cuda-gdb`:
```bash
DEBUG_MODE=1 uv sync && cuda-gdb --args .venv/bin/python examples/torch_reduce_add.py
```

## Project Structure

```
.
├── CMakeLists.txt       # Not for builds, just for CLion integration 
├── Dockerfile
├── examples/...
├── MANIFEST.in          
├── pyproject.toml
├── setup.py
├── src
│   ├── my_proj
│   │   ├── ext/...      # C/C++/CUDA extensions
│   │   ├── __init__.py  # Extensions re-exports
│   │   └── torch.py     # PyTorch wrappers for the `torch` C++/CUDA extension
│   └── tests/...
└── uv.lock
```

### Extension Structure
Every C/C++/CUDA extension has the following structure:
```
_my_ext                   # Builds as `my_proj.my_ext` (no leading underscore)
├── binds
│   ├── <feature>.c/cpp     # Feature interfacing with Python/PyTorch/NumPy types
│   └── <feature>.h
├── lib
│   ├── <feature>.c/cpp/cu  # Standalone feature implementation 
│   └── <feature>.h
├── packages.json           # Required OS packages
└── module.c/cpp            # Methods table
```

The following extensions are available:
- `_math`: Pure C.
- `_linalg`: CUDA-only implementation NumPy interface in C (not installed if CUDA is missing, requires numpy).
- `_torch_ext`: Hybrid CUDA/CPU implementation with PyTorch interface in C++ (installed in CPU-only mode if CUDA is missing, requires torch).

## Dev. Environment Troubleshooting

### `nvcc` not found when running `uv sync`
**Issue:**
```
error: nvcc not found at 'XXXX/bin/nvcc'. Ensure CUDA path 'XXXX' is correct.
```
**Solution:**
```bash
# Replacing `XXXX` with the expected CUDA path
mkdir -p XXXX/bin
sudo ln -s $(which nvcc) XXXX/bin/nvcc
```

### `CMAKE_CUDA_COMPILER` not found when loading the project in CLion
**Issue:**
```
No CMAKE_CUDA_COMPILER could be found.

Tell CMake where to find the compiler by setting either the environment
variable "CUDACXX" or the CMake cache entry CMAKE_CUDA_COMPILER to the full
path to the compiler, or to the compiler name if it is in the PATH.
```

**Solution:**
1. Find the `nvcc` path:
   ```bash
   which nvcc
   ```
2. Go to `File` -> `Settings` -> `Build, Execution, Deployment` -> `CMake`
3. In every profile, go to `Environment` and add the environment variable `CUDACXX=.../nvcc` pointing to the `nvcc` path.
