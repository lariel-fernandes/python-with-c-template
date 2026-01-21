#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PY_ARRAY
#include "binds/matmul.h"

#include <numpy/arrayobject.h>

static PyMethodDef methods[] = {
    {"matmul",
     py_matmul,
     METH_VARARGS,
     "Matrix multiplication using CUDA.\n\n"
     "Computes C = A @ B using GPU acceleration.\n\n"
     "This function performs matrix multiplication on the GPU, which can be\n"
     "much faster than CPU-based NumPy for large matrices. All data transfers\n"
     "between CPU and GPU are handled automatically.\n\n"
     "Parameters\n"
     "----------\n"
     "A : numpy.ndarray\n"
     "    Input matrix A with shape (m, k) and dtype float32 or float64.\n"
     "    Will be automatically converted to float32 if needed.\n"
     "B : numpy.ndarray\n"
     "    Input matrix B with shape (k, n) and dtype float32 or float64.\n"
     "    Will be automatically converted to float32 if needed.\n\n"
     "Returns\n"
     "-------\n"
     "numpy.ndarray\n"
     "    Result matrix C with shape (m, n) and dtype float32.\n\n"
     "Raises\n"
     "------\n"
     "ValueError\n"
     "    If inputs are not 2D arrays or if shapes are incompatible.\n"
     "RuntimeError\n"
     "    If a CUDA error occurs during computation.\n\n"
     "Notes\n"
     "-----\n"
     "- Input matrices are automatically converted to float32 (single precision)\n"
     "- Matrices must be in row-major order (NumPy default)\n"
     "- For small matrices, CPU-based NumPy may be faster due to overhead\n"
     "- Optimal performance is typically achieved with matrices >= 256x256\n\n"
     "Examples\n"
     "--------\n"
     ">>> import numpy as np\n"
     ">>> from my_proj import linalg\n"
     ">>> A = np.random.rand(100, 50).astype(np.float32)\n"
     ">>> B = np.random.rand(50, 80).astype(np.float32)\n"
     ">>> C = linalg.matmul(A, B)\n"
     ">>> C.shape\n"
     "(100, 80)\n"},
    {NULL, NULL, 0, NULL} // Sentinel
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "linalg",
    "Fast linear algebra operations implemented in CUDA",
    -1,
    methods,
};

PyMODINIT_FUNC PyInit_linalg(void)
{
    import_array();
    return PyModule_Create(&module);
}
