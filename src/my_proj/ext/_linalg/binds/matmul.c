#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL PY_ARRAY
#define NO_IMPORT_ARRAY // https://numpy.org/doc/stable/reference/c-api/array.html#c.NO_IMPORT_ARRAY
#include "matmul.h"

#include <numpy/arrayobject.h>

#include "../lib/cuda_utils.h"
#include "../lib/matmul.h"

PyObject* py_matmul(PyObject* self, PyObject* args)
{
    PyObject *A_obj, *B_obj;
    if (!PyArg_ParseTuple(args, "OO", &A_obj, &B_obj)) {
        return NULL;
    }

    PyArrayObject* A_arr = (PyArrayObject*) PyArray_FROM_OTF(A_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* B_arr = (PyArrayObject*) PyArray_FROM_OTF(B_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (A_arr == NULL || B_arr == NULL) {
        Py_XDECREF(A_arr);
        Py_XDECREF(B_arr);
        return NULL;
    }

    if (PyArray_NDIM(A_arr) != 2 || PyArray_NDIM(B_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "Both inputs must be 2D arrays");
        Py_DECREF(A_arr);
        Py_DECREF(B_arr);
        return NULL;
    }

    const npy_intp m = PyArray_DIM(A_arr, 0);  // Rows in A
    const npy_intp k = PyArray_DIM(A_arr, 1);  // Cols in A
    const npy_intp k2 = PyArray_DIM(B_arr, 0); // Rows in B
    const npy_intp n = PyArray_DIM(B_arr, 1);  // Cols in B
    if (k != k2) {
        PyErr_Format(PyExc_ValueError, "Inner dimensions must match: (%ld, %ld), (%ld, %ld)", (long) m, (long) k, (long) k2, (long) n);
        Py_DECREF(A_arr);
        Py_DECREF(B_arr);
        return NULL;
    }

    const npy_intp dims[2] = {m, n};
    PyArrayObject* C_arr = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (C_arr == NULL) {
        Py_DECREF(A_arr);
        Py_DECREF(B_arr);
        Py_XDECREF(C_arr);
        return NULL;
    }

    const float* A = PyArray_DATA(A_arr);
    const float* B = PyArray_DATA(B_arr);
    float* C = PyArray_DATA(C_arr);

    const cudaError_t error = matmul(A, B, C, (int) m, (int) k, (int) n);
    if (error != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, cuda_get_error_string(error));
        Py_DECREF(A_arr);
        Py_DECREF(B_arr);
        Py_DECREF(C_arr);
        return NULL;
    }

    Py_DECREF(A_arr);
    Py_DECREF(B_arr);
    return (PyObject*) C_arr;
}
