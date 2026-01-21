/*
 * square.c - Python bindings for square operations
 * Wraps pure C functions from lib/square.c
 */
#include "../lib/square.h"

#include <Python.h>

PyObject* py_square(PyObject* self, PyObject* args)
{
    long n, result;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "l", &n)) {
        return NULL;
    }

    // Call pure C function
    result = square(n);

    // Convert result back to Python
    return PyLong_FromLong(result);
}
