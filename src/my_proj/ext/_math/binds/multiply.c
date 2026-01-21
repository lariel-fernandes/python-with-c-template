#include "../lib/multiply.h"

#include <Python.h>

PyObject* py_multiply(PyObject* self, PyObject* args)
{
    long a, b;
    if (!PyArg_ParseTuple(args, "ll", &a, &b)) {
        return NULL;
    }
    return PyLong_FromLong(multiply(a, b));
}
