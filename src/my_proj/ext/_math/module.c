#include "binds/multiply.h"
#include "binds/square.h"

#include <Python.h>

static PyMethodDef MathMethods[] = {
    {"multiply",
     py_multiply,
     METH_VARARGS,
     "Multiply two integers quickly using C.\n\n"
     "Args:\n"
     "    a (int): First integer\n"
     "    b (int): Second integer\n\n"
     "Returns:\n"
     "    int: The product of a and b"},
    {"square",
     py_square,
     METH_VARARGS,
     "Square an integer quickly using C.\n\n"
     "Args:\n"
     "    n (int): Integer to square\n\n"
     "Returns:\n"
     "    int: The square of n"},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef math_module = {
    PyModuleDef_HEAD_INIT,
    "math",
    "Fast math operations implemented in C",
    -1,
    MathMethods,
};

PyMODINIT_FUNC PyInit_math(void)
{
    return PyModule_Create(&math_module);
}
