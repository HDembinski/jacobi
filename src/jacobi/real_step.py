import numpy as np


def jacobi(f, x):
    # implementation based on boost.math

    x = np.asfarray(x)
    step = np.finfo(x.dtype).eps
    c_dtype_str = np.dtype(step).str.replace("f", "c")

    inv_step = 1 / step
    x = np.asarray(x, dtype=c_dtype_str)
    x.imag = step
    r = f(x).imag / step
    return r
