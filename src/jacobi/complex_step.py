import numpy as np


def jacobi(f, x):
    step = np.finfo(float).eps
    c_dtype_str = np.dtype(step).str.replace("f", "c")
    cx = np.asarray(x, dtype=c_dtype_str)
    cx.imag = step
    r = f(cx).imag / step
    return r
