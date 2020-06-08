import numpy as np


def jacobi(f, x):

    # implementation based on boost.math
    step = np.finfo(x.dtype).eps
    inv_step = 1 / step
    c_dtype_str = x.dtype.str.replace("f", "c")
    x = np.asarray(x, dtype=c_dtype_str)
    x.imag = step
    r = f(x).imag / step
    return r
