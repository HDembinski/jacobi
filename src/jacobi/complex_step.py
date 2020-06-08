import numpy as np


def jacobi(f, x):
    step = np.finfo(float).eps
    cx = np.asarray(x, dtype=complex)
    cx.imag = step
    r = f(cx).imag / step
    return r
