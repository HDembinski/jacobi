import numpy as np


def jacobi(f, x):
    step = np.finfo(float).eps
    cx = np.asarray(x, dtype=complex)
    assert cx.ndim < 2
    if cx.ndim == 1:
        nx = cx.shape[0]
        cx.imag[0] = step
    else:
        cx.imag = step
    r = np.imag(f(cx)) / step
    if cx.ndim == 1:
        r_shape = np.shape(r)
        t = np.empty(r_shape + (nx,))
        t[..., 0] = r
        for i in range(1, nx):
            cx.imag[i - 1] = 0
            cx.imag[i] = step
            t[..., i] = np.imag(f(cx)) / step
        r = t
    return r
