# TODO:
# - compute second derivative and optimal step size based on that
#   http://web.media.mit.edu/~crtaylor/calculator.html
#   h = 2 sqrt(eps * abs(f(x) / f''(x))) for f''(x) != 0
#   https://en.wikipedia.org/wiki/Numerical_differentiation
# - Support arbitrary functons ℝⁱ → ℝⁿ
#   - support axis keyword

import numpy as np

any = np.any
abs = np.abs
max = np.maximum
abs = np.abs
any = np.any
sqrt = np.sqrt

eps = np.finfo(float).eps
default_rel_step = eps ** 0.5


class ErrorCounter:
    __slots__ = "n"

    def __init__(self):
        self.n = 0

    def __call__(self, type, flag):
        self.n += 1


error_count = ErrorCounter()


def jacobi(f, x, return_error=False):
    # Use central calculation and try forward/backward for those that failed.
    # Usually only one or two end-points are affected, so we don't need to be
    # extra clever.

    x = np.asarray(x, float)
    dim = x.ndim
    if dim == 0:
        x.shape = (1,)

    saved = np.seterrcall(error_count)
    try:
        with np.errstate(invalid="call"):
            error_count.n = 0
            if return_error:
                r, e = central(f, x, return_error=True)
            else:
                r = central(f, x, return_error=False)
            if error_count.n > 0:
                m = np.isnan(r)
                x2 = x[m]
                assert len(x2) > 0
                error_count.n = 0
                if return_error:
                    r2, e2 = forward(f, x2, return_error=True)
                else:
                    r2 = forward(f, x2, return_error=False)
                if error_count.n > 0:
                    m2 = np.isnan(r2)
                    x3 = x2[m2]
                    assert len(x3) > 0
                    if return_error:
                        r3, e3 = forward(f, x3, return_error=True, dir=-1)
                    else:
                        r3 = forward(f, x3, return_error=False, dir=-1)
                    r2[m2] = r3
                    if return_error:
                        e2[m2] = e3
                r[m] = r2
                if return_error:
                    e[m] = e2
    finally:
        np.seterrcall(saved)

    if dim == 0 and r.shape == (1,):
        r.shape = ()
        if return_error:
            e.shape = ()
    if return_error:
        return r, e
    return r


def _central(f, x, h, return_error):
    # assumes h > 0

    fa = f(x - h)
    fd = f(x + h)

    r1 = 0.5 * (fd - fa)

    if return_error:
        fb = f(x - 0.5 * h)
        fc = f(x + 0.5 * h)

        r2 = (4.0 / 3.0) * (fc - fb) - (1.0 / 3.0) * r1

        # round-off error estimate
        e = (1.5 * (abs(fb) + abs(fc)) + 0.5 * (abs(fa) + abs(fd))) * eps
        dy = max(abs(r1), abs(r2)) / h * abs(x) / h * eps

        # result, estimated truncation error, estimated rounding error
        return r2 / h, abs(r2 - r1) / h, e / h + dy

    return r1 / h


def _forward(f, x, h, dir, return_error):
    # assumes h > 0

    hd = h * dir
    fb = f(x + 0.5 * hd)
    fd = f(x + hd)

    r1 = 2.0 * (fd - fb)

    if return_error:
        fa = f(x + 0.25 * hd)
        fc = f(x + 0.75 * hd)

        r2 = 22.0 / 3.0 * (fd - fc) - 62.0 / 3.0 * (fc - fb) + 52.0 / 3.0 * (fb - fa)

        # round-off error estimate
        e = (20 * abs(fa) + 40 * abs(fb) + 30 * abs(fc) + 8 * abs(fd)) * eps
        dy = max(abs(r1), abs(r2)) / h * abs(x) / h * eps

        # result, estimated truncation error, estimated rounding error
        return r2 / hd, abs(r2 - r1) / h, e / h + dy

    return r1 / hd


def central(f, x, h=None, return_error=False):
    x = np.asarray(x, float)
    dim = x.ndim
    if dim == 0:
        x.shape = (1,)

    if h is None:
        h = np.abs(x) * default_rel_step  # h must be positive
        h[x == 0] = default_rel_step
    else:
        h = np.asarray(h, float)
        if h.ndim == 0:
            h = h * np.ones_like(x)

    if not return_error:
        r = _central(f, x, h, False)
        if dim == 0 and r.shape == (1,):
            r.shape = ()
        return r

    r, et, er = _central(f, x, h, True)
    e = et + er

    # with np.errstate(invalid="ignore"):
    #     m = (er < et) & (er > 0)
    #
    # if any(m):
    #     ho = h[m] * (er[m] / (2 * et[m])) ** (1.0 / 3.0)
    #     xo = x[m]
    #     ro, eto, ero = _central(f, xo, ho, True)
    #     eo = eto + ero
    #     m2 = (eo < e[m]) & (abs(ro - r[m]) < 4 * e[m])
    #     ro = ro[m2]
    #     eo = eo[m2]
    #     m[m] = m2
    #     if any(m):
    #         r[m] = ro
    #         e[m] = eo

    if dim == 0 and r.shape == (1,):
        r.shape = ()
        e.shape = ()
    return r, e


def forward(f, x, h=None, return_error=False, dir=1):
    x = np.asarray(x, float)
    dim = x.ndim
    if dim == 0:
        x.shape = (1,)

    if h is None:
        h = np.abs(x) * default_rel_step  # sign of h must be positive
        h[x == 0] = default_rel_step
    else:
        h = np.asarray(h, float)
        if h.ndim == 0:
            h = h * np.ones_like(x)

    if not return_error:
        r = _central(f, x, h, False)
        if dim == 0 and r.shape == (1,):
            r.shape = ()
        return r

    r, et, er = _forward(f, x, h, dir, True)
    e = et + er

    # with np.errstate(invalid="ignore"):
    #     m = (er < et) & (er > 0)
    #
    # if any(m):
    #     ho = h[m] * sqrt(er[m] / et[m])
    #     xo = x[m]
    #     ro, eto, ero = _forward(f, xo, ho, dir, True)
    #     eo = eto + ero
    #     m2 = (eo < e[m]) & (abs(ro - r[m]) < 4 * e[m])
    #     ro = ro[m2]
    #     eo = eo[m2]
    #     m[m] = m2
    #     if any(m):
    #         r[m] = ro
    #         e[m] = eo

    if dim == 0 and r.shape == (1,):
        r.shape = ()
        e.shape = ()
    return r, e
