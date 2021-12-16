import numpy as np


def _steps(p, h0=0.125, factor=0.5):
    eps = np.finfo(float).resolution
    h = p * h0
    if h == 0:
        h = h0
    n = int(np.log(eps) / np.log(h0 ** 2)) + 1
    return h * factor ** np.arange(n)


def _fodd(f, x, p):
    return 0.5 * (f(x, p) - f(x, -p))


def _central(f, x, p, h):
    hinv = 1.0 / h
    return _fodd(f, x, p + h) * hinv


def vjacobi(f, x, p, *, rtol=None, maxiter=None):
    x = np.atleast_1d(x)

    shape = np.shape(x)

    h = _steps(p)
    todo = np.ones(shape, dtype=bool)

    eps = np.finfo(float).resolution
    if rtol is None:
        rtol = 10 * eps
    else:
        rtol = max(rtol, 10 * eps)

    fd = []
    for i, hi in enumerate(h):
        fdi = _central(f, x[todo], p, hi)
        fd.append(fdi)

        if i == 0:
            f_shape = np.shape(fdi)
            # maybe this can be relaxed later
            assert f_shape == shape
            r = np.empty(f_shape)
            re = np.full(f_shape, np.inf)
            r[:] = fdi
            continue

        # polynomial fit with one extra degree of freedom
        q, C = np.polyfit(h[: i + 1] ** 2, fd, i - 1, rcond=None, cov=True)
        ri = q[-1]
        # pulls have roughly unit variance, however,
        # the pull distribution is not gaussian and looks
        # more like student's t
        rei = np.maximum(C[-1, -1] ** 0.5, eps * np.abs(ri))

        # update estimates that have improved (smaller error)
        sub_todo = rei < re[todo]
        todo1 = todo.copy()
        todo[todo1] = sub_todo
        r[todo] = ri[sub_todo]
        re[todo] = rei[sub_todo]

        if maxiter and i >= maxiter:
            break

        # do not improve estimates further which meet the tolerance
        sub_todo &= rei > rtol * np.abs(ri)
        todo[todo1] = sub_todo

        if np.sum(todo) == 0:
            break

        # shrink previous vectors of estimates
        fd = [fdi[sub_todo] for fdi in fd]

    return r, re


def jacobi(f, p, *, rtol=None, maxiter=None):
    h = _steps(p)

    eps = np.finfo(float).resolution
    if rtol is None:
        rtol = 10 * eps
    else:
        rtol = max(rtol, 10 * eps)

    fd = []
    for i, hi in enumerate(h):
        fdi = _central(f, p, hi)
        fd.append(fdi)

        if i == 0:
            f_shape = np.shape(fdi)
            r = np.empty(f_shape)
            re = np.full(f_shape, np.inf)
            r[:] = fdi
            continue

        # polynomial fit with one extra degree of freedom
        q, C = np.polyfit(h[: i + 1] ** 2, fd, i - 1, rcond=None, cov=True)
        ri = q[-1]
        # pulls have roughly unit variance, however,
        # the pull distribution is not gaussian and looks
        # more like student's t
        rei = np.maximum(C[-1, -1] ** 0.5, eps * np.abs(ri))

        # update estimates that have improved (smaller error)
        better = rei < re
        r[better] = ri[better]
        re[better] = rei[better]

        if maxiter and i >= maxiter:
            break

        # do not improve estimates further which meet the tolerance
        better &= rei > rtol * np.abs(ri)
        todo[todo1] = sub_todo

        if np.sum(todo) == 0:
            break

        # shrink previous vectors of estimates
        fd = [fdi[sub_todo] for fdi in fd]

    return r, re
