import numpy as np
import typing as _tp
from ._typing import Indexable as _Indexable


def jacobi(
    fn: _tp.Callable,
    x: _tp.Union[float, _Indexable[float]],
    *args,
    diagonal: bool = False,
    method: int = None,
    mask: np.ndarray = None,
    rtol: float = 0,
    maxiter: int = 10,
    maxgrad: int = 3,
    step: _tp.Tuple[float, float] = None,
    diagnostic: dict = None,
) -> _tp.Tuple[np.ndarray, np.ndarray]:
    """
    Return first derivative and its error estimate.

    Parameters
    ----------
    fn : Callable
        Function with the signature `fn(x, *args)`, where `x` is a number or an
        array of numbers and `*args` are optional auxiliary arguments.
    x : Number or array of numbers
        The derivative is computed with respect to `x`. If `x` is an array, the Jacobi
        matrix is computed with respect to each element of `x`.
    *args : tuple
        Additional arguments passed to the function.
    diagonal : boolean, optional
        If it is known that the Jacobian computed from the function contains has
        off-diagonal entries that are all zero, the calculation can be speed up
        significantly. Set this to true to only compute the diagonal entries of the
        Jacobi matrix, which are returned as a 1D array. This is faster and uses much
        less memory if the vector x is very large. Default is False.
    method : {-1, 0, 1} or None, optional
        Whether to compute central (0), forward (1) or backward derivatives (-1).
        The default (None) uses auto-detection.
    mask : array or None, optional
        If `x` is an array and `mask` is not None, compute the Jacobi matrix only for the
        part of the array selected by the mask.
    rtol : float, optional
        Relative tolerance for the derivative. The algorithm stops when this relative
        tolerance is reached. If 0 (the default), the algorithm iterates until the
        error estimate of the derivative does not improve further.
    maxiter : int, optional
        Maximum number of iterations of the algorithm.
    maxgrad : int, optional
        Maximum grad of the extrapolation polynomial.
    step : tuple of float or None, optional
        Factors that reduce the step size in each iteration relative to the previous
        step.
    diagnostic : dict or None, optional
        If an empty dict is passed to this keyword, it is filled with diagnostic
        information produced by the algorithm. This reduces performance and is only
        intended for debugging.

    Returns
    -------
    array, array
        Derivative and its error estimate.
    """
    if diagonal:
        # TODO maybe solve this without introducing a wrapper function
        return jacobi(
            lambda dx, x, *args: fn(x + dx, *args),
            0,
            x,
            *args,
            method=method,
            rtol=rtol,
            maxiter=maxiter,
            maxgrad=maxgrad,
            step=step,
            diagnostic=diagnostic,
        )

    if maxiter <= 0:
        raise ValueError("maxiter must be > 0")
    if maxgrad < 0:
        raise ValueError("maxgrad must be >= 0")
    if step is not None:
        if not (0 < step[0] < 0.5):
            raise ValueError("step[0] must be between 0 and 0.5")
        if not (0 < step[1] < 1):
            raise ValueError("step[1] must be between 0 and 1")
    if method is not None and method not in (-1, 0, 1):
        raise ValueError("method must be -1, 0, 1")

    x = np.asarray(x, dtype=float)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise ValueError("mask must be a boolean array")
        if mask.shape != x.shape:
            raise ValueError("mask shape must match x shape")

    if diagnostic is not None:
        diagnostic["method"] = np.zeros(x.size, dtype=np.int8)
        diagnostic["iteration"] = np.zeros(x.size, dtype=np.uint8)
        diagnostic["residual"] = [[] for _ in range(x.size)]

    f0 = None
    jac = None
    err = None
    it = np.nditer(x, flags=["c_index", "multi_index"])
    while not it.finished:
        k = it.index
        kx = it.multi_index if it.has_multi_index else ...
        if mask is not None and not mask[kx]:
            it.iternext()
            continue
        xk = it[0]
        # if step is None, use optimal step sizes for central derivatives
        h = _steps(xk, step or (0.25, 0.5), maxiter)
        # if method is None, auto-detect for each x[k]
        md, f0, r = _first(method, f0, fn, x, kx, h[0], args)
        # f0 is not guaranteed to be set here and can be still None

        if md != 0 and step is None:
            # need different step sizes for forward derivatives to avoid overlap
            h = _steps(xk, (0.25, 0.125), maxiter)

        r = np.asarray(r, dtype=float)
        re = np.full_like(r, np.inf)
        todo = np.ones_like(r, dtype=bool)
        fd = [np.reshape(r.copy(), -1)]

        if jac is None:  # first iteration
            jac = np.zeros(r.shape + x.shape, dtype=r.dtype)
            err = np.zeros(r.shape + x.shape, dtype=r.dtype)
            if diagnostic is not None:
                diagnostic["call"] = np.zeros((r.size, x.size), dtype=np.uint8)

        if diagnostic is not None:
            diagnostic["method"][k] = md
            diagnostic["call"][:, k] = 2 if md == 0 else 3

        for i in range(1, len(h)):
            fdi = _derive(md, f0, fn, x, kx, h[i], args)
            fd.append(np.reshape(fdi, -1) if i == 1 else fdi[todo])
            if diagnostic is not None:
                diagnostic["call"][todo, k] += 2
                diagnostic["iteration"][k] += 1

            # polynomial fit with one extra degree of freedom;
            # use latest maxgrad + 1 data points
            grad = min(i - 1, maxgrad)
            start = i - (grad + 1)
            stop = i + 1
            q, c = np.polyfit(
                h[start:stop] ** 2, fd[start:], grad, rcond=None, cov=True
            )
            ri = q[-1]
            # pulls have roughly unit variance, however,
            # the pull distribution is not gaussian and looks
            # more like student's t
            rei = c[-1, -1] ** 0.5

            # update estimates that have smaller estimated error
            sub_todo = rei < re[todo]
            todo1 = todo.copy()
            todo[todo1] = sub_todo
            r[todo] = ri[sub_todo]
            re[todo] = rei[sub_todo]

            # do not improve estimates further which meet the tolerance
            if rtol > 0:
                sub_todo &= rei > rtol * np.abs(ri)
                todo[todo1] = sub_todo

            if diagnostic is not None:
                re2 = re.copy()
                re2[todo1] = rei
                diagnostic["residual"][k].append(re2)

            if np.sum(todo) == 0:
                break

            # shrink previous vectors of estimates
            fd = [fdi[sub_todo] for fdi in fd]

        if jac.ndim == 0:
            jac[...] = r
            err[...] = re
        elif jac.ndim == 1:
            jac[kx] = r
            err[kx] = re
        else:
            jac[(...,) + kx] = r
            err[(...,) + kx] = re

        it.iternext()

    return jac, err


def _steps(p, step, maxiter):
    h0, factor = step
    h = p * h0
    if not h != 0:  # also works if p is NaN
        h = h0
    return h * factor ** np.arange(maxiter)


def _derive(mode, f0, f, x, i, h, args):
    x1 = x.copy()
    x2 = x.copy()
    if mode == 0:
        x1[i] += h
        x2[i] -= h
        return (f(x1, *args) - f(x2, *args)) * (0.5 / h)
    h = h * mode
    x1[i] += h
    x2[i] += 2 * h
    f1 = f(x1, *args)
    f2 = f(x2, *args)
    return (-3 * f0 + 4 * f1 - f2) * (0.5 / h)


def _first(method, f0, f, x, i, h, args):
    norm = 0.5 / h
    f1 = None
    f2 = None
    if method is None or method == 0:
        x1 = x.copy()
        x2 = x.copy()
        x1[i] -= h
        x2[i] += h
        f1 = f(x1, *args)
        f2 = f(x2, *args)
        if method is None:
            if np.any(np.isnan(f1)):  # forward method
                method = 1
            elif np.any(np.isnan(f2)):  # backward method
                method = -1
            else:
                method = 0
    if method == 0:
        return method, None, (f1 - f2) * norm
    if f0 is None:
        f0 = f(x, *args)
    if method == -1:
        h = -h
        norm = -norm
    if f1 is None:
        x1 = x.copy()
        x1[i] += h
        f1 = f(x1, *args)
    elif method == 1:
        f1 = f2
    x2 = x.copy()
    x2[i] += 2 * h
    f2 = f(x2, *args)
    return method, f0, (-3 * f0 + 4 * f1 - f2) * norm
