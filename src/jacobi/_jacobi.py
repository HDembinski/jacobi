"""Numerical computation of the Jacobi matrix with error estimates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


def jacobi(
    fn: Callable[..., ArrayLike],
    x: ArrayLike,
    *args: Any,
    diagonal: bool = False,
    method: int | None = None,
    mask: ArrayLike | None = None,
    rtol: float = 0,
    maxiter: int = 10,
    maxgrad: int = 3,
    step: tuple[float, float] | None = None,
    diagnostic: dict[str, Any] | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Return first derivative and its error estimate.

    Parameters
    ----------
    fn : Callable
        Function with the signature `fn(x, *args)`, where `x` is a number or a sequence
        of numbers and `*args` are optional auxiliary arguments. The function must
        return a number or a regular shape of numbers (ideally as a numpy array). The
        length of `x` can differ from the output sequence. Derivatives are only
        computed with respect to `x`, the auxiliary arguments are ignored.
    x : Number or array of numbers
        The derivative is computed with respect to `x`. If `x` is an array, the Jacobi
        matrix is computed with respect to each element of `x`.
    *args : tuple
        Additional arguments passed to the function.
    diagonal : boolean, optional
        If it is known that the Jacobian computed from the function has off-diagonal
        entries that are all zero, the calculation can be sped up significantly. Set
        this to true to only compute the diagonal entries of the Jacobi matrix, which
        are returned as a 1D array. This is faster and uses much less memory if the
        vector x is very large. Default is False.
    method : {-1, 0, 1} or None, optional
        Whether to compute central (0), forward (1) or backward derivatives (-1).
        The default (None) uses auto-detection.
    mask : array or None, optional
        If `x` is an array and `mask` is not None, compute the Jacobi matrix only for
        the part of the array selected by the mask.
    rtol : float, optional
        Relative tolerance for the derivative. The algorithm stops when this relative
        tolerance is reached. If 0 (the default), the algorithm iterates until the
        error estimate of the derivative does not improve further.
    maxiter : int, optional
        Maximum number of iterations of the algorithm.
    maxgrad : int, optional
        Maximum degree of the extrapolation polynomial.
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
    if maxiter <= 0:
        msg = "maxiter must be > 0"
        raise ValueError(msg)
    if maxgrad < 0:
        msg = "maxgrad must be >= 0"
        raise ValueError(msg)
    if step is not None:
        if not 0 < step[0] < 0.5:
            msg = "step[0] must be between 0 and 0.5"
            raise ValueError(msg)
        if not 0 < step[1] < 1:
            msg = "step[1] must be between 0 and 1"
            raise ValueError(msg)
    if method is not None and method not in (-1, 0, 1):
        msg = "method must be -1, 0, 1"
        raise ValueError(msg)

    xa = np.asarray(x, dtype=float)
    ma: NDArray[Any] | None = None
    if mask is not None:
        ma = np.asarray(mask)
        if ma.dtype != bool:
            msg = "mask must be a boolean array"
            raise ValueError(msg)
        if ma.shape != xa.shape:
            msg = "mask shape must match x shape"
            raise ValueError(msg)

    if xa.size == 0 or (ma is not None and not ma.any()):
        # nothing to compute, but the output shape is needed for the result
        _, fval = _wrap_function_if_needed(fn, fn(xa, *args))
        shape = np.shape(fval) if diagonal else np.shape(fval) + xa.shape
        return np.zeros(shape), np.zeros(shape)

    if diagonal:
        # TODO maybe solve this without introducing a wrapper function
        j, je = jacobi(
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
        if ma is not None:
            j[~ma] = 0.0
            je[~ma] = 0.0
        return j, je

    if diagnostic is not None:
        diagnostic["method"] = np.zeros(xa.size, dtype=np.int8)
        diagnostic["iteration"] = np.zeros(xa.size, dtype=np.uint8)
        diagnostic["residual"] = [[] for _ in range(xa.size)]

    f0: Any = None
    jac: NDArray[Any] | None = None
    err: NDArray[Any] | None = None
    it = np.nditer(xa, flags=["c_index", "multi_index"])
    while not it.finished:
        k = it.index
        kx = it.multi_index
        if ma is not None and not ma[kx]:
            it.iternext()
            continue
        xk = it[0]
        # if step is None, use optimal step sizes for central derivatives
        h = _steps(xk, step or (0.25, 0.5), maxiter)
        # if method is None, auto-detect for each x[k]
        fn, md, f0, r = _first(method, f0, fn, xa, kx, h[0], args)
        # f0 is not guaranteed to be set here and can be still None

        if md != 0 and step is None:
            # need different step sizes for forward derivatives to avoid overlap
            h = _steps(xk, (0.25, 0.125), maxiter)

        res = np.asarray(r, dtype=float)
        res_err = np.full_like(res, np.inf)
        todo = np.ones_like(res, dtype=bool)
        fd = [np.reshape(res.copy(), -1)]

        if jac is None or err is None:  # first iteration
            jac = np.zeros(res.shape + xa.shape, dtype=res.dtype)
            err = np.zeros(res.shape + xa.shape, dtype=res.dtype)
            if diagnostic is not None:
                diagnostic["call"] = np.zeros((res.size, xa.size), dtype=np.uint8)

        if diagnostic is not None:
            diagnostic["method"][k] = md
            diagnostic["call"][:, k] = 2 if md == 0 else 3

        for i in range(1, len(h)):
            fdi = np.asarray(_derive(md, f0, fn, xa, kx, h[i], args))
            fd.append(np.reshape(fdi, -1) if i == 1 else fdi[todo])
            if diagnostic is not None:
                diagnostic["call"][todo.reshape(-1), k] += 2
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
            sub_todo = rei < res_err[todo]
            todo1 = todo.copy()
            todo[todo1] = sub_todo
            res[todo] = ri[sub_todo]
            res_err[todo] = rei[sub_todo]

            # do not improve estimates further which meet the tolerance
            if rtol > 0:
                sub_todo &= rei > rtol * np.abs(ri)
                todo[todo1] = sub_todo

            if diagnostic is not None:
                re2 = res_err.copy()
                re2[todo1] = rei
                diagnostic["residual"][k].append(re2)

            if np.sum(todo) == 0:
                break

            # shrink previous vectors of estimates
            fd = [v[sub_todo] for v in fd]

        idx: tuple[Any, ...] = (..., *kx)
        jac[idx] = res
        err[idx] = res_err

        it.iternext()

    assert jac is not None
    assert err is not None
    return jac, err


def _steps(p: Any, step: tuple[float, float], maxiter: int) -> NDArray[Any]:
    h0, factor = step
    h = p * h0
    if h == 0:  # if p is NaN, h stays NaN
        h = h0
    return np.asarray(h * factor ** np.arange(maxiter))


def _derive(
    mode: int,
    f0: Any,
    f: Callable[..., Any],
    x: NDArray[Any],
    i: tuple[int, ...],
    h: Any,
    args: tuple[Any, ...],
) -> Any:
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


def _first(
    method: int | None,
    f0: Any,
    fn: Callable[..., Any],
    x: NDArray[Any],
    i: tuple[int, ...],
    h: Any,
    args: tuple[Any, ...],
) -> tuple[Callable[..., Any], int, Any, Any]:
    # This is the first derivative that we calculate.
    # This function is special because we collect a lot of diagnostic
    # information about the function for the remainder of the iterations.
    norm = 0.5 / h
    f1: Any = None
    f2: Any = None
    if method is None or method == 0:
        x1 = x.copy()
        x2 = x.copy()
        x1[i] -= h
        x2[i] += h
        f1 = fn(x1, *args)
        fn, f1 = _wrap_function_if_needed(fn, f1)
        f2 = fn(x2, *args)
        if method is None:
            if np.any(np.isnan(f1)):  # forward method
                method = 1
            elif np.any(np.isnan(f2)):  # backward method
                method = -1
            else:
                method = 0
    if method == 0:
        return fn, method, None, (f1 - f2) * norm
    if f0 is None:
        f0 = fn(x, *args)
        fn, f0 = _wrap_function_if_needed(fn, f0)
    if method == -1:
        h = -h
        norm = -norm
    if f1 is None:
        x1 = x.copy()
        x1[i] += h
        f1 = fn(x1, *args)
    elif method == 1:
        f1 = f2
    x2 = x.copy()
    x2[i] += 2 * h
    f2 = fn(x2, *args)
    return fn, method, f0, (-3 * f0 + 4 * f1 - f2) * norm


def _wrap_function_if_needed(
    fn: Callable[..., Any], fval: Any
) -> tuple[Callable[..., Any], Any]:
    if not isinstance(fval, float):
        try:
            fval_a = np.asarray(fval, dtype=float)
        except ValueError as e:
            msg = (
                "function return value cannot be converted into "
                "1D numpy array of floats"
            )
            raise ValueError(msg) from e
        if isinstance(fval, np.ndarray):
            # fn already returns arrays and needs no wrapper; wrapping it again
            # for every element of x exceeds the recursion limit for large x
            return fn, fval_a
        return lambda *args: np.asarray(fn(*args)), fval_a
    return fn, fval
