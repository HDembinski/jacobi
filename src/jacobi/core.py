import numpy as np
import typing as _tp

_T = _tp.TypeVar("_T")


class _Indexable(_tp.Iterable, _tp.Sized, _tp.Generic[_T]):
    """Indexable type for mypy."""

    def __getitem__(self, idx: int) -> _T:
        """Get item at index idx."""
        ...  # pragma: no cover


def _steps(p, step, maxiter):
    h0, factor = step
    h = p * h0
    if h == 0:
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


def jacobi(
    fn: _tp.Callable,
    x: _tp.Union[int, float, _tp.Sequence],
    *args,
    method: _tp.Optional[int] = None,
    mask: _tp.Optional[np.ndarray] = None,
    rtol: float = 0,
    maxiter: int = 10,
    maxgrad: int = 3,
    step: _tp.Optional[_tp.Tuple[float, float]] = None,
    diagnostic: _tp.Optional[dict] = None,
):
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
    if maxiter <= 0:
        raise ValueError("invalid value for keyword maxiter")
    if maxgrad < 0:
        raise ValueError("invalid value for keyword maxgrad")
    if step is not None:
        if not (0 < step[0] < 0.5):
            raise ValueError("invalid value for step[0]")
        if not (0 < step[1] < 1):
            raise ValueError("invalid value for step[1]")

    squeeze = np.ndim(x) == 0
    x = np.atleast_1d(x).astype(float)
    assert x.ndim == 1

    x_indices = np.arange(len(x))
    if mask is not None:
        x_indices = x_indices[mask]
    nx = len(x_indices)

    if isinstance(diagnostic, dict):
        diagnostic["method"] = np.zeros(nx, dtype=np.int8)
        diagnostic["iteration"] = np.zeros(len(x_indices), dtype=np.uint8)

    if method is not None and method not in (-1, 0, 1):
        raise ValueError("invalid value for keyword method")

    f0 = None
    jac = None
    err = None
    for ik, k in enumerate(x_indices):
        # if step is None, use optimal step sizes for central derivatives
        h = _steps(x[k], step or (0.25, 0.5), maxiter)
        # if method is None, auto-detect for each x[k]
        md, f0, r = _first(method, f0, fn, x, k, h[0], args)

        if diagnostic:
            diagnostic["method"][ik] = md

        if md != 0 and step is None:
            # optimal step sizes for forward derivatives
            h = _steps(x[k], (0.125, 0.125), maxiter)

        r_shape = np.shape(r)
        r = np.reshape(r, -1)
        nr = len(r)
        re = np.full(nr, np.inf)
        todo = np.ones(nr, dtype=bool)
        fd = [r]

        if jac is None:
            jac = np.empty(r_shape + (nx,), dtype=r.dtype)
            err = np.empty(r_shape + (nx,), dtype=r.dtype)
            if diagnostic:
                diagnostic["call"] = np.zeros((nr, nx), dtype=np.uint8)

        if diagnostic:
            diagnostic["call"][:, ik] = 2 if md == 0 else 3

        for i in range(1, len(h)):
            fdi = _derive(md, f0, fn, x, k, h[i], args)
            fdi = np.reshape(fdi, -1)
            fd.append(fdi if i == 1 else fdi[todo])
            if diagnostic:
                diagnostic["call"][todo, ik] += 2
                diagnostic["iteration"][ik] += 1

            # polynomial fit with one extra degree of freedom
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

            # update estimates that have significantly smaller error
            sub_todo = rei < re[todo]
            todo1 = todo.copy()
            todo[todo1] = sub_todo
            r[todo] = ri[sub_todo]
            re[todo] = rei[sub_todo]

            # do not improve estimates further which meet the tolerance
            if rtol > 0:
                sub_todo &= rei > rtol * np.abs(ri)
                todo[todo1] = sub_todo

            if np.sum(todo) == 0:
                break

            # shrink previous vectors of estimates
            fd = [fdi[sub_todo] for fdi in fd]

        jac[..., ik] = r.reshape(r_shape)
        err[..., ik] = re.reshape(r_shape)

    if diagnostic:
        diagnostic["call"].shape = r_shape + (nx,)

    if squeeze:
        if diagnostic:
            diagnostic["call"] = np.squeeze(diagnostic["call"])
        jac = np.squeeze(jac)
        err = np.squeeze(err)

    return jac, err


def propagate(
    fn: _tp.Callable,
    x: _tp.Union[float, _Indexable[float]],
    cov: _tp.Union[float, _Indexable[float], _Indexable[_Indexable[float]]],
    **kwargs,
) -> _tp.Tuple[np.ndarray, np.ndarray]:
    """
    Numerically propagates the covariance of function inputs to function outputs.

    The function computes C' = J C J^T, where C is the covariance matrix of the input,
    C' the matrix of the output, and J is the Jacobi matrix of first derivatives of the
    mapping function fn. The Jacobi matrix is computed numerically.

    Parameters
    ----------
    fn: callable
        Function that computes y = fn(x). x and y are each allowed to be scalars or
        one-dimensional arrays.
    x: float or array-like with shape (N,)
        Input vector.
    cov: float or array-like with shape (N,) or shape(N, N)
        Covariance matrix of input vector. If the array is one-dimensional, it is
        interpreted as the diagonal of a covariance matrix with zero off-diagonal
        elements.
    **kwargs:
        Extra arguments are passed to :func:`jacobi`.

    Returns
    -------
    y, ycov
        y is the result of fn(x)
        ycov is the propagated covariance matrix.
    """
    x = np.array(x)
    y = fn(x)
    jac = jacobi(fn, x, **kwargs)[0]

    x_nd = np.ndim(x)
    y_nd = np.ndim(y)
    x_len = len(x) if x_nd == 1 else 1
    y_len = len(y) if y_nd == 1 else 1

    jac_nd = np.ndim(jac)
    if jac_nd == 0:
        jac = np.atleast_2d(jac)
    elif jac_nd == 1:
        jac = jac.reshape((y_len, x_len))

    xcov = np.atleast_1d(cov)
    xcov_nd = np.ndim(xcov)

    if xcov.shape[0] != x_len:
        raise ValueError("x and cov have incompatible shapes")

    ycov = np.einsum("il,kl,l" if xcov_nd == 1 else "ij,kl,jl", jac, jac, xcov)
    return y, ycov
