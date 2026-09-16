"""Numerical error propagation based on the Jacobi matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._jacobi import jacobi

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

__all__ = ["propagate"]


def propagate(
    fn: Callable[..., ArrayLike],
    x: ArrayLike,
    cov: ArrayLike,
    *args: ArrayLike,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """
    Numerically propagate the covariance of function inputs to function outputs.

    The function computes C' = J C J^T, where C is the covariance matrix of the input,
    C' the matrix of the output, and J is the Jacobi matrix of first derivatives of the
    mapping function fn. The Jacobi matrix is computed numerically.

    Parameters
    ----------
    fn : callable
        Function with the signature `fn(x, [y, ...])`, where `x` is a number or a
        sequence of numbers, likewise if other arguments are present they must have the
        same format. The function must return a number or a sequence of numbers (ideally
        as a numpy array). The length of `x` can differ from the output sequence. The
        function should accept more than one argument only if there are no
        correlations between these arguments. See example below for use cases.
    x : float or array-like with shape (N,)
        Input vector. An array-like is converted before passing it to the callable.
    cov : float or array-like with shape (N,) or shape (N, N)
        Covariance matrix of input vector. If the array is one-dimensional, it is
        interpreted as the diagonal of a covariance matrix with zero off-diagonal
        elements.
    *args : tuple
        If the function accepts several arguments that are mutually independent, it
        is possible to pass those values and covariance matrices pairwise, see examples.
    **kwargs : dict
        Extra arguments are passed to :func:`jacobi`.

    Returns
    -------
    y, ycov
        y is the result of fn(x).
        ycov is the propagated covariance matrix.
        ycov is a matrix, unless y is a number. In that case, ycov is also
        reduced to a number.

    Notes
    -----
    For callables `fn` which perform only element-wise computation, the jacobian is
    a diagonal matrix. This special case is detected and the computation optimised,
    although one can further speed up the computation by passing the argument
    `diagonal=True`.

    In this special case, error propagation works correctly even if the output of `fn`
    is NaN for some inputs.

    Examples
    --------
    General error propagation maps input vectors to output vectors.

    >>> def fn(x):
    ...     return x**2 + 1
    >>> x = [1, 2]
    >>> xcov = [[3, 1], [1, 4]]
    >>> y, ycov = propagate(fn, x, xcov)

    In the previous example, the function ``y = fn(x)`` treats all x values
    independently and the Jacobian computed from ``fn(x)`` has zero off-diagonal
    entries. In this case, one can speed up the calculation significantly with a special
    keyword.

    >>> y, ycov = propagate(fn, x, xcov, diagonal=True)

    This produces the same result, but is faster and uses less memory. If the function
    accepts several arguments, their uncertainties are treated as uncorrelated.

    >>> def fn(x, y):
    ...     return x + y
    >>> x = 1
    >>> y = 2
    >>> xcov = 2
    >>> ycov = 3
    >>> z, zcov = propagate(fn, x, xcov, y, ycov)

    Functions that accept several correlated arguments must be wrapped.

    >>> def fn(x, y):
    ...     return x + y
    >>> rho_xy = 0.5
    >>> cov_xy = rho_xy * (xcov * ycov) ** 0.5
    >>> r = [x, y]
    >>> rcov = [[xcov, cov_xy], [cov_xy, ycov]]
    >>> z, zcov = propagate(lambda r: fn(r[0], r[1]), r, rcov)

    See Also
    --------
    jacobi
    """
    if args:
        if len(args) % 2 != 0:
            msg = "number of extra positional arguments must be even"
            raise ValueError(msg)

        args_a = [np.asarray(arg) for arg in (x, cov, *args)]
        x_parts = args_a[::2]
        cov_parts = args_a[1::2]
        y_a = np.asarray(fn(*x_parts))
        return _propagate_independent(fn, y_a, x_parts, cov_parts, **kwargs)

    x_a = np.asarray(x)
    cov_a = np.asarray(cov)
    try:
        y_a = np.asarray(fn(x_a))
    except ValueError as e:
        msg = "function return value cannot be converted into numpy array"
        raise ValueError(msg) from e

    # TODO lift this limitation
    if x_a.ndim > 1:
        msg = "x must have dimension 0 or 1"
        raise ValueError(msg)

    # TODO lift this limitation
    if y_a.ndim > 1:
        msg = "function return value must have dimension 0 or 1"
        raise ValueError(msg)

    # TODO lift this limitation
    if cov_a.ndim > 2:
        msg = "cov must have dimension 0, 1, or 2"
        raise ValueError(msg)

    return _propagate(fn, y_a, x_a, cov_a, **kwargs)


def _propagate(
    fn: Callable[..., Any],
    y: NDArray[Any],
    x: NDArray[Any],
    xcov: NDArray[Any],
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    _check_x_xcov_compatibility(x, xcov)

    jac = jacobi(fn, x, **kwargs)[0]

    diagonal = kwargs.get("diagonal", False)

    if jac.ndim == 2:
        # Check if jacobian is diagonal, count NaN as zero.
        # This is important to speed up the product below and
        # to get the right answer for covariance matrices that
        # contain NaN values.
        jac = _try_reduce_jacobian(jac)
    elif not diagonal:
        jac = jac.reshape(y.size, x.size)

    ycov = _jac_cov_product(jac, xcov)

    if y.ndim == 0:
        ycov = np.squeeze(ycov)

    return y, ycov


def _propagate_independent(
    fn: Callable[..., Any],
    y: NDArray[Any],
    x_parts: list[NDArray[Any]],
    xcov_parts: list[NDArray[Any]],
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    ycov: NDArray[Any] | None = None

    mask = kwargs.get("mask")
    mask_parts = []
    if mask is None:
        kwargs2 = kwargs
    else:
        kwargs2 = kwargs.copy()
        for i, x in enumerate(x_parts):
            # this fails if mask is not indexable, but mask is always an array
            if np.shape(x) == np.shape(mask[i]):
                mask_parts.append(mask[i])
            elif np.shape(x) == np.shape(mask):
                mask_parts.append(mask)
            else:
                msg = "mask shapes do not match arguments"
                raise ValueError(msg)

    for i, x in enumerate(x_parts):
        wrapped = _fix_other_arguments(fn, x_parts, i)
        xcov = xcov_parts[i]

        if mask_parts:
            kwargs2["mask"] = mask_parts[i]

        yc = _propagate(wrapped, y, x, xcov, **kwargs2)[1]
        if ycov is None:
            ycov = np.array(yc)
        elif ycov.ndim == 2 and yc.ndim == 1:
            ycov[np.diag_indices_from(ycov)] += yc
        elif ycov.ndim == 1 and yc.ndim == 2:
            ycov = np.diag(ycov) + yc
        else:
            ycov = ycov + yc

    if ycov is None:
        msg = "at least one pair of value and covariance must be passed"
        raise ValueError(msg)

    return y, ycov


def _fix_other_arguments(
    fn: Callable[..., Any], x_parts: list[NDArray[Any]], i: int
) -> Callable[[NDArray[Any]], Any]:
    def wrapped(x: NDArray[Any]) -> Any:
        return fn(*x_parts[:i], x, *x_parts[i + 1 :])

    return wrapped


def _jac_cov_product(jac: NDArray[Any], xcov: NDArray[Any]) -> NDArray[Any]:
    # if jac or xcov are 1D, they represent diagonal matrices
    if xcov.ndim == 2:
        if jac.ndim == 2:
            return np.asarray(np.einsum("ij,kl,jl", jac, jac, xcov))
        if jac.ndim == 1:
            return np.asarray(np.einsum("i,j,ij -> ij", jac, jac, xcov))
        return np.asarray(jac**2 * xcov)
    assert xcov.ndim < 2
    if jac.ndim == 2:
        if xcov.ndim == 1:
            return np.asarray(np.einsum("ij,kj,j", jac, jac, xcov))
        assert xcov.ndim == 0  # xcov.ndim == 2 is already covered above
        return np.asarray(np.einsum("ij,kj", jac, jac) * xcov)
    assert jac.ndim < 2
    assert xcov.ndim < 2
    return np.asarray(xcov * jac**2)


def _check_x_xcov_compatibility(x: NDArray[Any], xcov: NDArray[Any]) -> None:
    if xcov.ndim > 0 and len(xcov) != (len(x) if x.ndim == 1 else 1):
        # this works for 1D and 2D xcov
        msg = "x and cov have incompatible shapes"
        raise ValueError(msg)


def _try_reduce_jacobian(jac: NDArray[Any]) -> NDArray[Any]:
    if jac.ndim != 2 or jac.shape[0] != jac.shape[1]:
        return jac
    # if jacobian contains only off-diagonal elements
    # that are zero or NaN, we reduce it to diagonal form
    ndv = _nodiag_view(jac)
    m = np.isnan(ndv)
    ndv[m] = 0
    if np.count_nonzero(ndv) == 0:
        return np.diag(jac)
    return jac


def _nodiag_view(a: NDArray[Any]) -> NDArray[Any]:
    # https://stackoverflow.com/a/43761941/ @Divakar
    m = a.shape[0]
    p, q = a.strides
    return np.lib.stride_tricks.as_strided(a[:, 1:], (m - 1, m), (p + q, q))
