from typing import Callable, Union, Tuple, List
from ._typing import Indexable
from ._jacobi import jacobi
import numpy as np


__all__ = ["propagate"]


def propagate(
    fn: Callable,
    x: Union[float, Indexable[float]],
    cov: Union[float, Indexable[float], Indexable[Indexable[float]]],
    *args,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerically propagates the covariance of function inputs to function outputs.

    The function computes C' = J C J^T, where C is the covariance matrix of the input,
    C' the matrix of the output, and J is the Jacobi matrix of first derivatives of the
    mapping function fn. The Jacobi matrix is computed numerically.

    Parameters
    ----------
    fn: callable
        Function that computes r = fn(x, [y, ...]). The arguments of the function are
        each allowed to be scalars or one-dimensional arrays. If the function accepts
        several arguments, their uncertainties are treated as uncorrelated.
        Functions that accept several correlated arguments must be wrapped, see examples.
        The result of the function may be a scalar or a one-dimensional array with a
        different lenth as the input.
    x: float or array-like with shape (N,)
        Input vector. An array-like is converted before passing it to the callable.
    cov: float or array-like with shape (N,) or shape(N, N)
        Covariance matrix of input vector. If the array is one-dimensional, it is
        interpreted as the diagonal of a covariance matrix with zero off-diagonal
        elements.
    *args:
        If the function accepts several arguments that are mutually independent, these
        is possible to pass those values and covariance matrices pairwise, see examples.
    **kwargs:
        Extra arguments are passed to :func:`jacobi`.

    Returns
    -------
    y, ycov
        y is the result of fn(x).
        ycov is the propagated covariance matrix.
        If ycov is a matrix, unless y is a number. In that case, ycov is also
        reduced to a number.

    Notes
    -----
    For callables `fn` which perform only element-wise computation, the jacobian is
    a diagonal matrix. This special case is detected and the computation optimised,
    although can further speed up the computation by passing the argumet `diagonal=True`.

    In this special case, error propagation works correctly even if the output of `fn`
    is NaN for some inputs.

    Examples
    --------
    General error propagation maps input vectors to output vectors::

        def fn(x):
            return x ** 2 + 1

        x = [1, 2]
        xcov = [[3, 1],
                [1, 4]]

        y, ycov = propagate(fn, x, xcov)

    In the previous example, the function y = fn(x) treats all x values independently,
    so the Jacobian computed from fn(x) has zero off-diagonal entries. In this case,
    one can speed up the calculation significantly with a special keyword::

        # same result as before, but faster and uses much less memory
        y, ycov = propagate(fn, x, xcov, diagonal=True)

    If the function accepts several arguments, their uncertainties are treated as
    uncorrelated::

        def fn(x, y):
            return x + y

        x = 1
        y = 2
        xcov = 2
        ycov = 3

        z, zcov = propagate(fn, x, xcov, y, ycov)

    Functions that accept several correlated arguments must be wrapped::

        def fn(x, y):
            return x + y

        x = 1
        y = 2
        sigma_x = 3
        sigma_y = 4
        rho_xy = 0.5

        r = [x, y]
        cov_xy = rho_xy * sigma_x * sigma_y
        rcov = [[sigma_x ** 2, cov_xy], [cov_xy, sigma_y ** 2]]

        def fn_wrapped(r):
            return fn(r[0], r[1])

        z, zcov = propagate(fn_wrapped, r, rcov)

    See Also
    --------
    jacobi
    """
    if args:
        if len(args) % 2 != 0:
            raise ValueError("number of extra positional arguments must be even")

        args_a = [np.asarray(_) for _ in ((x, cov) + args)]
        x_parts = args_a[::2]
        cov_parts = args_a[1::2]
        y_a = np.asarray(fn(*x_parts))
        return _propagate_independent(fn, y_a, x_parts, cov_parts, **kwargs)

    x_a = np.asarray(x)
    cov_a = np.asarray(cov)
    y_a = np.asarray(fn(x_a))

    # TODO lift this limitation
    if x_a.ndim > 1:
        raise ValueError("x must have dimension 0 or 1")

    # TODO lift this limitation
    if y_a.ndim > 1:
        raise ValueError("f(x) must have dimension 0 or 1")

    # TODO lift this limitation
    if cov_a.ndim > 2:
        raise ValueError("cov must have dimension 0, 1, or 2")

    return _propagate(fn, y_a, x_a, cov_a, **kwargs)


def _propagate(fn: Callable, y: np.ndarray, x: np.ndarray, xcov: np.ndarray, **kwargs):
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
        jac.shape = (y.size, x.size)

    ycov = _jac_cov_product(jac, xcov)

    if y.ndim == 0:
        ycov = np.squeeze(ycov)

    return y, ycov


def _propagate_independent(
    fn: Callable,
    y: np.ndarray,
    x_parts: List[np.ndarray],
    xcov_parts: List[np.ndarray],
    **kwargs,
):
    ycov: Union[float, np.ndarray] = 0

    for i, x in enumerate(x_parts):
        rest = x_parts[:i] + x_parts[i + 1 :]

        def wrapped(x):
            args = rest[:i] + [x] + rest[i:]
            return fn(*args)

        xcov = xcov_parts[i]

        yc = _propagate(wrapped, y, x, xcov)[1]
        if np.ndim(ycov) == 2 and yc.ndim == 1:
            for i, yci in enumerate(yc):
                ycov[i, i] += yci  # type:ignore
        else:
            ycov += yc

    return y, ycov


def _jac_cov_product(jac: np.ndarray, xcov: np.ndarray):
    # if jac or xcov are 1D, they represent diagonal matrices
    if xcov.ndim == 2:
        if jac.ndim == 2:
            return np.einsum("ij,kl,jl", jac, jac, xcov)
        if jac.ndim == 1:
            return np.einsum("i,j,ij -> ij", jac, jac, xcov)
        return jac**2 * xcov
    assert xcov.ndim < 2
    if jac.ndim == 2:
        if xcov.ndim == 1:
            return np.einsum("ij,kj,j", jac, jac, xcov)
        assert xcov.ndim == 0  # xcov.ndim == 2 is already covered above
        return np.einsum("ij,kj", jac, jac) * xcov
    assert jac.ndim < 2 and xcov.ndim < 2
    return xcov * jac**2


def _check_x_xcov_compatibility(x: np.ndarray, xcov: np.ndarray):
    if xcov.ndim > 0 and len(xcov) != (len(x) if x.ndim == 1 else 1):
        # this works for 1D and 2D xcov
        raise ValueError("x and cov have incompatible shapes")


def _try_reduce_jacobian(jac: np.ndarray):
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


def _nodiag_view(a: np.ndarray):
    # https://stackoverflow.com/a/43761941/ @Divakar
    m = a.shape[0]
    p, q = a.strides
    return np.lib.stride_tricks.as_strided(a[:, 1:], (m - 1, m), (p + q, q))
