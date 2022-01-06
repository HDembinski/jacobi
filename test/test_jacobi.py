from numpy.testing import assert_allclose, assert_equal
import numpy as np
import pytest
from jacobi import jacobi


def test_squeeze():
    # also tests promotion of integer argument to float
    y, ye = jacobi(np.exp, 0)
    assert_allclose(y, 1)
    assert_allclose(ye, 0, atol=1e-10)


def f5(x):
    return np.mean(x ** 2)


def fd5(x):
    return 2 * x / len(x)


def f6(x):
    return np.outer(x, x)


def fd6(r):
    assert len(r) == 3
    x, y, z = r
    r = np.empty((3, 3, 3))
    r[..., 0] = [
        [2 * x, y, z],
        [y, 0, 0],
        [z, 0, 0],
    ]
    r[..., 1] = [
        [0, x, 0],
        [x, 2 * y, z],
        [0, z, 0],
    ]
    r[..., 2] = [
        [0, 0, x],
        [0, 0, y],
        [x, y, 2 * z],
    ]
    return r


@pytest.mark.parametrize(
    "fn",
    [
        (lambda x: 1.0, lambda x: 0.0),
        (lambda x: np.exp(x), lambda x: np.diagflat(np.exp(x))),
        (lambda x: x ** 2, lambda x: np.diagflat(2 * x)),
        (lambda x: np.ones_like(x), lambda x: np.diagflat(np.zeros_like(x))),
        (lambda x: x ** -1, lambda x: np.diagflat(-(x ** -2))),
        (lambda x: (x + 1) ** 0.5, lambda x: np.diagflat(0.5 * (x + 1) ** -0.5)),
        (f5, fd5),
        (f6, fd6),
    ],
)
def test_jacobi(fn):
    x = np.array([1, 2, 3], dtype=float)
    f, fd = fn
    y, ye = jacobi(f, x)
    assert_allclose(y, fd(x))
    assert_allclose(ye, np.zeros_like(y), atol=1e-10)


def test_abs_at_zero():
    fp, fpe = jacobi(np.abs, 0)
    assert_equal(fp, 0)
    assert_equal(fpe, 0)


def test_method_auto():
    d = {}
    fp, fpe = jacobi(lambda x: x, 0, diagnostic=d)
    assert_equal(d["method"], [0])
    assert_allclose(fp, 1)
    assert_allclose(fpe, 0, atol=1e-10)

    fp, fpe = jacobi(lambda x: x if x >= 0 else np.nan, 0, diagnostic=d)
    assert_equal(d["method"], [1])
    assert_allclose(fp, 1)
    assert_allclose(fpe, 0, atol=1e-10)

    fp, fpe = jacobi(lambda x: x if x <= 0 else np.nan, 0, diagnostic=d)
    assert_equal(d["method"], [-1])
    assert_allclose(fp, 1)
    assert_allclose(fpe, 0, atol=1e-10)


@pytest.mark.parametrize("method", (-1, 0, 1))
def test_method(method):
    d = {}
    fp, fpe = jacobi(lambda x: x, 0, method=method, diagnostic=d)
    assert_equal(d["method"], [method])
    assert_allclose(fp, 1)
    assert_allclose(fpe, 0, atol=1e-10)

    fp, fpe = jacobi(lambda x: x if x >= 0 else np.nan, 0, method=method, diagnostic=d)
    assert_equal(d["method"], [method])
    if method == 1:
        assert_allclose(fp, 1)
        assert_allclose(fpe, 0, atol=1e-10)
    else:
        assert_equal(fp, np.nan)
        assert_equal(fpe, np.inf)

    fp, fpe = jacobi(lambda x: x if x <= 0 else np.nan, 0, method=method, diagnostic=d)
    assert_equal(d["method"], [method])
    if method == -1:
        assert_allclose(fp, 1)
        assert_allclose(fpe, 0, atol=1e-10)
    else:
        assert_equal(fp, np.nan)
        assert_equal(fpe, np.inf)
