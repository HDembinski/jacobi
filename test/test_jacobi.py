from numpy.testing import assert_allclose
import numpy as np
from pytest import approx
import pytest

deps = np.finfo(np.float64).eps
feps = np.finfo(np.float32).eps


@pytest.mark.parametrize("kind", ("real", "complex"))
def test_jacobi_squeeze(kind):
    if kind == "real":
        from jacobi.real_step import jacobi
    else:
        from jacobi.complex_step import jacobi

    # also tests promotion of integer argument to float
    y, ye = jacobi(np.exp, 0)
    assert y == approx(1)
    assert ye == approx(0, abs=1e-6)


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


@pytest.mark.parametrize("kind", ("real", "complex"))
@pytest.mark.parametrize(
    "fn",
    [
        (lambda x: np.exp(x), lambda x: np.diagflat(np.exp(x))),
        (lambda x: x ** 2, lambda x: np.diagflat(2 * x)),
        (lambda x: np.ones_like(x), lambda x: np.diagflat(np.zeros_like(x))),
        (lambda x: x ** -1, lambda x: np.diagflat(-(x ** -2))),
        (lambda x: (x + 1) ** 0.5, lambda x: np.diagflat(0.5 * (x + 1) ** -0.5)),
        (f5, fd5),
        (f6, fd6),
    ],
)
def test_jacobi(kind, fn):
    if kind == "real":
        from jacobi.real_step import jacobi
    else:
        from jacobi.complex_step import jacobi

    x = np.array([1, 2, 3], dtype=float)
    f, fd = fn
    y, ye = jacobi(f, x)
    assert_allclose(y, fd(x))
    assert_allclose(ye, np.zeros_like(y), atol=1e-10)
