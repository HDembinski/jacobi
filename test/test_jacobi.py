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


@pytest.mark.parametrize("kind", ("real", "complex"))
@pytest.mark.parametrize(
    "fn",
    [
        (lambda x: np.exp(x), lambda x: np.diagflat(np.exp(x))),
        (lambda x: x ** 2, lambda x: np.diagflat(2 * x)),
        (lambda x: np.ones_like(x), lambda x: np.diagflat(np.zeros_like(x))),
        (lambda x: x ** -1, lambda x: np.diagflat(-(x ** -2))),
        (lambda x: np.sqrt(1 - x), lambda x: np.diagflat(-0.5 / np.sqrt(1 - x))),
        (
            lambda x: np.mean(x ** 2, axis=1),
            lambda x: np.diagflat(np.mean(2 * x, axis=1)),
        ),
        (
            lambda x: np.outer(x, x),
            lambda x: np.ones(2)[:, np.newaxis] * 2 * x,
        ),
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
