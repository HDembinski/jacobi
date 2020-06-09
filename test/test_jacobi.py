from numpy.testing import assert_allclose
import numpy as np
from pytest import approx
import pytest

deps = np.finfo(np.float64).eps
feps = np.finfo(np.float32).eps


def f1(x):
    return x ** 2


def f2(x):
    return np.zeros_like(x)


def f3(x):
    y = np.zeros_like(x)
    y[0] = np.nan
    return y


def f4(x):
    x = np.asarray(x)
    return np.sqrt(1.0 - x)


def df4(x):
    return -0.5 / f4(x)


def f5(x):
    return np.ones(2)[:, np.newaxis] * x ** 2


def f6(x):
    x = np.asarray(x)
    return np.mean(x ** 2, axis=1)


def df6(x):
    x = np.asarray(x)
    return np.mean(2 * x, axis=1)


def test_complex_step_jacobi():
    from jacobi.complex_step import jacobi

    assert jacobi(np.exp, 0) == approx(1)
    assert jacobi(np.exp, 1) == approx(np.exp(1))

    y = jacobi(np.exp, [1, 2, 3])
    assert_allclose(y, np.exp([1, 2, 3]))

    y = jacobi(np.sqrt, [0, 1, 2, 0])
    assert_allclose(y[1:-1], [0.5, 0.5 * 2 ** -0.5])
    # y[0] and y[-1] are formally infinite
    assert_allclose((1.0 / y[0], 1.0 / y[-1]), (0, 0), atol=1e-6)

    y = jacobi(f1, [1, 2, 3])
    assert_allclose(y, [2, 4, 6])

    y = jacobi(f2, [1, 2, 3])
    assert_allclose(y, [0, 0, 0])

    y = jacobi(f3, [1, 2, 3])
    assert_allclose(y, [0, 0, 0])  # gives 0 instead of nan

    y = jacobi(f5, [1, 2, 3])
    assert_allclose(y, [[2, 4, 6], [2, 4, 6]])

    y = jacobi(f6, [[1, 2, 3], [2, 3, 4]])
    assert_allclose(y, df6([[1, 2, 3], [2, 3, 4]]))


def test_real_step_central():
    from jacobi.real_step import central

    y, ye = central(np.exp, 0, return_error=True)
    assert y == approx(1)
    assert ye == approx(0, abs=1e-6)

    y = central(np.exp, 1, return_error=False)
    assert y == approx(np.exp(1))

    y, ye = central(np.exp, [1, 2, 3], return_error=True)
    assert_allclose(y, np.exp([1, 2, 3]))
    assert_allclose(ye, [0, 0, 0], atol=1e-6)

    y, ye = central(f1, [1, 2, 3], return_error=True)
    assert_allclose(y, [2, 4, 6])
    assert_allclose(ye, [0, 0, 0], atol=1e-6)

    y, ye = central(f2, [1, 2, 3], return_error=True)
    assert_allclose(y, [0, 0, 0])
    assert_allclose(ye, [0, 0, 0])

    y, ye = central(f3, [1, 2, 3], return_error=True)
    assert_allclose(y, [np.nan, 0, 0])
    assert_allclose(ye, [np.nan, 0, 0])

    with np.errstate(invalid="ignore"):
        y, ye = central(np.sqrt, [0, 1, 2, 0], return_error=True)
    assert_allclose(y, [np.nan, 0.5, 0.5 * 2.0 ** -0.5, np.nan])
    assert_allclose(ye, [np.nan, 0, 0, np.nan], atol=1e-6)

    y, ye = central(f5, [1, 2, 3], return_error=True)
    assert_allclose(y, [[2, 4, 6], [2, 4, 6]])
    assert_allclose(ye, [[0, 0, 0], [0, 0, 0]], atol=1e-6)

    y = central(f6, [[1, 2, 3], [2, 3, 4]])
    assert_allclose(y, df6([[1, 2, 3], [2, 3, 4]]))


@pytest.mark.parametrize("dir", (-1, 1))
def test_real_step_forward(dir):
    from jacobi.real_step import forward

    y, ye = forward(np.exp, 0, return_error=True, dir=dir)
    assert y == approx(1)
    assert ye == approx(0, abs=1e-5)

    y = forward(np.exp, 1, return_error=False, dir=dir)
    assert y == approx(np.exp(1))

    y, ye = forward(np.exp, [1, 2, 3], return_error=True, dir=dir)
    assert_allclose(y, np.exp([1, 2, 3]), atol=1e-6)
    assert_allclose(ye, [0, 0, 0], atol=1e-4)

    y, ye = forward(f1, [1, 2, 3], return_error=True, dir=dir)
    assert_allclose(y, [2, 4, 6], atol=1e-6)
    assert_allclose(ye, [0, 0, 0], atol=1e-4)

    y, ye = forward(f2, [1, 2, 3], return_error=True, dir=dir)
    assert_allclose(y, [0, 0, 0])
    assert_allclose(ye, [0, 0, 0])

    y, ye = forward(f3, [1, 2, 3], return_error=True, dir=dir)
    assert_allclose(y, [np.nan, 0, 0])
    assert_allclose(ye, [np.nan, 0, 0])

    with np.errstate(invalid="ignore"):
        y, ye = forward(np.sqrt, [0, 1, 2, 0], return_error=True, dir=dir)
    assert_allclose(y[1:-1], [0.5, 0.5 * 2.0 ** -0.5], atol=1e-7)
    assert_allclose(ye[1:-1], [0, 0], atol=1e-5)
    if dir == 1:
        # y[0] and y[-1] are formally infinite
        assert_allclose((1.0 / y[0], 1.0 / y[-1]), (0, 0), atol=1e-4)
    else:
        assert np.isnan(y[0]) and np.isnan(y[-1])

    with np.errstate(invalid="ignore"):
        y, ye = forward(f4, [0.1, 0.5, 1], return_error=True, dir=dir)
    assert_allclose(y[:2], df4([0.1, 0.5]), atol=1e-5)
    assert_allclose(ye[:2], [0, 0], atol=1e-4)
    if dir == -1:
        assert 1 / y[2] < 1 / ye[2]  # y[2] is formally infinite
    else:
        assert np.isnan(y[2])

    y, ye = forward(f5, [1, 2, 3], return_error=True, dir=dir)
    assert_allclose(y, [[2, 4, 6], [2, 4, 6]], atol=1e-5)
    assert_allclose(ye, [[0, 0, 0], [0, 0, 0]], atol=1e-5)


def test_real_step_jacobi():
    from jacobi.real_step import jacobi

    y, ye = jacobi(np.exp, 0, return_error=True)
    assert y == approx(1)
    assert ye == approx(0, abs=1e-6)

    y = jacobi(np.exp, 1, return_error=False)
    assert y == approx(np.exp(1))

    y, ye = jacobi(np.exp, [1, 2, 3], return_error=True)
    assert_allclose(y, np.exp([1, 2, 3]))
    assert_allclose(ye, [0, 0, 0], atol=1e-6)

    y, ye = jacobi(f1, [1, 2, 3], return_error=True)
    assert_allclose(y, [2, 4, 6])
    assert_allclose(ye, [0, 0, 0], atol=1e-6)

    y, ye = jacobi(f2, [1, 2, 3], return_error=True)
    assert_allclose(y, [0, 0, 0])
    assert_allclose(ye, [0, 0, 0])

    y, ye = jacobi(f3, [1, 2, 3], return_error=True)
    assert_allclose(y, [np.nan, 0, 0])
    assert_allclose(ye, [np.nan, 0, 0])

    y, ye = jacobi(np.sqrt, [0, 1, 2], return_error=True)
    assert_allclose(y[1:], [0.5, 0.5 * 2.0 ** -0.5])
    assert_allclose(ye[1:], [0, 0], atol=1e-6)
    assert 1 / y[0] < 1 / ye[0]  # y[0] is formally infinite

    y, ye = jacobi(f4, [0.1, 0.5, 1], return_error=True)
    assert_allclose(y[:2], df4([0.1, 0.5]), atol=1e-7)
    assert_allclose(ye[:2], [0, 0], atol=1e-6)
    assert 1 / y[2] < 1 / ye[2]  # y[2] is formally infinite

    y, ye = jacobi(f5, [1, 2, 3], return_error=True)
    assert_allclose(y, [[2, 4, 6], [2, 4, 6]])
    assert_allclose(ye, [[0, 0, 0], [0, 0, 0]], atol=1e-6)


@pytest.mark.parametrize("return_error", (False, True))
def test_real_step_jacobi_wide(return_error):
    from jacobi.real_step import jacobi

    x = np.linspace(-3, 50)
    out = jacobi(np.exp, x, return_error=return_error)
    if return_error:
        y, ye = out
    else:
        y = out
    assert_allclose(y, np.exp(x))
