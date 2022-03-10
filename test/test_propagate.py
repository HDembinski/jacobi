import numpy as np
from numpy.testing import assert_allclose
from jacobi import propagate
import pytest


def test_00():
    def fn(x):
        return x**2 + 5

    x = 2
    xcov = 3
    y, ycov = propagate(fn, x, xcov)
    assert_allclose(y, fn(x))
    assert_allclose(ycov, (2 * x) ** 2 * xcov)


def test_01():
    def fn(x):
        return np.ones(2) * x**2

    x = 2
    xcov = 3
    y, ycov = propagate(fn, x, xcov)
    assert_allclose(y, fn(x))
    jac = (2 * x) * np.ones((2, 1))
    xcov2 = np.atleast_2d(xcov)
    assert_allclose(ycov, np.linalg.multi_dot([jac, xcov2, jac.T]))

    with pytest.raises(ValueError):
        propagate(fn, x, np.ones(2))

    with pytest.raises(ValueError):
        propagate(fn, x, np.ones((2, 2)))


def test_10():
    def fn(x):
        return np.sum(x)

    x = np.arange(2)
    xcov = [[1, 0], [0, 1]]
    y, ycov = propagate(fn, x, xcov)
    assert_allclose(y, fn(x))
    jac = np.ones((1, len(x)))
    assert_allclose(ycov, np.linalg.multi_dot([jac, xcov, jac.T]))
    assert np.ndim(y) == 0
    assert np.ndim(ycov) == 0


def test_11():
    A = np.array([(1, 2), (3, 4), (5, 6)], dtype=float)

    def fn(x):
        return np.dot(A, x)

    x = np.arange(2)
    xcov = [[1, 0], [0, 1]]
    y, ycov = propagate(fn, x, xcov)
    assert_allclose(y, fn(x))
    jac = A
    assert_allclose(ycov, np.linalg.multi_dot([jac, xcov, jac.T]))
