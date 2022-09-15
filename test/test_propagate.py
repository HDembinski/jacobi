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
    for xcov in ([1, 1], [[1, 0], [0, 1]]):

        y, ycov = propagate(fn, x, xcov)
        assert_allclose(y, fn(x))
        jac = np.ones((1, len(x)))

        if np.ndim(xcov) == 1:
            xcov2 = np.zeros((2, 2))
            for i in range(2):
                xcov2[i, i] = xcov[i]
        else:
            xcov2 = xcov
        assert_allclose(ycov, np.linalg.multi_dot([jac, xcov2, jac.T]))
        assert np.ndim(y) == 0
        assert np.ndim(ycov) == 0


def test_11():
    A = np.array([(1, 2), (3, 4), (5, 6)], dtype=float)

    def fn(x):
        return np.dot(A, x)

    x = np.arange(2)

    for xcov in ([1, 1], [[1, 0], [0, 1]]):
        y, ycov = propagate(fn, x, xcov)
        assert_allclose(y, fn(x))
        jac = A
        if np.ndim(xcov) == 1:
            xcov2 = np.zeros((2, 2))
            for i in range(2):
                xcov2[i, i] = xcov[i]
        else:
            xcov2 = xcov
        assert_allclose(ycov, np.linalg.multi_dot([jac, xcov2, jac.T]))


@pytest.mark.parametrize("ndim", (1, 2))
def test_cov_1d_2d(ndim):
    def fn(x):
        return x

    x = [1, 2]
    xcov_1d = [3, 4]
    xcov_2d = np.diag(xcov_1d)

    y, ycov = propagate(fn, x, xcov_1d if ndim == 1 else xcov_2d)

    assert np.ndim(ycov) == 2

    assert_allclose(y, x)
    assert_allclose(ycov, xcov_2d)


def test_two_arguments_1():
    def fn1(x, y):
        return (x - y) / (x + y)

    x = 1
    xcov = 2
    y = 3
    ycov = 4

    z, zcov = propagate(fn1, x, xcov, y, ycov)
    assert z.ndim == 0
    assert zcov.ndim == 0

    def fn2(r):
        return fn1(r[0], r[1])

    r = [x, y]
    rcov = np.diag([xcov, ycov])

    z_ref, zcov_ref = propagate(fn2, r, rcov)

    assert_allclose(z_ref, z)
    assert_allclose(zcov_ref, zcov)


def test_two_arguments_2():
    def fn1(x, y):
        return np.concatenate([x, np.atleast_1d(y)])

    x = [1, 2]
    xcov = [2, 3]
    y = 3
    ycov = 4

    z, zcov = propagate(fn1, x, xcov, y, ycov)

    def fn2(r):
        return fn1(r[:2], r[2])

    r = [*x, y]
    rcov = np.diag([*xcov, ycov])

    z_ref, zcov_ref = propagate(fn2, r, rcov)

    assert_allclose(z, z_ref)
    assert_allclose(zcov, zcov_ref)


def test_bad_number_of_arguments():
    with pytest.raises(ValueError, match="number of extra"):
        propagate(lambda x: x, 1, 2, 3)


def test_bad_input_dimensions():
    def fn(x):
        return x

    with pytest.raises(ValueError):
        propagate(fn, [[1]], 1)

    with pytest.raises(ValueError):
        propagate(fn, 1, [[[1]]])


def test_diagonal_1():
    def fn(x):
        return x**2 + 3

    def fprime(x):
        x = np.atleast_1d(x)
        return 2 * x

    x = 2
    xcov = 3

    y, ycov = propagate(fn, x, xcov, diagonal=True)

    assert ycov.ndim == 0
    assert_allclose(ycov, fprime(x) ** 2 * xcov)


def test_diagonal_2():
    def fn(x):
        return x**2 + 3

    def fprime(x):
        x = np.atleast_1d(x)
        return 2 * x

    x = [1, 2]
    xcov = [3, 4]

    y, ycov = propagate(fn, x, xcov, diagonal=True)

    x_a = np.atleast_1d(x)
    assert ycov.ndim == 1
    assert_allclose(y, fn(x_a))

    jac = np.diag(fprime(x_a))
    cov_a = np.diag(xcov)
    ycov_ref = np.linalg.multi_dot((jac, cov_a, jac.T))
    assert ycov_ref[0, 1] == 0
    assert ycov_ref[1, 0] == 0
    assert_allclose(ycov, np.diag(ycov_ref))


def test_diagonal_3():
    def fn(x):
        return x**2 + 1

    x = [1, 2]
    xcov = [[3, 1], [1, 4]]

    y, ycov = propagate(fn, x, xcov, diagonal=True)

    y_ref, ycov_ref = propagate(fn, x, xcov)

    assert_allclose(y, y_ref)
    assert_allclose(ycov, ycov_ref)
