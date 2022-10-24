import numpy as np
from numpy.testing import assert_allclose
from jacobi import propagate, jacobi
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
@pytest.mark.parametrize("diagonal", (False, True))
@pytest.mark.parametrize("len", (1, 2))
def test_cov_1d_2d(ndim, diagonal, len):
    def fn(x):
        return 2 * x

    x = [1, 2][:len]
    xcov = [3, 4][:len]
    if ndim == 2:
        xcov = np.diag(xcov)

    y, ycov = propagate(fn, x, xcov, diagonal=diagonal)

    assert np.ndim(ycov) == ndim

    assert_allclose(y, np.multiply(x, 2))
    assert_allclose(ycov, np.multiply(xcov, 4))


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


def test_on_nan():
    def fn(x):
        return x**2 + 1

    x = [1.0, np.nan, 2.0]
    xcov = [[3.0, np.nan, 1.0], [np.nan, np.nan, np.nan], [1.0, np.nan, 5.0]]

    y, ycov = propagate(fn, x, xcov, diagonal=True)

    y_ref = [2, np.nan, 5]
    assert_allclose(y, y_ref)

    jac = jacobi(fn, x, diagonal=True)[0]
    jac_ref = [2.0, np.nan, 4.0]
    assert_allclose(jac, jac_ref)

    # ycov_ref = jac @ np.array(xcov) @ jac.T
    ycov_ref = [[12, np.nan, 8], [np.nan, np.nan, np.nan], [8, np.nan, 80]]
    assert_allclose(ycov, ycov_ref)

    # propagate now detects the special case where jac is effectively diagonal
    # and does the equivalent of propagate(fn, x, xcov, diagonal=True), which
    # is nevertheless faster
    y2, ycov2 = propagate(fn, x, xcov)
    assert_allclose(y2, y_ref)
    assert_allclose(ycov2, ycov_ref)
