from numpy.testing import assert_allclose
import numpy as np
from pytest import approx


def test_complex_step():
    from jacobi.complex_step import jacobi

    def f1(x):
        return x ** 2

    y = jacobi(f1, [1, 2, 3])
    assert_allclose(y, [2, 4, 6])

    y = jacobi(f1, 0)
    assert y == 0

    y = jacobi(f1, 1)
    assert y == approx(2)

    def f2(x):
        return np.ones(2)[:, np.newaxis] * x ** 2

    y = jacobi(f2, [1, 2, 3])

    assert_allclose(y, [[2, 4, 6], [2, 4, 6]])


def test_real_step():
    from jacobi.real_step import jacobi
