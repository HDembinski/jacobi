from jacobi import jacobi, propagate
import numpy as np
import pytest


@pytest.mark.parametrize("diagonal", (False, True))
def test_jacobi(diagonal, benchmark):
    x = np.linspace(0, 1, 100)

    def fn(x):
        return x**2 + 3 * x + 1

    def run():
        jacobi(fn, x, diagonal=diagonal)

    benchmark(run)


@pytest.mark.parametrize("diagonal", (False, True))
def test_propagate(diagonal, benchmark):
    x = np.linspace(0, 1, 100)

    def fn(x):
        return x**2 + 3 * x + 1

    def run():
        propagate(fn, x, x, diagonal=diagonal)

    benchmark(run)


@pytest.mark.parametrize("diagonal", (False, True))
def test_propagate_2dcov(diagonal, benchmark):
    x = np.linspace(0, 1, 100)

    def fn(x):
        return x**2 + 3 * x + 1

    def run():
        propagate(fn, x, np.diag(x), diagonal=diagonal)

    benchmark(run)
