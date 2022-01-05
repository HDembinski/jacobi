from jacobi import jacobi
import pytest
import numpy as np

fn = [
    lambda p, x: (x + p) ** 2,
    lambda p, x: (x + p) ** -1,
    lambda p, x: np.exp(x + p),
    lambda p, x: np.ones_like(x + p),
    lambda p, x: (x + p + 1) ** 0.5,
    lambda p, x: np.mean((x + p) ** 2),
    lambda p, x: np.outer(x + p, x + p),
]


def ja(f, x):
    return


@pytest.mark.parametrize("f", fn)
@pytest.mark.parametrize("n", (1, 10, 100))
def test_jacobi(benchmark, f, n):
    x = np.linspace(0.1, 10, n)
    benchmark(lambda: jacobi(f, 0, x))
