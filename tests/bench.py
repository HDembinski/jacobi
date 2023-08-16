from jacobi import jacobi
import pytest
import numpy as np

fn = ["0", "x ** 2", "x ** -1", "exp(x)", "sin(x)", "ones_like(x)", "mean(x ** 2)"]


@pytest.mark.parametrize("f", fn)
@pytest.mark.benchmark(group="1")
def test_jacobi_1(benchmark, f):
    f = eval(f"lambda x:{f}", np.__dict__)
    benchmark(jacobi, f, 0)


@pytest.mark.parametrize("f", fn)
@pytest.mark.benchmark(group="1000")
def test_jacobi_1000(benchmark, f):
    f = eval(f"lambda x:{f}", np.__dict__)
    x = np.linspace(0.1, 10, 1000)
    benchmark(jacobi, lambda p: f(x + p), 0)
